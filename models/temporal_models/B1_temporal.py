"""
B1-NoRelations-temporal Description:

Temporal version of B1 to track effective of relation layer modeling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import argparse
import torch.nn as nn
import albumentations as A
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchinfo import summary
from utils import load_config, Group_Activity_DataSet, group_activity_labels, model_eval

class PersonActivityClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PersonActivityClassifier, self).__init__()
        
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        b, bb, c, h, w = x.shape  # x.shape => batch, bbox, channals , hight, width
        x = x.view(b*bb, c, h, w) # (batch * bbox, c, h, w)
        x = self.resnet50(x)      # (batch * bbox, 2048, 1 , 1)
        x = x.view(b*bb, -1)      # (batch * bbox, 2048)
        x = self.fc(x)            # (batch * bbox, num_class)          
        return x

class GroupActivityClassifer(nn.Module):
    def __init__(self, person_feature_extraction, num_classes):
        super(GroupActivityClassifer, self).__init__()
   
        self.feature_extraction = nn.Sequential(*list(person_feature_extraction.resnet50.children())[:-1])

        for param in self.feature_extraction.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 1024))  # [6, 2048] -> [1, 1024]
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=1024,
            batch_first=True
        ) 

        self.fc = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        b, bb, seq , c, h, w = x.shape # batch, bbox, seq, channals, hight, width
        x = x.view(b*bb*seq, c, h, w)  # (b*bb*seq, c, h, w)
        x = self.feature_extraction(x) # (b*bb*seq, 2048, 1, 1) 
        x = x.view(b*seq, bb, -1)      # (b*seq, bb, 2048)
        
        # Pool each team Independently
        team1 = self.pool(x[:, :6, :]) # (b*seq, 1, 1024)
        team2 = self.pool(x[:, 6:, :]) # (b*seq, 1, 1024)
        
        x1 = torch.cat([team1, team2], dim=1) # (b*seq, 2, 1024)
        x1 = x1.view(b, seq, -1)   # (b, seq, 2048)
        x2, (h, c) = self.lstm(x1) # (b, seq, hidden)

        x1 = x1[:, -1, :] # (b, 2048)
        x2 = x2[:, -1, :] # (b, hidden)
        
        # Concat Special representation (from resnet50) and temporal representation 
        x = torch.cat([x1, x2], dim=1) # (b, hidden+2048)
        x = self.fc(x) # [b, num_classes] 
        return x 

def collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame and selecting the last frame label. 
    """
    clips, labels = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []

    for clip in clips:
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            clip = torch.cat((clip, clip_padding), dim=0)
    
        padded_clips.append(clip)
       
    padded_clips = torch.stack(padded_clips)
    labels = torch.stack(labels)
    
    labels = labels[:,-1, :] # utils the label of last frame
    
    return padded_clips, labels

def eval(root, config, checkpoint_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    person_classifer = PersonActivityClassifier(
        num_classes=config.model['num_classes']['person_activity']
    )

    model = GroupActivityClassifer(
        person_feature_extraction=person_classifer, 
        num_classes=config.model['num_classes']['group_activity'],
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    test_dataset = Group_Activity_DataSet(
        videos_path=f"{root}/{config.data['videos_path']}",
        annot_path=f"{root}/{config.data['annot_path']}",
        split=config.data['video_splits']['test'],
        labels=group_activity_labels,
        transform=test_transforms,
        seq=True,
        sort=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=14,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    prefix = "Group Activity B1-No-Relation-temporal eval on testset"
    path = str(Path(checkpoint_path).parent)

    metrics = model_eval(
        model=model, 
        data_loader=test_loader, 
        criterion=criterion, 
        device=device, 
        path=path, 
        prefix=prefix, 
        class_names=config.model["num_clases_label"]['group_activity']
    )

    return metrics

if __name__ == "__main__":
    ROOT = "/teamspace/studios/this_studio/Relational-Group-Activity-Recognition"
    CONFIG_PATH = f"{ROOT}/configs/temporal_models/B1_temporal.yml"
    MODEL_CHECKPOINT = f"{ROOT}/experiments/experiments/temporal_models/B1_no_relations_temporal_V1_2025_03_16_18_55/checkpoint_epoch_40.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ROOT", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH,
                        help="Path to the YAML configuration file")

    CONFIG = load_config(CONFIG_PATH)

    person_classifer = PersonActivityClassifier(9)
    group_classifer = GroupActivityClassifer(person_classifer, 8)
    
    summary(group_classifer)
    eval(ROOT, CONFIG, MODEL_CHECKPOINT)