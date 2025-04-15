"""
RCRG-R2-C11-conc-temporal (RCRG-2R-11C-conc) Description:
--------------------------------
Uses 2 relational layers (2R) of sizes 256 and 128. 
The graphs of these 2 layers are 1 clique (11C) of all people. 
but this time using graph attentional operator instead of MLP for relational layers.

- Conc: postfix is used to indicate concatenation pooling instead of max-pooling.
- Temporal: postfix is used to indicate model work with seqance of frames not a frame.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import argparse
import itertools
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchinfo import summary
from .relational_attention import RelationalUnit
from utils import load_config, Group_Activity_DataSet_END2END, activities_labels, model_eval, model_eval_TTA

class PersonActivityClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PersonActivityClassifier, self).__init__()
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.resnet50.parameters():
            param.requires_grad = False
                
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

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
    def __init__(self, person_feature_extraction, num_classes, device):
        super(GroupActivityClassifer, self).__init__()

        self.device = device
        self.resnet50 = person_feature_extraction.resnet50
        self.fc_1 = person_feature_extraction.fc

        self.r1 = RelationalUnit( # relational layer one
            in_channels=2048, 
            out_channels=128, 
        ) 

        self.r2 = RelationalUnit( # relational layer two
            in_channels=2048, 
            out_channels=256, 
        ) 

        self.layer_norm = nn.LayerNorm(384) 
        self.lstm = nn.LSTM(
            input_size=384,
            hidden_size=384,
            batch_first=True
        ) 

        self.fc_2 = nn.Sequential(
            nn.Linear(12*384, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes), 
        )
    
    def forward(self, x):
        b, bb, seq, c, h, w = x.shape   # batch, bbox, frames, channals, hight, width
        x = x.view(b*bb*seq, c, h, w)   # (b*bb*seq, c, h, w)
        x = self.resnet50(x)            # (b*bb*seq, 2048, 1, 1) 
        x_p = x.view(b*bb, seq, -1)     # (b*bb, seq, 2048) 
        y1 = self.fc_1(x_p[:, -1, :])   # (b*bb, person_num_classes)
        
        # The frist and second relation layer has 1 cliques
        x = x.view(b*seq, bb, -1)         # (b*seq, bb, 2048) 
        num_nodes = x.shape[1]            # all 12 player at one graph 
        edge_index = torch.tensor([(i, j) for i, j in itertools.permutations(range(num_nodes), 2)]).t().to(self.device) # Generate all (i, j) pairs where i ≠ j

        x1 = self.r1(x, edge_index)       # (b*seq, bb, 128)
        x2 = self.r2(x, edge_index)       # (b*seq, bb, 256)    
        x = torch.concat([x1, x2], dim=2) # (b*seq, bb, 384) 
        
        x = x.view(b*bb, seq, -1)         # (b*bb, seq, 384)
        x = self.layer_norm(x)            # (b*bb, seq, 384)
        x, (h, c) = self.lstm(x)          # (b*bb, seq, 384)
        x = x[:, -1, :]                   # (b*bb, 384)
       
        x = x.contiguous().view(b, -1)    # (b, bb*384) 
        y2 = self.fc_2(x)                 # (b, num_classes)

        return {'person_output': y1, 'group_output': y2}

def collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame and selecting the last frame label. 
    """
    clips, person_labels, group_labels  = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []
    padded_person_labels = []

    for clip, label in zip(clips, person_labels):
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            label_padding = torch.zeros((max_bboxes - num_bboxes, label.size(1), label.size(2)))
            
            clip = torch.cat((clip, clip_padding), dim=0)
            label = torch.cat((label, label_padding), dim=0)
            
        padded_clips.append(clip)
        padded_person_labels.append(label)
    
    padded_clips = torch.stack(padded_clips)
    padded_person_labels = torch.stack(padded_person_labels)
    group_labels = torch.stack(group_labels)
    
    group_labels = group_labels[:,-1, :] # # utils the label of last frame
    padded_person_labels = padded_person_labels[:, :, -1, :]  # utils the label of last frame for each player
    b, bb, num_class = padded_person_labels.shape # batch, bbox, num_clases
    padded_person_labels = padded_person_labels.view(b*bb, num_class)

    return padded_clips, padded_person_labels, group_labels

def eval(root, config, checkpoint_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    person_classifer = PersonActivityClassifier(
        num_classes=config.model['num_classes']['person_activity']
    )

    model = GroupActivityClassifer(
        person_feature_extraction=person_classifer, 
        num_classes=config.model['num_classes']['group_activity'],
        device=device
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
    
    test_dataset = Group_Activity_DataSet_END2END(
        videos_path=f"{root}/{config.data['videos_path']}",
        annot_path=f"{root}/{config.data['annot_path']}",
        split=config.data['video_splits']['test'],
        labels=activities_labels,
        transform=test_transforms,
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
    prefix = "Group Activity RCRG-R2-C11-conc-temporal-attention eval on testset"
    path = str(Path(checkpoint_path).parent)

    metrics = model_eval(
        model=model, 
        data_loader=test_loader, 
        criterion=criterion, 
        device=device, 
        path=path, 
        prefix=prefix, 
        END2END=True,
        class_names=config.model["num_clases_label"]['group_activity']
    )

    return metrics

def eval_with_TTA(root, config, checkpoint_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    person_classifer = PersonActivityClassifier(
        num_classes=config.model['num_classes']['person_activity']
    )

    model = GroupActivityClassifer(
        person_feature_extraction=person_classifer, 
        num_classes=config.model['num_classes']['group_activity'],
        device=device
    )

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    dataset_params = {
    'videos_path': f"{root}/{config.data['videos_path']}",
    'annot_path': f"{root}/{config.data['annot_path']}",
    'split': config.data['video_splits']['test'],
    'labels': activities_labels,
    'batch_size': 14,
    'num_workers': 4,
    'collate_fn': collate_fn,
    'pin_memory': True
    }

    tta_transforms = [
        A.Compose([ #  transform (base)
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ColorJitter(brightness=0.2),
                A.RandomBrightnessContrast(),
                A.GaussNoise(),
                A.MotionBlur(blur_limit=5), 
                A.MedianBlur(blur_limit=5)  
            ], p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),

        A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.ColorJitter(brightness=0.2),
                A.RandomBrightnessContrast(),
                A.GaussNoise(),
                A.MotionBlur(blur_limit=5), 
                A.MedianBlur(blur_limit=5)  
            ], p=1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    ]

    criterion = nn.CrossEntropyLoss()
    prefix = "Group Activity RCRG-R2-C11-conc-TTA eval on testset"
    path = str(Path(checkpoint_path).parent)

    metrics = model_eval_TTA(
        model=model,
        dataset=Group_Activity_DataSet_END2END,
        dataset_params=dataset_params,
        tta_transforms=tta_transforms,
        criterion=criterion,
        path=path,
        device=device,
        prefix=prefix,
        END2END=True,
        class_names=config.model["num_clases_label"]['group_activity']
    )

    return metrics

if __name__ == "__main__":
    ROOT = "/teamspace/studios/this_studio/Relational-Group-Activity-Recognition"
    CONFIG_PATH = f"{ROOT}/configs/attention_models/RCRG_R2_C11_conc_temporal.yml"
    MODEL_CHECKPOINT = f"{ROOT}/experiments/attention_models/RCRG_R2_C11_conc_attention_V1_2025_04_13_03_09/Best_Model.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ROOT", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH,
                        help="Path to the YAML configuration file")

    CONFIG = load_config(CONFIG_PATH)

    person_classifer = PersonActivityClassifier(9)
    group_classifer = GroupActivityClassifer(person_classifer, 8, 'cpu')
    
    # summary(group_classifer, input_size=(2, 12, 9, 3, 224, 224))
    # eval(ROOT, CONFIG, MODEL_CHECKPOINT)
    # eval_with_TTA(ROOT, CONFIG, MODEL_CHECKPOINT)
