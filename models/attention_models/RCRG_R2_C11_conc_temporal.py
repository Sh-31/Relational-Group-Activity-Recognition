"""
RCRG-R2-C11-conc-temporal (RCRG-2R-11C-conc) Description:
--------------------------------
Uses 2 relational layers (2R) The graphs of these 2 layers are 1 clique (11C) of all player. 
but this time using graph attentional operator instead of MLP for relational layers.

- Conc: postfix is used to indicate concatenation pooling instead of max-pooling.
- Temporal: postfix is used to indicate model work with seqance of frames not a one frame.
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
from utils import load_config, Group_Activity_DataSet, group_activity_labels, model_eval, model_eval_TTA

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
    def __init__(self, person_feature_extraction, num_classes, device):
        super(GroupActivityClassifer, self).__init__()

        self.device = device
        self.resnet50 = person_feature_extraction.resnet50
        
        for module in [self.resnet50]:
            for param in module.parameters():
                param.requires_grad = False

        self.r1 = RelationalUnit( # relational layer one
            in_channels=2048, 
            out_channels=2048,
            num_heads=4,
            dropout_rate=0.5
        ) 
        
        self.r2 = RelationalUnit( # relational layer two
            in_channels=2048, 
            out_channels=2048, 
            num_heads=4,
            dropout_rate=0.5
        ) 

        self.proj = nn.Linear(2048, 512)
        self.ln3 = nn.LayerNorm(512) 
        self.ln4 = nn.LayerNorm(512) 
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            batch_first=True
        ) 

        self.fc = nn.Sequential(
            nn.Linear(12*512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes), 
        )
    
    def forward(self, x):
        b, bb, seq, c, h, w = x.shape       # batch, bbox, frames, channals, hight, width
        x = x.view(b*bb*seq, c, h, w)       # (b*bb*seq, c, h, w)
        x = self.resnet50(x)                # (b*bb*seq, 2048, 1, 1) 
        x = x.view(b*seq, bb, -1)           # (b*seq, bb, 2048) 
        # The frist and second relation layer has 1 cliques
        num_nodes = x.shape[1]              # all 12 player at one graph 
        edge_index = torch.tensor([(i, j) for i, j in itertools.permutations(range(num_nodes), 2)]).t().to(self.device) # Generate all (i, j) pairs where i ≠ j
   
        x_r1 = self.r1(x, edge_index)       # (b*seq, bb, 2048)
        x_r2 = self.r2(x_r1, edge_index)    # (b*seq, bb, 2048)

        x_r2 = x_r2.view(b*bb, seq, -1)     # (b*bb, seq, 2048)
        x_t, (h, c) = self.lstm(x_r2)       # (b*bb, seq, 512)
        x_t = self.ln3(x_t)                 # (b*bb, seq, 512)
        
        x_r2 = x_r2[:, -1, :]               # (b*bb, 2048)
        x_t = x_t[:, -1, :]                 # (b*bb, 512)

        x = self.ln4(self.proj(x_r2) + x_t) # (b*bb, 512)
        x = x.contiguous().view(b, -1)      # (b, bb*512) 
        x = self.fc(x)                      # (b, num_classes)

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
    prefix = "Group Activity RCRG-R2-C11-Conc-Temporal-Attention Eval On Testset"
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
    'labels': group_activity_labels,
    'seq': True,
    'sort': True,
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
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90()
            ], p=0.01),
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
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90()
            ], p=0.01),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]

    criterion = nn.CrossEntropyLoss()
    prefix = "Group Activity RCRG-R2-C11-Conc-Temporal-Attention-TTA Eval On Testset"
    path = str(Path(checkpoint_path).parent)

    metrics = model_eval_TTA(
        model=model,
        dataset=Group_Activity_DataSet,
        dataset_params=dataset_params,
        tta_transforms=tta_transforms,
        criterion=criterion,
        path=path,
        device=device,
        prefix=prefix,
        class_names=config.model["num_clases_label"]['group_activity']
    )

    return metrics

if __name__ == "__main__":
    ROOT = "/teamspace/studios/this_studio/Relational-Group-Activity-Recognition"
    CONFIG_PATH = f"{ROOT}/configs/attention_models/RCRG_R2_C11_conc.yml"
    MODEL_CHECKPOINT = f"{ROOT}/experiments/attention_models/RCRG-R2-C11-conc-temporal_V1_2025_06_28_18_37/checkpoint_epoch_42.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ROOT", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH,
                        help="Path to the YAML configuration file")

    CONFIG = load_config(CONFIG_PATH)

    person_classifer = PersonActivityClassifier(9)
    group_classifer = GroupActivityClassifer(person_classifer, 8, 'cpu')
    
    summary(group_classifer)
    eval(ROOT, CONFIG, MODEL_CHECKPOINT)
    eval_with_TTA(ROOT, CONFIG, MODEL_CHECKPOINT)