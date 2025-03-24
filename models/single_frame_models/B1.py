"""
B1-NoRelations Description:

In the first stage, Resnet50 is fined tuned and a person is represented with 
2048-d features. In the second stage, each person is connected to a shared dense layer of 128 features, 
then the person representations (each of length 128 features) are pooled, 
then fed to a softmax layer for group activity classification. 
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

        self.dense_layer = nn.Linear(2048, 128)
        self.pool = nn.AdaptiveMaxPool2d((1, 128))  
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes), 
        )
    
    def forward(self, x):
        b, bb, c, h, w = x.shape  # batch, bbox, channals, hight, width
        x = x.view(b*bb, c, h, w) # (b*bb, c, h, w)
        x = self.resnet50(x)      # (b*bb, 2048, 1, 1) 

        x = x.view(b, bb, -1)     # (b, bb, 2048)
        x = self.dense_layer(x)   # (b, bb, 128)

        team_1 = self.pool(x[:, :6, :]) # (b, 1, 128) 
        team_2 = self.pool(x[:, 6:, :]) # (b, 1, 128) 
        
        x = torch.concat([team_1, team_2], dim=1) # (b, 2, 128) 
        x = x.view(b, -1)                         # (b, 256) 
        x = self.fc(x)                            # (b, num_classes)

        return x 

def collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame.
    """
    clips, labels = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []
    padded_labels = []

    for clip, label in zip(clips, labels) :
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3)))
            clip = torch.cat((clip, clip_padding), dim=0)
            
        padded_clips.append(clip)
        padded_labels.append(label)
    
    padded_clips = torch.stack(padded_clips)
    padded_labels = torch.stack(padded_labels)
    
    return padded_clips, padded_labels

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
        seq=False,
        sort=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    prefix = "Group Activity B1-No-Relation eval on testset"
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
        'seq': False,
        'sort': True,
        'batch_size': 128,
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
            ], p=0.55),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),

        A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.GaussNoise(),
            ], p=0.55),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]), 

        A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.MotionBlur(blur_limit=5), 
                A.MedianBlur(blur_limit=5)      
            ], p=0.55),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]

    criterion = nn.CrossEntropyLoss()
    prefix = "Group Activity B1-No-Relation-TTA eval on testset"
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
    CONFIG_PATH = f"{ROOT}/configs/single_frame_models/B1.yml"
    MODEL_CHECKPOINT = f"{ROOT}/experiments/single_frame_models/B1_no_relations_V1_2025_03_08_01_07/checkpoint_epoch_19.pkl"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ROOT", type=str, default=ROOT,
                        help="Path to the root directory of the project")
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH,
                        help="Path to the YAML configuration file")

    CONFIG = load_config(CONFIG_PATH)

    person_classifer = PersonActivityClassifier(9)
    group_classifer = GroupActivityClassifer(person_classifer, 8, 'cpu')
    
    summary(group_classifer)
    # eval(ROOT, CONFIG, MODEL_CHECKPOINT)
    eval_with_TTA(ROOT, CONFIG, MODEL_CHECKPOINT)