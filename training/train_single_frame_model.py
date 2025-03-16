import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from trainer import ddp_train
from utils.helper_utils import load_config
from utils.data_utils import Group_Activity_DataSet, group_activity_labels

from models.single_frame_models.RCRG_R3_C421 import PersonActivityClassifier, GroupActivityClassifer, collate_fn

import albumentations as A
from albumentations.pytorch import ToTensorV2


def train(
    ROOT,
    config_path,
    person_cls_checkpoint_path,
    output_dir,
 ):
   
    config = load_config(config_path)

    train_transforms = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise(),
            A.MotionBlur(blur_limit=5), 
            A.MedianBlur(blur_limit=5)  
        ], p=0.95),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90()
        ], p=0.01),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = Group_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['train'],
        labels=group_activity_labels,
        transform=train_transforms,
        seq=False, 
        sort=True
    )
    
    val_dataset = Group_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['validation'],
        labels=group_activity_labels,
        transform=val_transforms,
        seq=False,
        sort=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    person_cls_checkpoint = torch.load(person_cls_checkpoint_path)

    person_classifer = PersonActivityClassifier(
        num_classes=config.model['num_classes']['person_activity']
    )

    person_classifer.load_state_dict(person_cls_checkpoint['model_state_dict'])

    model = GroupActivityClassifer(
        person_feature_extraction=person_classifer, 
        num_classes=config.model['num_classes']['group_activity'],
        device=device
    )

    ddp_train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        config=config, 
        output_dir=output_dir,
        logger_step=50,
        task='group_activity',
        class_names=config.model['num_clases_label']['group_activity']
    )

if __name__ == "__main__":
   
    ROOT = "/kaggle"
    config_path = "/kaggle/working/Relational-Group-Activity-Recognition/configs/..."
    person_cls_checkpoint_path = "/kaggle/input/person-activity-classifier/Relational-Group-Activity-Recognition/experiments/person_activity_classifier_V1_2025_03_03_17_08/checkpoint_epoch_16.pkl" # Personn Activity Classifier checkpoint
    output_dir = "/kaggle/working/Relational-Group-Activity-Recognition/experiments/single_frame_models"
    
    train(
        ROOT=ROOT,
        config_path=config_path,
        person_cls_checkpoint_path=person_cls_checkpoint_path,
        output_dir=output_dir
    )