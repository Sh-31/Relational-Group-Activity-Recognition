import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import ddp_train
from utils.helper_utils import load_config
from utils.data_utils import Person_Activity_DataSet, person_activity_labels
from models.PersonActivityClassifier import PersonActivityClassifier

import albumentations as A
from albumentations.pytorch import ToTensorV2


def Train_Person_Activity_Classifier(
    ROOT,
    config_path,
    experiment_name,
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
        ], p=0.70),
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

    train_dataset = Person_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['train'],
        labels=person_activity_labels,
        transform=train_transforms,
    )
    
    val_dataset = Person_Activity_DataSet(
        videos_path=f"{ROOT}/{config.data['videos_path']}",
        annot_path=f"{ROOT}/{config.data['annot_path']}",
        split=config.data['video_splits']['validation'],
        labels=person_activity_labels,
        transform=val_transforms,
    )

    model = PersonActivityClassifier(num_classes=config.model['num_classes']['person_activity'])

    ddp_train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_names=config.model['num_clases_label'],
        config=config, 
        experiment_name=experiment_name, 
        output_dir=output_dir,
        logger_step=100,
        task='person_activity'
    )




if __name__ == "__main__":
    ROOT = "/kaggle"
    config_path = "/kaggle/working/Relational-Group-Activity-Recognition/configs/person_activity_classifier.yml"
    output_dir = "/kaggle/working/Relational-Group-Activity-Recognition/experiments"
    
    Train_Person_Activity_Classifier(
        ROOT=ROOT,
        config_path=config_path,
        experiment_name="person_activity_classifier",
        output_dir=output_dir
    )