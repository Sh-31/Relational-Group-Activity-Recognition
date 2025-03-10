"""
RCRG-R3-C421-conc (RCRG-3R-421C-conc) Description:
--------------------------------
There relational layers (of sizes 512, 256 and 128) 
with clique sizes of the layers set to (4, 2, 1).
The first layer has 4 cliques, with each team divided into 2 cliques.
-conc postfix is used to indicate concatenation pooling is used instead of max pooling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import argparse
import itertools
import torch.nn as nn
import albumentations as A
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchinfo import summary
from .relational_unit import RelationalUnit
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
    def __init__(self, person_feature_extraction, num_classes, device):
        super(GroupActivityClassifer, self).__init__()

        self.device = device
        self.resnet50 = person_feature_extraction.resnet50
        
        for module in [self.resnet50]:
            for param in module.parameters():
                param.requires_grad = False

        self.r1 = RelationalUnit( # relational layer one
            in_channels=2048, 
            out_channels=128, 
            hidden_dim=256
        ) 

        self.r2 = RelationalUnit( # relational layer two
            in_channels=2048, 
            out_channels=256, 
            hidden_dim=512
        ) 

        self.r3 = RelationalUnit( # relational layer there
            in_channels=2048, 
            out_channels=512, 
            hidden_dim=1024
        ) 
        
        self.fc = nn.Sequential( 
            nn.Linear(896*12, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes), 
        )
    
    def forward(self, x):
        b, bb, c, h, w = x.shape  # batch, bbox, channals, hight, width
        x = x.view(b*bb, c, h, w) # (b*bb, c, h, w)
        x = self.resnet50(x)      # (b*bb, 2048, 1, 1) 
        x = x.view(b, bb, -1)     # (b, bb, 2048)
        
        # The first relational layer has 4 cliques
        num_nodes = (x.shape[1] + 1) // 4 # num_nodes = 3      
        edge_index = torch.tensor([(i, j) for i, j in itertools.permutations(range(0, num_nodes), 2)]).t().to(self.device) # Generate all (i, j) pairs where i ≠ j
        x1 = self.r1(x[:,  :3, :], edge_index)       # (b, bb/4, 128) {0, 1, 2}                                     
        x2 = self.r1(x[:, 3:6, :], edge_index)       # (b, bb/4, 128) {3, 4, 5} 
        x3 = self.r1(x[:, 6:9, :], edge_index)       # (b, bb/4, 128) {6, 7, 8}
        x4 = self.r1(x[:, 9: , :], edge_index)       # (b, bb/4, 128) {9, 10, 11}                                                     
        x_r1 = torch.concat([x1, x2, x3, x4], dim=1) # (b, bb,   128)     

        # The second relational layer has 2 cliques
        num_nodes = (x.shape[1] + 1) // 2 # num_nodes = 6
        edge_index = torch.tensor([(i, j) for i, j in itertools.permutations(range(num_nodes), 2)]).t().to(self.device) # Generate all (i, j) pairs where i ≠ j
        x1 = self.r2(x[:, :6, :], edge_index) # (b, bb/2, 256) {0, 1, 2, 3, 4, 5}                                  
        x2 = self.r2(x[:, 6:, :], edge_index) # (b, bb/2, 256) {6, 7, 8, 9, 10, 11}                                                    
        x_r2 = torch.concat([x1, x2], dim=1)  # (b, bb,   256)     
        
        # The thrid relational layer has 1 cliques
        num_nodes = x.shape[1]  # all 12 player at one graph 
        edge_index = torch.tensor([(i, j) for i, j in itertools.permutations(range(num_nodes), 2)]).t().to(self.device) # Generate all (i, j) pairs where i ≠ j
        x_r3 = self.r3(x, edge_index) # (b, bb, 512)

        x = torch.concat([x_r1, x_r2, x_r3], dim=2) # (b, bb, 896)
        x = x.view(b, -1)                           # (b, bb*896) 
        x = self.fc(x)                              # (b, num_classes)

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
    prefix = "Group Activity RCRG-R3-C421-conc eval on testset"
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
    CONFIG_PATH = f"{ROOT}/configs/RCRG_R3_C421_conc.yml"
    MODEL_CHECKPOINT = f"{ROOT}/experiments/RCRG_R3_C421_conc_V1_2025_03_08_23_39/checkpoint_epoch_34.pkl"


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