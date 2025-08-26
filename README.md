<h1 align="center">Relational-Group-Activity-Recognition</h1>

<p align="center">
  This repository provides an implementation of the <strong>ECCV 2018 paper</strong>, 
  <a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Mostafa_Ibrahim_Hierarchical_Relational_Networks_ECCV_2018_paper.pdf">
    <em>Hierarchical Relational Networks for Group Activity Recognition</em>
  </a>.  
  Unlike traditional pooling methods (max, average, or attention pooling) that reduce dimensionality but discard important spatial and relational details, this paper introduces a <strong>relational layer</strong>. 
  The relational layer enhances a person’s representation by explicitly modeling interactions with its neighbors in a structured relationship graph, leading to richer scene-level understanding.
</p>

---

## Table of Contents
1. [Key Updates](#key-updates)
2. [Introduction](#introduction)
   - [How the Relational Layer Works](#how-the-relational-layer-works)
3. [Usage](#usage)  
   - [Clone the Repository](#1-clone-the-repository)  
   - [Install Dependencies](#2-install-the-required-dependencies)  
   - [Download Model Checkpoint](#3-download-the-model-checkpoint)  
     - [Option 1: Use Python Code](#option-1-use-python-code)  
     - [Option 2: Download Directly](#option-2-download-directly)  
4. [Dataset Overview](#dataset-overview)  
   - [Example Annotations](#example-annotations)  
   - [Train-Test Split](#train-test-split)  
   - [Dataset Statistics](#dataset-statistics)  
     - [Group Activity Labels](#group-activity-labels)  
     - [Player Action Labels](#player-action-labels)  
   - [Dataset Organization](#dataset-organization)  
   - [Dataset Download Instructions](#dataset-download-instructions)  
5. [Ablation Study](#ablation-study)  
   - [Baselines](#baselines)  
     - [Single Frame Models](#single-frame-models)  
       - [Performance Comparison (Original Paper)](#performance-comparison)  
       - [My Scores (Accuracy)](#my-scores-accuracy)  
     - [Temporal Models](#temporal-models)  
       - [Performance Comparison (Original Paper)](#performance-comparison-1)  
       - [My Scores (Accuracy)](#my-scores-accuracy-1)  
     - [Attention Models (New Baseline)](#attention-models-new-baseline)
       - [My Scores (Accuracy)](#my-scores-accuracy-2)    
       - [Confusion Matrix](#rcrg-2r-11c-conc-v1-attention-confusion-matrix)

## Key Updates

- **ResNet-50 Backbone**: Replaced VGG19 with ResNet-50 for stronger feature extraction.  
- **Ablation Studies**: Comprehensive experiments to evaluate the contribution of each model component.  
- **Test-Time Augmentation (TTA)**: Implemented to improve robustness and reliability during inference.
- **Graph Attention Operator**: Implementation for an attention-based relational layer.
- **Improved Performance**: Achieves consistently higher accuracy across all baselines compared to the original paper.  
- **Modern Implementation**: Fully implemented in **PyTorch** with support from **PyTorch Geometric**.  

---

# Introduction

Traditional pooling methods (max, average, or attention pooling) reduce dimensionality but often discard important **spatial** and **relational** details between people. The **Hierarchical Relational Network (HRN)** addresses this by introducing a **relational layer** that explicitly models interactions between individuals in a **structured relationship graph**.

<p align="center">
  <img width="512" height="512" src="https://github.com/user-attachments/assets/639cb140-a4df-4cd4-befc-2e965030723c" alt="Relational Layer Illustration"/>
</p>


## How the Relational Layer Works

1. **Graph Construction**  
   - Each person in a frame is represented as a node.  
   - People are ordered based on the top-left corner `(x, y)` of their bounding boxes (first by x, then by y if tied).  
   - Edges connect a person to their neighbors, forming **cliques** in the graph.  

2. **Initial Person Features**  
   Each person’s initial representation comes from a CNN backbone (e.g., ResNet50):  

   $$P_i^0 = \text{CNN}(I_i)$$

   where $I_i$ is the cropped image around person $i$.  

3. **Relational Update**  
  
  <p align="center">
  <img src="https://github.com/user-attachments/assets/d965ea0e-8599-4d0a-b20e-73c50fbfe6d0" alt="Graph Structure Illustration" width="750"/>
  </p>

   At relational layer $\ell$, person $i$’s updated representation is:  

   $$P_i^\ell = \sum_{j \in E_i^\ell} F^\ell(P_i^{\ell-1} \oplus P_j^{\ell-1}; \theta^\ell)$$

   - $E_i^\ell$: neighbors of person $i$ in graph $G^\ell$  
   - $\oplus$: concatenation operator  
   - $F^\ell$: shared MLP for layer $\ell$ (input size $2N_{\ell-1}$, output size $N_\ell$)  

   * This step computes pairwise relation vectors between $i$ and its neighbors, then aggregates them.  

3. **Hierarchical Stacking**  

   <p align="center">
   <img src="https://github.com/user-attachments/assets/8d8f4ea7-803c-486d-8fb7-00638445ddb7" alt="Graph Structure Illustration" width="750"/>
   </p>
  
   - Multiple relational layers are stacked, compressing person features while refining relational context.  
   - The architecture supports a variable number of people $K$ (robust to occlusions or false detections).  

4. **Scene Representation**  
   The final scene feature $S$ is obtained by pooling person features from the last relational layer:  

   $$S = P_1^L \ ▽ \ P_2^L \ ▽ \dots \ ▽ \ P_K^L$$

   where $▽$ is a pooling operator (e.g., concatenation or element-wise max pooling).  

-----
## Usage

---

### 1. Clone the Repository
```bash
git clone https://github.com/Sh-31/Relational-Group-Activity-Recognition.git
```

### 2. Install the Required Dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Download the Model Checkpoint
This is a manual step that involves downloading the model checkpoint files.

#### Option 1: Use Python Code
Replace the `modeling` folder with the downloaded folder:
```python
import kagglehub

# Download the latest version
path = kagglehub.model_download("sherif31/group-activity-recognition/pyTorch/v1")

print("Path to model files:", path)
```

#### Option 2: Download Directly
Browse and download the specific checkpoint from Kaggle:  
[Relational-Group-Activity-Recognition - PyTorch Checkpoint](https://www.kaggle.com/models/sherif31/relational-group-activity-recognition/)

-----
## Dataset Overview

The dataset was created using publicly available YouTube volleyball videos. The authors annotated 4,830 frames from 55 videos, categorizing player actions into 9 labels and team activities into 8 labels. 

### Example Annotations

![image](https://github.com/user-attachments/assets/50f906ad-c68c-4882-b9cf-9200f5a380c7)

- **Figure**: A frame labeled as "Left Spike," with bounding boxes around each player, demonstrating team activity annotations.

![image](https://github.com/user-attachments/assets/cca9447a-8b40-4330-a11d-dbc0feb230ff)

### Train-Test Split

- **Training Set**: 3,493 frames
- **Testing Set**: 1,337 frames

### Dataset Statistics

#### Group Activity Labels
| Group Activity Class | Instances |
|-----------------------|-----------|
| Right set            | 644       |
| Right spike          | 623       |
| Right pass           | 801       |
| Right winpoint       | 295       |
| Left winpoint        | 367       |
| Left pass            | 826       |
| Left spike           | 642       |
| Left set             | 633       |

#### Player Action Labels
| Action Class | Instances |
|--------------|-----------|
| Waiting      | 3,601     |
| Setting      | 1,332     |
| Digging      | 2,333     |
| Falling      | 1,241     |
| Spiking      | 1,216     |
| Blocking     | 2,458     |
| Jumping      | 341       |
| Moving       | 5,121     |
| Standing     | 38,696    |

### Dataset Organization

- **Videos**: 55, each assigned a unique ID (0–54).
- **Train Videos**: 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54.
- **Validation Videos**: 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51.
- **Test Videos**: 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47.

### Dataset Download Instructions

1. Enable Kaggle's public API. Follow the guide here: [Kaggle API Documentation](https://www.kaggle.com/docs/api).  
2. Use the provided shell script:
```bash
  chmod 600 .kaggle/kaggle.json 
  chmod +x script/script_download_volleball_dataset.sh
  .script/script_download_volleball_dataset.sh
```
For further information about dataset, you can check out the paper author's repository:  
[link](https://github.com/mostafa-saad/deep-activity-rec)

---

## [Ablation Study](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence)#:~:text=In%20artificial%20intelligence%20(AI)%2C,resultant%20performance%20of%20the%20system)

### Baselines

#### Single Frame Models:

- **B1-NoRelations:** In the first stage, Resnet50 is fine-tuned and a person is represented with 2048-d features. In the second stage, each person is connected to a shared dense layer of 128 features, then the person representations (each of length 128 features) are pooled, and then fed to a softmax layer for group activity classification.

- **RCRG-1R-1C:** Pretrained Resnet50 network is fine-tuned and a person is represented with 4096-d features, then a single relational layer (1R), all people in 1 clique (1C), so all-pairs relationships are learned.

- **RCRG-1R-1C-!tuned:** Same as previous variant, but Pretrained Resnet50 network without fine-tuning.

- **RCRG-2R-11C:** Close to the RCRG-1R-1C variant, but uses 2 relational layers (2R) of sizes 256 and 128. The graphs of these 2 layers are 1 clique (11C) of all people. This variant and the next ones explore stacking layers with different graph structures.

- **RCRG-2R-21C:** Same as the previous model, but the first layer has 2 cliques, one per team. The second layer is all-pairs relations (1C).

- **RCRG-3R-421C:** There relational layers (of sizes 512, 256, and 128) with clique sizes of the layers set to (4, 2, 1). The first layer has 4 cliques, with each team divided into 2 cliques.

##### Performance comparison

###### Original Paper Baselines Score

<img width="615" height="542" alt="{98F2C621-4E89-47FD-A112-A25946D611F3}" src="https://github.com/user-attachments/assets/4f7cc2f7-2b6d-472d-9555-d4d9b2de65cc" />

###### My Scores (Accuracy)

| Model | Test Acc | Test Acc TTA (4) | Paper Test ACC |
| :---- | :---: | :---: | :---: |
| B1-no-relations | 89.07% | 89.06% | 85.1% |
| RCRG-1R-1C | 89.42% | \- | 86.5% |
| RCRG-1R-1C-untuned | 80.86% | \- | 75.4% |
| RCRG-2R-11C | 89.15% | \- | 86.1% |
| RCRG-2R-21C | 89.49% | \- | 87.2% |
| RCRG-3R-421C | 88.97% | \- | 86.4% |
| **RCRG-2R-11C-conc** | **89.60%** | **89.71%** | 88.3% |
| **RCRG-2R-21C-conc** | **89.60%** | 89.60% | 86.7% |
| RCRG-3R-421C-conc | 89.23% | \- | 87.3% |

Notes:
- `-conc` postfix is used to indicate concatenation pooling instead of max-pooling.
- Used 4 transform augmentation at TTA.
  
#### Temporal Models:

- **RCRG-2R-11C-conc-temporal:** Uses 2 relational layers (2R) of sizes 256 and 128. The graphs of these 2 layers are 1 clique (11C) of all people. 

- **RCRG-2R-21C:** The first layer has 2 cliques, one per team. The second layer is all-pairs relations (1C).

Notes:
- `Temporal`: postfix is used to indicate model work with a sequence of frames, not a frame.
- `-conc` postfix is used to indicate concatenation pooling instead of max-pooling.
- The original paper did not clearly specify where the LSTM unit should be integrated into the model.  
  To explore this, I implemented three possible variants:  
  - `V1`: LSTM **before** the relational layer → allows the relational layer to learn richer spatio-temporal features.  
  - `V2`: LSTM **after** the relational layer → enhances the relational features with temporal modeling.  
  - `V3`: LSTMs **both before and after** the relational layer → combines the strengths of V1 and V2.  

##### Performance comparison

###### Original Paper Baselines Score

<img width="523" height="323" alt="{848262DC-9865-4F49-A7CA-60B08675A6B8}" src="https://github.com/user-attachments/assets/de4eb7cb-3f3c-4320-baef-c7679055a6dd" />

###### My Scores (Accuracy)

| Model | Test Acc | Test Acc TTA (3) | Paper Test ACC |
| :---- | :---: | :---: | :---: |
| B1-no-relations-temporal | 88.93% | 89.60% | \- |
| RCRG-2R-11C-conc-V1 | 90.50% | 90.73% | 89.5% |
| RCRG-2R-11C-conc-V2 | **91.55%** | 91.62% | 89.5% |
| RCRG-2R-11C-conc-V3 | 91.40% | **91.77%** | 89.5% |
| RCRG-2R-21C | \- | \- | 89.4% |

Note:
- I decide to train `RCRG-2R-11C-conc` only for now because it's best on my and paper Implementation.


#### Attention Models (new baseline):

- Uses 2 relational layers (2R). The graphs of these 2 layers are 1 clique (11C) of all players, but this time using a graph attentional operator instead of an MLP for relational layers.

###### My Scores (Accuracy)

| Model | Test Acc | Test Acc TTA (3) | Paper Test ACC |
| :---- | :---: | :---: | :---: |
| RCRG-2R-11C-conc-V1 | 91.77% | **92.00%** | \- |

###### RCRG-2R-11C-conc-V1-Attention Confusion Matrix

<img src="experiments/attention_models/RCRG-R2-C11-conc-temporal_V1_2025_06_28_18_37/Group_Activity_RCRG-R2-C11-Conc-Temporal-Attention-TTA_Eval_On_Testset_TTA_confusion_matrix.png" alt="RCRG-2R-11C-conc-V1" >

-----
