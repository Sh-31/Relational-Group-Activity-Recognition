import os
import torch

def get_sampler_weights(dataset):
    if os.path.exists('sampler_weights.pth'):
        sampler_weight = torch.load('sampler_weights.pth') 
        return sampler_weight['samples_weights'], sampler_weight['class_weights']

    labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        labels.append(label.argmax().item()) 
    
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = class_counts.sum() / (len(class_counts) * class_counts.float())
    class_weights = class_weights / class_weights.sum()
    samples_weights = torch.tensor([class_weights[label] for label in labels], dtype=torch.float)
    
    torch.save({
        'samples_weights': samples_weights, 
        'class_weights': class_weights
    }, 'sampler_weights.pth')

    return samples_weights, class_weights