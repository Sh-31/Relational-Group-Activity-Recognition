import logging
import torch
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def get_f1_score(y_true, y_pred, average='weighted', report=False):
    if report:
        print("Classification Report:\n")
        print(classification_report(y_true, y_pred, zero_division=1))
    else:
        f1 = f1_score(y_true, y_pred, average=average)
        print(f"F1 Score: {f1:.4f}")
        return f1


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100  
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_title("Confusion Matrix (Counts)")
    
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].set_title("Confusion Matrix (Percentage)")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrices saved to {save_path}")
    
    plt.close(fig)
    
    return fig


def model_eval(model, data_loader, criterion=None, path="", device=None, prefix="Group Activity Test Set Classification Report", class_names=None, log_path="evaluation.log", END2END=False):
    
    logging.basicConfig(
        filename=f"{path}/{log_path}", 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s', 
        filemode='a'
    )
    
    model.eval()  
    y_true = []
    y_pred = []
    total_loss = 0.0
    
    if END2END:
        with torch.no_grad():
            for inputs, person_labels, group_labels in data_loader:
                inputs = inputs.to(device)
                person_labels = person_labels.to(device)
                group_labels = group_labels.to(device)
                
                outputs = model(inputs)
                loss_1 = criterion(outputs['person_output'], person_labels)
                loss_2 = criterion(outputs['group_output'], group_labels)
                
                total_loss += (loss_2 + (0.60 * loss_1)).item()
                
                _, predicted = outputs['group_output'].max(1)
                _, target_class = group_labels.max(1)
    
                y_true.extend(target_class.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy()) 
    else:
        with torch.no_grad(): 
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                
                if criterion:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                _, target_class = targets.max(1)
                
                y_true.extend(target_class.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    accuracy = report_dict.get("accuracy", 0) * 100
 
    avg_loss = total_loss / len(data_loader) if criterion else None
    f1 = f1_score(y_true, y_pred, average='weighted')

    log_message = f"\n{'=' * 50}\n{prefix}\n{'=' * 50}\n" \
                  f"Accuracy : {accuracy:.2f}%\n"
    if criterion:
        log_message += f"Average Loss: {avg_loss:.4f}\n"
    log_message += f"F1 Score (Weighted): {f1:.4f}\n\nClassification Report:\n"
    log_message += classification_report(y_true, y_pred, target_names=class_names)
    
    print(log_message)
    logging.info(log_message)

    if class_names:
        save_path = f"{path}/{prefix.replace(' ', '_')}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path)
    
    metrics = {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "f1_score": f1,
        "classification_report": report_dict,
    }
    return metrics

def model_eval_TTA(
    model, 
    dataset, 
    dataset_params, 
    tta_transforms, 
    criterion=None, 
    path="", 
    device=None, 
    prefix="Group Activity Test Set Classification Report", 
    class_names=None, 
    END2END=False,
    log_path="TTA-evaluation.log"
    ):

    logging.basicConfig(
        filename=f"{path}/{log_path}", 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s', 
        filemode='a'
    )

    model.eval()
    all_predictions = []
    all_targets = []

    total_loss = 0.0

    for transform_idx, transform in enumerate(tta_transforms):
        print(f"Processing TTA transform {transform_idx+1}/{len(tta_transforms)}")
        
        params = dataset_params.copy()
        params['transform'] = transform
        
        if END2END:
            test_dataset = dataset(
                videos_path=params['videos_path'],
                annot_path=params['annot_path'],
                split=params['split'],
                labels=params['labels'],
                transform=params['transform'],
            )
        else:    
            test_dataset = dataset(
                videos_path=params['videos_path'],
                annot_path=params['annot_path'],
                split=params['split'],
                labels=params['labels'],
                transform=params['transform'],
                seq=params.get('seq', True),
                sort=params.get('sort', True),
                only_tar=params.get('only_tar', False)
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=params.get('batch_size', 12),
            shuffle=False,  # to keep order consistent
            num_workers=params.get('num_workers', 1),
            collate_fn=params.get('collate_fn', None),
            pin_memory=params.get('pin_memory', True)
        )
        
        transform_predictions = []
        transform_targets = []
    
        if END2END:
            with torch.no_grad():
                for inputs, person_labels, group_labels in test_loader:
                    inputs = inputs.to(device)
                    person_labels = person_labels.to(device)
                    group_labels = group_labels.to(device)
                    
                    outputs = model(inputs)
                    
                    transform_predictions.extend(outputs['group_output'].cpu().tolist())
                    transform_targets.extend(group_labels.cpu().tolist())
        
            all_predictions.append(torch.tensor(transform_predictions))
            all_targets.append(torch.tensor(transform_targets))
        else:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)  
                    
                    outputs = model(inputs)
                    
                    transform_predictions.extend(outputs.cpu().tolist())
                    transform_targets.extend(targets.cpu().tolist())

            all_predictions.append(torch.tensor(transform_predictions))
            all_targets.append(torch.tensor(transform_targets))

    # Convert predictions and targets to tensors
    avg_predictions = torch.mean(torch.stack(all_predictions), dim=0) # (len(test_dataset), 8)
    all_targets = torch.tensor(all_targets[0].clone().detach()) # all_targets are same in dataloader (len_testset, 8)

    if criterion:
        avg_predictions_device = avg_predictions.to(device)
        all_targets_device = all_targets.to(device)
        loss = criterion(avg_predictions_device, all_targets_device)
        total_loss = loss.item() * all_targets.size(0)

    _, all_targets = all_targets.max(1)   # (len(test_dataset), 1)
    _, predicted = avg_predictions.max(1) # (len(test_dataset), 1)
    
    y_true = all_targets.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    accuracy = report_dict.get("accuracy", 0) * 100
 
    avg_loss = total_loss / len(y_true) if criterion else None
    f1 = f1_score(y_true, y_pred, average='weighted')

    log_message = f"\n{'=' * 50}\n{prefix} (with TTA - {len(tta_transforms)} transforms)\n{'=' * 50}\n" \
                  f"Accuracy : {accuracy:.2f}%\n"
    if criterion:
        log_message += f"Average Loss: {avg_loss:.4f}\n"
    log_message += f"F1 Score (Weighted): {f1:.4f}\n\nClassification Report:\n"
    log_message += classification_report(y_true, y_pred, target_names=class_names)
    
    print(log_message)
    logging.info(log_message)

    if class_names:
        save_path = f"{path}/{prefix.replace(' ', '_')}_TTA_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=save_path)
    
    metrics = {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "f1_score": f1,
        "classification_report": report_dict,
    }
    return metrics