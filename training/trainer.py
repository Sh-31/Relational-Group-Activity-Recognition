import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
import random
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(
        self,
        model,
        config,
        device=None,
        task='group_activity',
        checkpoint_path=None,
        experiment_name: str = "",
        logger_step: int = 100,
        output_dir: str = ""
    ):
        self.model = model
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.best_val_acc = 0
        self.start_epoch = 0
        self.logger_step = logger_step
        self.task = task
        
        # Setup optimizer
        if self.config.training[task]['optimizer'] == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training[task]['learning_rate'],
                weight_decay=self.config.training[task]['weight_decay']
            )
        elif self.config.training[task]['optimizer'] == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training[task]['learning_rate'],
                momentum=self.config.training[task].get('momentum', 0),
                weight_decay=self.config.training[task]['weight_decay'],
                nesterov=True
            )
        
        # Setup criterion
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.training[task]['label_smoothing']
        )
        
        # Setup gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-6
        )
        
        # Setup experiment directory and logging
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        
            exp_name = experiment_name or self.config.experiment['name']
            version = self.config.experiment['version']
            
            self.exp_dir = os.path.join(
                output_dir,
                f"{exp_name}_V{version}_{timestamp}"
            )
            
            os.makedirs(self.exp_dir, exist_ok=True)
            
        self.logger = self.setup_logging(self.exp_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
        
        # Log configuration
        self.logger.info(f"Starting experiment: {self.config.experiment['name']}_V{self.config.experiment['version']}")
        self.logger.info(f"Using optimizer: {self.config.training[task]['optimizer']}, "
                f"lr: {self.config.training[task]['learning_rate']}, "
                f"momentum: {self.config.training[task].get('momentum', 0)}, "
                f"weight_decay: {self.config.training[task]['weight_decay']}")
        self.logger.info(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        self.set_seed(self.config.experiment['seed'])
        self.logger.info(f"Set random seed: {self.config.experiment['seed']}")
        
        # Save config
        config_save_path = os.path.join(self.exp_dir, 'config.yml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
    
    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_logging(self, exp_dir):
        """Setup logging to file and console."""
        import logging
        
        logger = logging.getLogger('trainer')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers if any
        if logger.handlers:
            logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(exp_dir, 'training.log'))
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train_one_epoch(self, train_loader, epoch):
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            predicted = outputs.argmax(1)
            target_class = targets.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(target_class).sum().item()
            
            if batch_idx % self.logger_step == 0 and batch_idx != 0:
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Training/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Training/BatchAccuracy', 100.*correct/total, step)
                
                log_msg = f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%'
                self.logger.info(log_msg)
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        self.writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
        self.writer.add_scalar('Training/EpochAccuracy', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, epoch, class_names):
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                _, target_class = targets.max(1)
                total += targets.size(0)
                correct += predicted.eq(target_class).sum().item()
                
                y_true.extend(target_class.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, average="weighted")
        self.writer.add_scalar('Validation/F1Score', f1, epoch)
        
        # Plot confusion matrix
        fig = self.plot_confusion_matrix(y_true, y_pred, class_names)
        self.writer.add_figure('Validation/ConfusionMatrix', fig, epoch)
        
        self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
        self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)
        
        return avg_loss, accuracy, f1
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        return fig
    
    def save_checkpoint(self, epoch, val_acc):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.exp_dir, f'checkpoint_epoch_{epoch}.pkl')
        torch.save(checkpoint, checkpoint_path)     
        self.logger.info(f"Saved checkpoint at {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            self.logger.info(f"Checkpoint {checkpoint_path} does not exist.")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_acc = checkpoint.get('val_acc', 0)
        
        loaded_config = checkpoint.get('config', None)
        if loaded_config:
            self.config = loaded_config
        
        self.exp_dir = os.path.dirname(checkpoint_path)
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resumed training from epoch {self.start_epoch}")
    
    def train(self, train_dataset, val_dataset, class_names, collate_fn=None):
        """Train model for multiple epochs."""
        
        self.logger.info("Starting training...")
        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")

        batch_size = self.config.training[self.task]["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

        for epoch in range(self.start_epoch, self.config.training[self.task]["epochs"]):
            self.logger.info(f'\nEpoch {epoch+1}/{self.config.training[self.task]["epochs"]}')
            
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            
            val_loss, val_acc, val_f1_score = self.validate(val_loader, epoch, class_names)

            self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
            self.logger.info(f"Epoch {epoch} | Valid Loss: {val_loss:.4f} | Valid Accuracy: {val_acc:.2f}% | Valid F1 Score: {val_f1_score:.4f}")
            
            self.scheduler.step(val_loss)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                
            self.save_checkpoint(epoch, val_acc)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
            self.logger.info(f'Current learning rate: {current_lr}')
        
        self.writer.close()
        self.logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.best_val_acc

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()


class DDPTrainer:
    def __init__(
        self,
        model,
        config,
        rank,
        world_size,
        task='group_activity',
        checkpoint_path=None,
        class_weights=None,
        experiment_name: str = "",
        logger_step: int = 100,
        output_dir: str = ""
    ):
        self.model = model
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[rank])
        self.best_val_acc = 0
        self.start_epoch = 0
        self.logger_step = logger_step
        self.task = task
        self.is_main_process = (self.rank == 0)
        self.class_weights = class_weights
        
        # Setup optimizer
        if self.config.training[task]['optimizer'] == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training[task]['learning_rate'],
                weight_decay=self.config.training[task]['weight_decay']
            )
        elif self.config.training[task]['optimizer'] == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training[task]['learning_rate'],
                momentum=self.config.training[task].get('momentum', 0),
                weight_decay=self.config.training[task]['weight_decay'],
                nesterov=True
            )
        
        # Setup criterion
        if self.config.training[task]['balance']:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.training[task]['label_smoothing'],
                weight=self.class_weights.to(self.device)
            )
        else:    
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.training[task]['label_smoothing']
            )
        
        # Setup gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-6
        )
        
        # Setup experiment directory and logging (only on main process)
        if self.is_main_process:
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)
            else:
                timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
            
                exp_name = experiment_name or self.config.experiment['name']
                version = self.config.experiment['version']
                
                self.exp_dir = os.path.join(
                    output_dir,
                    f"{exp_name}_V{version}_{timestamp}"
                )
                
                os.makedirs(self.exp_dir, exist_ok=True)
                
            self.logger = self.setup_logging(self.exp_dir)
            self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
            
            # Log configuration
            self.logger.info(f"Starting experiment: {self.config.experiment['name']}_V{self.config.experiment['version']}")
            self.logger.info(f"Using optimizer: {self.config.training[task]['optimizer']}, "
                    f"lr: {self.config.training[task]['learning_rate']}, "
                    f"momentum: {self.config.training[task].get('momentum', 0)}, "
                    f"weight_decay: {self.config.training[task]['weight_decay']}")
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"World Size: {self.world_size}")
            
            # Save config
            config_save_path = os.path.join(self.exp_dir, 'config.yml')
            with open(config_save_path, 'w') as f:
                yaml.dump(config, f)
        else:
            self.logger = None
            self.writer = None
            self.exp_dir = None
        
        # Set random seed for reproducibility
        self.set_seed(self.config.experiment['seed'] + rank)  # Add rank to seed to ensure different seeds across processes
        if self.is_main_process:
            self.logger.info(f"Set random seed: {self.config.experiment['seed']}")
        
    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_logging(self, exp_dir):
        """Setup logging to file and console."""
        import logging
        
        logger = logging.getLogger(f'trainer_rank{self.rank}')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers if any
        if logger.handlers:
            logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(exp_dir, f'training_rank{self.rank}.log'))
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train_one_epoch(self, train_loader, epoch):
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Set epoch for the DistributedSampler
        train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            predicted = outputs.argmax(1)
            target_class = targets.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(target_class).sum().item()
            
            if self.is_main_process and batch_idx % self.logger_step == 0 and batch_idx != 0:
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Training/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Training/BatchAccuracy', 100.*correct/total, step)
                
                log_msg = f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%'
                self.logger.info(log_msg)
        
        # Gather metrics from all processes
        metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=self.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        total_loss = metrics[0].item()
        correct = metrics[1].item()
        total = metrics[2].item()
        
        epoch_loss = total_loss / (len(train_loader) * self.world_size)
        epoch_acc = 100. * correct / total
        
        if self.is_main_process:
            self.writer.add_scalar('Training/EpochLoss', epoch_loss, epoch)
            self.writer.add_scalar('Training/EpochAccuracy', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, epoch, class_names):
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                _, target_class = targets.max(1)
                total += targets.size(0)
                correct += predicted.eq(target_class).sum().item()
                
                # Collect predictions for confusion matrix (only on main process)
                if self.is_main_process:
                    y_true.extend(target_class.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
        
        # Gather metrics from all processes
        metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=self.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        total_loss = metrics[0].item()
        correct = metrics[1].item()
        total = metrics[2].item()
        
        avg_loss = total_loss / (len(val_loader) * self.world_size)
        accuracy = 100. * correct / total
        
        if self.is_main_process:
            # Calculate F1 score
            from sklearn.metrics import f1_score
            f1 = f1_score(y_true, y_pred, average="weighted")
            self.writer.add_scalar('Validation/F1Score', f1, epoch)
            
            # Plot confusion matrix
            fig = self.plot_confusion_matrix(y_true, y_pred, class_names)
            self.writer.add_figure('Validation/ConfusionMatrix', fig, epoch)
            
            self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
            self.writer.add_scalar('Validation/Accuracy', accuracy, epoch)
            
            return avg_loss, accuracy, f1
        
        return avg_loss, accuracy, 0.0
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        return fig
    
    def save_checkpoint(self, epoch, val_acc):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
            
        # Save only the model parameters, not the DDP wrapper
        model_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in self.model.module.state_dict().items()}
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.exp_dir, f'checkpoint_epoch_{epoch}.pkl')
        torch.save(checkpoint, checkpoint_path)     
        self.logger.info(f"Saved checkpoint at {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            if self.is_main_process:
                self.logger.info(f"Checkpoint {checkpoint_path} does not exist.")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load the model state dict to the DDP model
        if 'model_state_dict' in checkpoint:
            # Handle both DDP and non-DDP saved models
            state_dict = checkpoint['model_state_dict']
            if not any(k.startswith('module.') for k in state_dict.keys()):
                # If not saved with DDP, we need to add 'module.' prefix
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_acc = checkpoint.get('val_acc', 0)
        
        loaded_config = checkpoint.get('config', None)
        if loaded_config:
            self.config = loaded_config
        
        self.exp_dir = os.path.dirname(checkpoint_path)
        
        if self.is_main_process:
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            self.logger.info(f"Resumed training from epoch {self.start_epoch}")
    
    def train(self, train_dataset, val_dataset, class_names, collate_fn=None):
        """Train model for multiple epochs."""
        
        if self.is_main_process:
            self.logger.info("Starting training...")
            self.logger.info(f"Training dataset size: {len(train_dataset)}")
            self.logger.info(f"Validation dataset size: {len(val_dataset)}")

        batch_size = self.config.training[self.task]["batch_size"]
        
        # Setup DistributedSampler for both datasets
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # Create dataloaders with the distributed samplers
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

        for epoch in range(self.start_epoch, self.config.training[self.task]["epochs"]):
            if self.is_main_process:
                self.logger.info(f'\nEpoch {epoch+1}/{self.config.training[self.task]["epochs"]}')
            
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            
            val_loss, val_acc, val_f1_score = self.validate(val_loader, epoch, class_names)

            if self.is_main_process:
                self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
                self.logger.info(f"Epoch {epoch} | Valid Loss: {val_loss:.4f} | Valid Accuracy: {val_acc:.2f}% | Valid F1 Score: {val_f1_score:.4f}")
                
                # Update learning rate scheduler (only on main process)
                self.scheduler.step(val_loss)
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    
                self.save_checkpoint(epoch, val_acc)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
                self.logger.info(f'Current learning rate: {current_lr}')
            
            # Broadcast new learning rate to all processes
            lr_tensor = torch.tensor([self.optimizer.param_groups[0]['lr']], dtype=torch.float32, device=self.device)
            dist.broadcast(lr_tensor, 0)
            
            if not self.is_main_process:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_tensor.item()
        
        if self.is_main_process:
            self.writer.close()
            self.logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.best_val_acc


def run_ddp_training(
    rank, 
    world_size, 
    model, 
    config, 
    train_dataset, 
    val_dataset, 
    class_names, 
    class_weights,
    checkpoint_path=None, 
    logger_step=100,
    experiment_name="", 
    output_dir="",
    collate_fn=None,
    task='group_activity'
):
    """
    Run training on a single process.
    
    Args:
        rank: The rank of the current process
        world_size: Total number of processes
        model: model instance
        config: Training configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        class_names: List of class names
        checkpoint_path: Path to checkpoint to resume training from
        experiment_name: Name of the experiment
        output_dir: Directory to save outputs
        collate_fn: Collate function for the dataloaders
        task: Task name for configuration lookup
    """
    # Initialize the distributed environment
    setup(rank, world_size)
    
    # Create trainer
    trainer = DDPTrainer(
        model=model,
        config=config,
        rank=rank,
        world_size=world_size,
        task=task,
        class_weights=class_weights,
        checkpoint_path=checkpoint_path,
        experiment_name=experiment_name,
        output_dir=output_dir,
        logger_step=logger_step
    )
    
    # Train the model
    best_acc = trainer.train(train_dataset, val_dataset, class_names, collate_fn)
    
    # Clean up
    cleanup()
    
    return best_acc


def ddp_train(
    model, 
    config, 
    train_dataset, 
    val_dataset, 
    class_names, 
    checkpoint_path=None, 
    experiment_name="", 
    output_dir="",
    logger_step=100,
    class_weights=None,
    collate_fn=None,
    task='group_activity',
    world_size=None
):
    """
    Main function to start distributed training.
    
    Args:
        model: model instance
        config: Training configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        class_names: List of class names
        checkpoint_path: Path to checkpoint to resume training from
        experiment_name: Name of the experiment
        output_dir: Directory to save outputs
        collate_fn: Collate function for the dataloaders
        task: Task name for configuration lookup
        world_size: Number of processes to use (defaults to number of GPUs)
    """
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    if world_size <= 0:
        raise ValueError(f"No CUDA devices found or invalid world_size: {world_size}")
    
    print(f"Starting distributed training with {world_size} GPUs")
    
    mp.spawn(
        run_ddp_training,
        args=(world_size, model, config, train_dataset, val_dataset, class_names, class_weights, checkpoint_path, logger_step, experiment_name, output_dir, collate_fn, task),
        nprocs=world_size,
        join=True
    )     