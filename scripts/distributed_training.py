import os
import torch
import torch.distributed as dist
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import numpy as np
from tqdm import tqdm
import config
import utils

def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def is_main_process(rank):
    """Check if this is the main process"""
    return rank == 0

def get_deepspeed_config(strategy='ddp', batch_size=16, gradient_accumulation_steps=1):
    """Get DeepSpeed configuration based on strategy"""
    config_map = {
        'ddp': 'deepspeed_configs/ds_config_ddp.json',
        'fsdp_stage2': 'deepspeed_configs/ds_config_fsdp_stage2.json',
        'fsdp_stage3': 'deepspeed_configs/ds_config_fsdp_stage3.json'
    }
    
    if strategy not in config_map:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(config_map.keys())}")
    
    import json
    config_path = config_map[strategy]
    
    with open(config_path, 'r') as f:
        ds_config = json.load(f)
    
    ds_config['train_micro_batch_size_per_gpu'] = batch_size
    ds_config['gradient_accumulation_steps'] = gradient_accumulation_steps
    
    return ds_config

def prepare_model_for_deepspeed(model, strategy='ddp'):
    """Prepare model for DeepSpeed training"""
    if strategy in ['fsdp_stage2', 'fsdp_stage3']:
        for param in model.parameters():
            param.ds_numel = param.numel()
            param.ds_shape = param.shape
            param.ds_status = ZeroParamStatus.NOT_AVAILABLE
    
    return model

def create_deepspeed_optimizer(model_parameters, learning_rate=0.001):
    """Create optimizer optimized for DeepSpeed"""
    return FusedAdam(model_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)

def train_with_deepspeed(model, train_dataset, val_dataset, model_name="model",
                         strategy='ddp', epochs=100, batch_size=16, 
                         gradient_accumulation_steps=1, learning_rate=0.001):
    """Train model with DeepSpeed"""
    
    rank, world_size, local_rank = setup_distributed()
    
    if is_main_process(rank):
        utils.log_message(f"DeepSpeed Training: {model_name}")
        utils.log_message(f"Strategy: {strategy}")
        utils.log_message(f"World Size: {world_size}, Local Rank: {local_rank}")
        utils.log_message(f"Batch size per GPU: {batch_size}")
        utils.log_message(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        utils.log_message(f"Effective batch size: {batch_size * gradient_accumulation_steps * world_size}")
    
    ds_config = get_deepspeed_config(strategy, batch_size, gradient_accumulation_steps)
    
    model = prepare_model_for_deepspeed(model, strategy)
    
    optimizer = create_deepspeed_optimizer(model.parameters(), learning_rate)
    
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
        training_data=train_dataset,
        dist_init_required=False
    )
    
    best_val_accuracy = 0.0
    patience_counter = 0
    max_patience = config.PATIENCE
    
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        model_engine.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        if is_main_process(rank):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = train_loader
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(model_engine.local_rank)
            labels = labels.to(model_engine.local_rank)
            
            outputs = model_engine(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels.argmax(dim=1))
            
            model_engine.backward(loss)
            model_engine.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels.argmax(dim=1)).sum().item()
            
            if is_main_process(rank) and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * train_correct / train_total
                })
        
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = train_correct / train_total
        
        val_loss, val_acc = evaluate_with_deepspeed(
            model_engine, val_dataset, batch_size, local_rank
        )
        
        if is_main_process(rank):
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            utils.log_message(
                f"Epoch {epoch+1}/{epochs} - "
                f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                
                checkpoint_path = utils.get_model_checkpoint_path(model_name)
                model_engine.save_checkpoint(
                    os.path.dirname(checkpoint_path),
                    tag=os.path.basename(checkpoint_path).replace('.keras', '')
                )
                utils.log_message(f"Saved best model with val_acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    utils.log_message(f"Early stopping triggered at epoch {epoch+1}")
                    break
    
    if world_size > 1:
        dist.barrier()
    
    return history

def evaluate_with_deepspeed(model_engine, dataset, batch_size, local_rank):
    """Evaluate model with DeepSpeed"""
    model_engine.eval()
    
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(local_rank)
            labels = labels.to(local_rank)
            
            outputs = model_engine(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels.argmax(dim=1))
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels.argmax(dim=1)).sum().item()
    
    avg_loss = val_loss / len(val_loader)
    accuracy = val_correct / val_total
    
    return avg_loss, accuracy

def create_torch_dataset(images, labels):
    """Create PyTorch dataset from numpy arrays"""
    import torch
    from torch.utils.data import TensorDataset
    
    images_tensor = torch.from_numpy(images).float()
    labels_tensor = torch.from_numpy(labels).float()
    
    return TensorDataset(images_tensor, labels_tensor)

def get_world_size():
    """Get world size for distributed training"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank():
    """Get rank for distributed training"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

