import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score
import argparse
from pathlib import Path

from src.data.dataset import FFDIDataset, create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model import DualBranchModel
from src.models.loss import LabelSmoothingCrossEntropy
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.helpers import (
    AverageMeter, 
    save_checkpoint, 
    load_checkpoint, 
    setup_logger,
    seed_everything
)
from configs.config import TrainConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train Fake Face Detection Model')
    parser.add_argument('--config', type=str, default='configs/config.py',
                        help='path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    return parser.parse_args()

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: torch.device,
                epoch: int,
                logger) -> float:
    """
    训练一个epoch
    """
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (img, freq, high_freq, labels) in enumerate(pbar):
        # 移动数据到设备
        img = img.to(device)
        freq = freq.to(device)
        high_freq = high_freq.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(img, freq, high_freq)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 更新统计
        losses.update(loss.item(), img.size(0))
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # 记录到wandb
        if batch_idx % 100 == 0:
            wandb.log({
                'train_loss': losses.avg,
                'learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch
            })
    
    logger.info(f'Epoch {epoch+1} - Training Loss: {losses.avg:.4f}')
    return losses.avg

def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device,
            epoch: int,
            logger) -> tuple:
    """
    验证模型
    """
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for img, freq, high_freq, labels in tqdm(val_loader, desc='Validation'):
            img = img.to(device)
            freq = freq.to(device)
            high_freq = high_freq.to(device)
            labels = labels.to(device)
            
            outputs = model(img, freq, high_freq)
            loss = criterion(outputs, labels)
            
            losses.update(loss.item(), img.size(0))
            
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 计算指标
    metrics = compute_metrics(all_targets, all_predictions, np.array(all_probabilities))
    auc = roc_auc_score(all_targets, np.array(all_probabilities)[:, 1])
    
    # 记录到wandb
    wandb.log({
        'val_loss': losses.avg,
        'val_auc': auc,
        'val_accuracy': metrics['classification_report']['accuracy'],
        'epoch': epoch
    })
    
    logger.info(f'Epoch {epoch+1} - Validation Loss: {losses.avg:.4f}, AUC: {auc:.4f}')
    print_metrics(metrics)
    
    return losses.avg, auc, metrics

def main():
    args = parse_args()
    config = TrainConfig()
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 设置日志
    logger = setup_logger(config.log_dir)
    logger.info("Starting training with config:")
    logger.info(str(config.__dict__))
    
    # 初始化wandb
    wandb.init(
        project="fake-face-detection",
        config=config.__dict__
    )
    
    # 创建保存目录
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 准备数据
    train_df = pd.read_csv(config.train_data_path)
    val_df = pd.read_csv(config.val_data_path)
    
    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        img_dir=config.train_img_dir,
        train_transform=get_train_transforms(config.img_size),
        val_transform=get_val_transforms(config.img_size),
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # 创建模型
    model = DualBranchModel.create_model(
        num_classes=config.num_classes,
        model_name=config.model_name
    ).to(config.device)
    
    # 损失函数和优化器
    criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 恢复检查点
    start_epoch = 0
    best_auc = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_auc = load_checkpoint(args.resume, model, optimizer)
    
    # 训练循环
    logger.info("Starting training...")
    patience = 0
    for epoch in range(start_epoch, config.num_epochs):
        # 训练一个epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, 
            scheduler, config.device, epoch, logger
        )
        
        # 验证
        val_loss, val_auc, metrics = validate(
            model, val_loader, criterion, 
            config.device, epoch, logger
        )
        
        # 保存检查点
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            patience = 0
        else:
            patience += 1
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_auc': best_auc,
            'optimizer': optimizer.state_dict(),
        }, is_best, config.model_save_path)
        
        # 早停
        if patience >= config.early_stop_patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    logger.info(f"Training completed. Best AUC: {best_auc:.4f}")
    wandb.finish()

if __name__ == '__main__':
    main()