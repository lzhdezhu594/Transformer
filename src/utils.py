import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, filename):
    """保存训练检查点"""
    torch.save(state, filename)
    print(f"检查点已保存: {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """加载训练检查点"""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"检查点已加载: {filename}")
        return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))
    else:
        print(f"检查点不存在: {filename}")
        return 0, float('inf')

def plot_training_curves(train_losses, save_path=None):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")
    
    plt.close()

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_info(model):
    """打印模型信息"""
    total_params = count_parameters(model)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")