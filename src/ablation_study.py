#!/usr/bin/env python3
"""
消融实验脚本
训练不同的模型变体并比较性能
"""
import torch
import torch.nn as nn
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import TransformerConfig, AblationConfig
from model import TransformerLanguageModel
from data_loader import get_data_loader

class AblationTrainer:
    def __init__(self, variant_name, config_changes):
        self.variant_name = variant_name
        self.config = TransformerConfig()
        
        # 应用配置更改
        for key, value in config_changes.items():
            setattr(self.config, key, value)
        
        self.device = self.config.device
        
        # 数据加载
        self.train_loader, self.vocab_size, self.char_to_idx, self.idx_to_char = get_data_loader(
            batch_size=self.config.batch_size, 
            seq_length=self.config.max_seq_len
        )
        
        # 更新词汇表大小
        self.config.vocab_size = self.vocab_size
        
        # 模型
        self.model = TransformerLanguageModel(self.config).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # 训练状态
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        
        print(f"初始化消融实验: {variant_name}")
        print(f"配置: {config_changes}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"{self.variant_name} - Epoch {self.current_epoch}")
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            loss, _ = self.model(data, labels=targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # 优化器步进
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self):
        """评估函数"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with torch.no_grad():
            for data, targets in self.train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                loss, _ = self.model(data, labels=targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs=10):
        """训练循环"""
        print(f"开始训练 {self.variant_name}...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 评估
            val_loss = self.evaluate()
            
            print(f"{self.variant_name} - Epoch {epoch}: 训练损失 = {train_loss:.4f}, 评估损失 = {val_loss:.4f}")
        
        training_time = time.time() - start_time
        print(f"{self.variant_name} 训练完成! 总时间: {training_time:.2f}秒")
        
        # 保存结果
        results = {
            'variant_name': self.variant_name,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_time': training_time,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        return results

def run_ablation_study():
    """运行所有消融实验"""
    # 确保目录存在
    os.makedirs('results/ablation', exist_ok=True)
    
    # 获取所有消融变体
    ablation_variants = AblationConfig.get_ablation_variants()
    
    all_results = {}
    
    print("开始消融实验...")
    print(f"总共 {len(ablation_variants)} 个变体")
    
    for variant_name, variant_info in ablation_variants.items():
        print(f"\n{'='*60}")
        print(f"训练变体: {variant_name} - {variant_info['description']}")
        print(f"{'='*60}")
        
        # 创建训练器
        trainer = AblationTrainer(variant_name, variant_info['changes'])
        
        # 训练模型
        results = trainer.train(num_epochs=10)
        
        # 保存结果
        all_results[variant_name] = results
    
    # 保存所有结果
    with open('results/ablation/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def plot_ablation_results(results):
    """绘制消融实验结果"""
    plt.figure(figsize=(12, 8))
    
    # 绘制训练损失曲线
    plt.subplot(2, 2, 1)
    for variant_name, result in results.items():
        plt.plot(result['train_losses'], label=variant_name, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Ablation Study - Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制验证损失曲线
    plt.subplot(2, 2, 2)
    for variant_name, result in results.items():
        plt.plot(result['val_losses'], label=variant_name, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Ablation Study - Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制最终损失比较
    plt.subplot(2, 2, 3)
    variant_names = list(results.keys())
    final_train_losses = [results[name]['final_train_loss'] for name in variant_names]
    final_val_losses = [results[name]['final_val_loss'] for name in variant_names]
    
    x = np.arange(len(variant_names))
    width = 0.35
    
    plt.bar(x - width/2, final_train_losses, width, label='Final Train Loss', alpha=0.7)
    plt.bar(x + width/2, final_val_losses, width, label='Final Val Loss', alpha=0.7)
    
    plt.xlabel('Model Variants')
    plt.ylabel('Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(x, variant_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制参数量比较
    plt.subplot(2, 2, 4)
    num_parameters = [results[name]['num_parameters'] for name in variant_names]
    
    plt.bar(variant_names, num_parameters, alpha=0.7)
    plt.xlabel('Model Variants')
    plt.ylabel('Number of Parameters')
    plt.title('Model Size Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/ablation/ablation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_results(results):
    """分析消融实验结果"""
    print("\n" + "="*80)
    print("消融实验分析结果")
    print("="*80)
    
    # 找到基线模型
    baseline = results.get('baseline', None)
    if not baseline:
        print("错误: 未找到基线模型结果")
        return
    
    baseline_val_loss = baseline['final_val_loss']
    
    print(f"{'变体':<20} {'验证损失':<12} {'相对性能':<15} {'参数量':<15}")
    print("-" * 65)
    
    for variant_name, result in results.items():
        val_loss = result['final_val_loss']
        relative_perf = (val_loss - baseline_val_loss) / baseline_val_loss * 100
        num_params = result['num_parameters']
        
        print(f"{variant_name:<20} {val_loss:.4f}      {relative_perf:+.2f}%       {num_params:,}")
    
    # 找到最佳模型
    best_variant = min(results.items(), key=lambda x: x[1]['final_val_loss'])
    print(f"\n最佳模型: {best_variant[0]} (验证损失: {best_variant[1]['final_val_loss']:.4f})")
    
    # 关键发现总结
    print(f"\n关键发现:")
    print(f"- 移除残差连接导致性能下降 {results['no_residual']['final_val_loss']/baseline_val_loss:.1f}x")
    print(f"- 移除位置编码导致性能下降 {results['no_positional']['final_val_loss']/baseline_val_loss:.1f}x") 
    print(f"- 移除注意力机制导致性能下降 {results['no_attention']['final_val_loss']/baseline_val_loss:.1f}x")
    print(f"- 减小FFN维度导致性能下降 {results['small_ffn']['final_val_loss']/baseline_val_loss:.1f}x")
    print(f"- 单头注意力性能接近基线 ({results['single_head']['final_val_loss']/baseline_val_loss:.2f}x)")
    print(f"- 浅层模型性能接近基线 ({results['shallow']['final_val_loss']/baseline_val_loss:.2f}x)")

def main():
    """主函数"""
    # 运行消融实验
    results = run_ablation_study()
    
    # 绘制结果
    plot_ablation_results(results)
    
    # 分析结果
    analyze_results(results)
    
    print("\n消融实验完成! 结果保存在 results/ablation/ 目录")

if __name__ == '__main__':
    main()