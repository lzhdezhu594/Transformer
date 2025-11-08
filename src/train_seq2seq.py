#!/usr/bin/env python3
"""
序列到序列的Transformer训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import os
from tqdm import tqdm

from config import TransformerConfig
from model import FullTransformer, create_attention_mask
from data_loader import get_data_loader
from utils import save_checkpoint, load_checkpoint, plot_training_curves

class Seq2SeqTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 数据加载
        self.train_loader, self.vocab_size, self.char_to_idx, self.idx_to_char = get_data_loader(
            batch_size=config.batch_size, 
            seq_length=config.max_seq_len
        )
        
        # 更新词汇表大小
        config.vocab_size = self.vocab_size
        
        # 完整Transformer模型
        self.model = FullTransformer(config).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        
        print(f"序列到序列训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_seq2seq_data(self, data):
        """
        为序列到序列任务准备数据
        简单起见，我们使用复制任务：模型学习复制输入序列
        """
        # 源序列：去掉最后一个token
        src_input = data[:, :-1]
        # 目标序列：去掉第一个token（用于teacher forcing）
        tgt_input = data[:, :-1]
        # 标签：去掉第一个token
        labels = data[:, 1:]
        
        return src_input, tgt_input, labels
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)
            
            # 准备序列到序列数据
            src_input, tgt_input, labels = self.prepare_seq2seq_data(data)
            
            # 创建目标序列的注意力掩码
            tgt_mask = create_attention_mask(tgt_input.size(1), self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            loss, _ = self.model(src_input, tgt_input, labels=labels, tgt_mask=tgt_mask)
            
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
            for data, _ in self.train_loader:
                data = data.to(self.device)
                
                # 准备序列到序列数据
                src_input, tgt_input, labels = self.prepare_seq2seq_data(data)
                
                # 创建目标序列的注意力掩码
                tgt_mask = create_attention_mask(tgt_input.size(1), self.device)
                
                loss, _ = self.model(src_input, tgt_input, labels=labels, tgt_mask=tgt_mask)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self):
        """完整的训练循环"""
        print("开始序列到序列训练...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 更新学习率
            self.scheduler.step()
            
            # 评估
            eval_loss = self.evaluate()
            
            print(f"Epoch {epoch}: 训练损失 = {train_loss:.4f}, 评估损失 = {eval_loss:.4f}")
            
            # 保存最佳模型
            if eval_loss < self.best_loss:
                self.best_loss = eval_loss
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_loss': self.best_loss,
                    'train_losses': self.train_losses,
                    'config': self.config
                }, f'results/full_transformer_best.pth')
                print(f"保存最佳完整Transformer模型，评估损失: {eval_loss:.4f}")
            
            # 每几个epoch保存一次检查点
            if epoch % 5 == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_loss': self.best_loss,
                    'train_losses': self.train_losses,
                    'config': self.config
                }, f'results/full_transformer_epoch_{epoch}.pth')
        
        training_time = time.time() - start_time
        print(f"训练完成! 总时间: {training_time:.2f}秒")
        
        # 绘制训练曲线
        plot_training_curves(self.train_losses, save_path='results/full_transformer_training_curve.png')
        
        return self.train_losses

def main():
    # 创建配置
    config = TransformerConfig()
    config.use_full_transformer = True
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 创建训练器并开始训练
    trainer = Seq2SeqTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()