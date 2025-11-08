import torch
from torch.utils.data import Dataset, DataLoader
import requests
import numpy as np

class ShakespeareDataset(Dataset):
    def __init__(self, seq_length=256):
        self.seq_length = seq_length
        self.data = self.load_data()
        self.chars, self.vocab_size, self.char_to_idx, self.idx_to_char = self.build_vocab()
        print(f"词汇表大小: {self.vocab_size}")
        # 使用 repr 来显示字符
        print(f"字符列表: {repr(''.join(self.chars))}")
        
    def load_data(self):
        """加载Tiny Shakespeare数据集"""
        try:
            with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
                text = f.read()
            print(f"原始数据长度: {len(text)}")
            print(f"前500字符: {repr(text[:500])}")
            return text
        except FileNotFoundError:
            print("未找到 tinyshakespeare.txt，使用示例文本")
            # 使用包含换行符的示例文本
            return "Hello World!\nThis is a test.\nAnother line.\n"
        
    def build_vocab(self):
        """构建字符词汇表 - 确保包含换行符"""
        # 首先检查数据中是否包含换行符
        unique_chars = set(self.data)
        print(f"数据中唯一字符数量: {len(unique_chars)}")
        
        # 使用 chr(10) 来表示换行符进行检查
        newline_char = chr(10)
        print(f"是否包含换行符: {newline_char in unique_chars}")
        print(f"是否包含空格: {' ' in unique_chars}")
        
        # 确保包含基本字符
        essential_chars = {newline_char, ' ', '\t'}
        all_chars = unique_chars.union(essential_chars)
        
        chars = sorted(list(all_chars))
        vocab_size = len(chars)
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # 调试信息
        print(f"最终词汇表大小: {vocab_size}")
        print(f"换行符索引: {char_to_idx.get(newline_char, '未找到')}")
        print(f"空格索引: {char_to_idx.get(' ', '未找到')}")
        
        return chars, vocab_size, char_to_idx, idx_to_char
        
    def encode(self, text):
        """将文本编码为索引"""
        try:
            return [self.char_to_idx[ch] for ch in text]
        except KeyError as e:
            print(f"编码错误: 字符 {repr(e.args[0])} 不在词汇表中")
            # 对于未知字符，使用空格代替
            return [self.char_to_idx.get(ch, self.char_to_idx.get(' ', 0)) for ch in text]
        
    def decode(self, indices):
        """将索引解码为文本"""
        return ''.join([self.idx_to_char[idx] for idx in indices])
        
    def __len__(self):
        return len(self.data) // self.seq_length
        
    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1  # +1 for target
        
        chunk = self.data[start_idx:end_idx]
        if len(chunk) < self.seq_length + 1:
            # 填充最后一个chunk（使用空格填充）
            padding = ' ' * (self.seq_length + 1 - len(chunk))
            chunk = chunk + padding
            
        encoded = self.encode(chunk)
        input_seq = torch.tensor(encoded[:-1], dtype=torch.long)
        target_seq = torch.tensor(encoded[1:], dtype=torch.long)
        
        return input_seq, target_seq

def get_data_loader(batch_size=64, seq_length=256):
    """获取数据加载器"""
    dataset = ShakespeareDataset(seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.vocab_size, dataset.char_to_idx, dataset.idx_to_char