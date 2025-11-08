#!/usr/bin/env python3
"""
序列到序列的Transformer生成脚本
"""
import argparse
import torch
import os
from config import TransformerConfig
from model import FullTransformer
from data_loader import ShakespeareDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='results/full_transformer_best.pth')
    parser.add_argument('--prompt', default="He's one honest enough: would all the rest were so!")
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.8)
    args = parser.parse_args()

    config = TransformerConfig()
    config.use_full_transformer = True
    ds = ShakespeareDataset()
    device = config.device
    
    # 加载完整Transformer模型
    model = FullTransformer(config).to(device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载完整Transformer模型: {args.checkpoint}")
    else:
        print(f"错误: 检查点文件不存在 - {args.checkpoint}")
        return
    
    model.eval()

    # 编码源序列
    src_ids = [ds.char_to_idx.get(c, ds.char_to_idx[' ']) for c in args.prompt]
    src_input = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"源序列: {repr(args.prompt)}")
    print("生成中...\n")
    
    # 生成目标序列
    with torch.no_grad():
        generated_ids = model.generate(src_input, max_length=args.max_length, temperature=args.temperature)
    
    # 解码结果
    generated_text = ds.decode(generated_ids[0].tolist())
    
    print('=' * 60)
    print("序列到序列生成结果:")
    print('=' * 60)
    print(f"输入: {args.prompt}")
    print(f"输出: {generated_text}")
    print('=' * 60)

if __name__ == '__main__':
    main()