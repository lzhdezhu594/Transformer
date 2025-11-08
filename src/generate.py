# smart_generate.py
import argparse
import torch
import os
from config import TransformerConfig
from model import TransformerLanguageModel
from data_loader import ShakespeareDataset

def should_stop_generation(generated_text, recent_chars, max_consecutive_same=10, max_consecutive_newlines=3):
    """
    智能判断是否应该停止生成
    """
    # 检查连续相同字符
    if len(recent_chars) >= max_consecutive_same:
        if len(set(recent_chars)) == 1:  # 所有字符都一样
            return True, "重复字符过多"
    
    # 检查连续换行符
    if generated_text.count('\n') > max_consecutive_newlines:
        last_chars = generated_text[-max_consecutive_newlines:]
        if all(c == '\n' for c in last_chars):
            return True, "换行符过多"
    
    # 检查无意义的重复模式
    if len(generated_text) > 50:
        last_50 = generated_text[-50:]
        # 如果最近50个字符中超过80%是同一个字符
        char_counts = {}
        for c in last_50:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_count = max(char_counts.values())
        if max_count / len(last_50) > 0.8:
            return True, "模式重复"
    
    return False, "继续生成"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='results/best_model.pth')
    parser.add_argument('--prompt', default="He's one honest enough: would all the rest were so!")
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--min_length', type=int, default=50, help='最小生成长度')
    args = parser.parse_args()

    config = TransformerConfig()
    ds = ShakespeareDataset()
    device = config.device
    
    model = TransformerLanguageModel(config).to(device)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型: {args.checkpoint}")
    else:
        print(f"错误: 检查点文件不存在")
        return
    
    model.eval()

    # 编码提示文本
    prompt_ids = []
    for c in args.prompt:
        if c in ds.char_to_idx:
            prompt_ids.append(ds.char_to_idx[c])
        else:
            prompt_ids.append(ds.char_to_idx.get(' ', 0))
    
    input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = prompt_ids.copy()

    print(f"提示: {repr(args.prompt)}")
    print(f"参数: temp={args.temperature}, top_k={args.top_k}")
    print("生成中...\n")
    
    recent_chars = []
    
    with torch.no_grad():
        for i in range(args.max_length - len(prompt_ids)):
            logits = model(input_ids)[0][:, -1, :] / args.temperature
            
            # Top-k 采样
            if args.top_k > 0:
                v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 更新输入
            if input_ids.size(1) >= config.max_seq_len:
                input_ids = torch.cat([input_ids[:, 1:], next_token], dim=1)
            else:
                input_ids = torch.cat([input_ids, next_token], dim=1)
            
            next_token_idx = next_token.item()
            generated.append(next_token_idx)
            
            # 更新最近字符（用于停止判断）
            current_char = ds.idx_to_char[next_token_idx]
            recent_chars.append(current_char)
            if len(recent_chars) > 10:  # 只保留最近10个字符
                recent_chars.pop(0)
            
            # 检查是否应该停止（仅在达到最小长度后）
            if len(generated) > args.min_length:
                current_text = ds.decode(generated)
                should_stop, reason = should_stop_generation(current_text, recent_chars)
                if should_stop:
                    print(f"提前停止: {reason}")
                    break
    
    # 最终结果
    full_text = ds.decode(generated)
    print('=' * 60)
    print("生成结果:")
    print('=' * 60)
    print(full_text)
    print('=' * 60)
    
    # 统计信息
    lines = full_text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    print(f"\n统计信息:")
    print(f"总字符数: {len(full_text)}")
    print(f"总行数: {len(lines)}")
    print(f"非空行数: {len(non_empty_lines)}")
    print(f"换行符数量: {full_text.count(chr(10))}")

if __name__ == '__main__':
    main()