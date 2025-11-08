# test_vocab.py
from data_loader import ShakespeareDataset

def test_vocab():
    ds = ShakespeareDataset()
    
    print("=== 词汇表测试 ===")
    print(f"词汇表大小: {ds.vocab_size}")
    print(f"字符列表: {[repr(c) for c in ds.chars]}")
    
    # 测试换行符
    test_text = "Hello\nWorld"
    print(f"\n测试文本: {repr(test_text)}")
    
    encoded = ds.encode(test_text)
    print(f"编码结果: {encoded}")
    
    decoded = ds.decode(encoded)
    print(f"解码结果: {repr(decoded)}")
    
    # 检查换行符
    if '\n' in ds.char_to_idx:
        nl_idx = ds.char_to_idx['\n']
        nl_char = ds.idx_to_char[nl_idx]
        print(f"\n换行符信息:")
        print(f"  索引: {nl_idx}")
        print(f"  字符: {repr(nl_char)}")
    else:
        print("\n警告: 词汇表中没有换行符!")

if __name__ == '__main__':
    test_vocab()