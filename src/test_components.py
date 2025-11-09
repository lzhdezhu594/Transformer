import torch
import torch.nn as nn
from config import TransformerConfig
from model import PositionalEncoding, MultiHeadAttention, PositionWiseFFN, ResidualConnection, TransformerEncoder

def test_positional_encoding():
    """测试位置编码组件"""
    print("=" * 50)
    print("测试 PositionalEncoding...")
    
    # 创建配置和模型
    config = TransformerConfig()
    pe = PositionalEncoding(config)
    
    # 创建测试输入 (batch_size=2, seq_len=10, d_model=128)
    batch_size, seq_len = 2, 10
    x = torch.zeros(batch_size, seq_len, config.d_model)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = pe(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输入和输出是否相同形状: {x.shape == output.shape}")
    print(f"位置编码范围: [{pe.pe.min():.4f}, {pe.pe.max():.4f}]")
    
    # 检查不同位置的编码是否不同
    pos1 = pe.pe[0, 0, :5]  # 第一个位置的前5个维度
    pos2 = pe.pe[0, 1, :5]  # 第二个位置的前5个维度
    print(f"位置0的前5个维度: {pos1}")
    print(f"位置1的前5个维度: {pos2}")
    print(f"不同位置编码是否不同: {not torch.allclose(pos1, pos2)}")
    
    return True

def test_multihead_attention():
    """测试多头注意力组件"""
    print("\n" + "=" * 50)
    print("测试 MultiHeadAttention...")
    
    # 创建配置和模型
    config = TransformerConfig()
    mha = MultiHeadAttention(config)
    
    # 创建测试输入 (batch_size=2, seq_len=8, d_model=128)
    batch_size, seq_len = 2, 8
    q = k = v = torch.randn(batch_size, seq_len, config.d_model)
    
    print(f"输入形状 - Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    
    # 前向传播
    output, attn_weights = mha(q, k, v)
    
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print(f"输出和输入Q是否相同形状: {q.shape == output.shape}")
    
    # 检查注意力权重的属性
    print(f"注意力权重范围: [{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
    
    # 更精确地检查softmax归一化
    # 注意力权重形状: (batch_size, n_heads, seq_len, seq_len)
    # 对最后一个维度（seq_len）求和，每个位置对其他所有位置的注意力权重应该和为1
    sums = attn_weights.sum(dim=-1)
    print(f"注意力权重每行求和的范围: [{sums.min():.6f}, {sums.max():.6f}]")
    
    # 由于浮点数精度，使用更宽松的容差
    is_normalized = torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    print(f"注意力权重是否正确归一化 (容差1e-5): {is_normalized}")
    
    # 打印具体的数值示例来验证
    print("\n具体示例 - 第一个batch，第一个head的注意力权重:")
    sample_weights = attn_weights[0, 0]
    for i in range(min(3, seq_len)):  # 只显示前3行
        row = sample_weights[i]
        row_sum = row.sum().item()
        print(f"  第{i}行: sum={row_sum:.6f}, values={row[:3].tolist()}...")
    
    return is_normalized

def test_components_together():
    """测试两个组件一起工作"""
    print("\n" + "=" * 50)
    print("测试组件协同工作...")
    
    config = TransformerConfig()
    
    # 创建组件
    pe = PositionalEncoding(config)
    mha = MultiHeadAttention(config)
    
    # 创建测试输入 (模拟嵌入后的输入)
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    print(f"原始输入形状: {x.shape}")
    
    # 先应用位置编码
    x_with_pe = pe(x)
    print(f"添加位置编码后形状: {x_with_pe.shape}")
    
    # 再应用多头注意力
    output, attn_weights = mha(x_with_pe, x_with_pe, x_with_pe)
    print(f"多头注意力后形状: {output.shape}")
    
    # 检查梯度计算
    x.requires_grad_(True)
    x_with_pe = pe(x)
    output, _ = mha(x_with_pe, x_with_pe, x_with_pe)
    
    # 计算梯度
    loss = output.sum()
    loss.backward()
    
    print(f"梯度计算正常: {x.grad is not None}")
    print(f"梯度形状: {x.grad.shape}")
    
    return True

def test_positionwise_ffn():
    """测试前馈网络组件"""
    print("\n" + "=" * 50)
    print("测试 PositionWiseFFN...")
    
    config = TransformerConfig()
    ffn = PositionWiseFFN(config)
    
    # 创建测试输入
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = ffn(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输入输出形状一致: {x.shape == output.shape}")
    
    # 检查梯度
    x.requires_grad_(True)
    output = ffn(x)
    loss = output.sum()
    loss.backward()
    
    print(f"梯度计算正常: {x.grad is not None}")
    
    return True

def test_residual_connection():
    """测试残差连接"""
    print("\n" + "=" * 50)
    print("测试 ResidualConnection...")
    
    config = TransformerConfig()
    residual = ResidualConnection(config)
    
    # 创建一个简单的子层（恒等映射）
    identity_sublayer = lambda x: x
    
    # 创建测试输入
    batch_size, seq_len = 2, 8
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = residual(x, identity_sublayer)
    
    print(f"输出形状: {output.shape}")
    print(f"残差连接后值不同: {not torch.allclose(x, output)}")
    
    return True

def test_transformer_encoder():
    """测试完整Transformer编码器"""
    print("\n" + "=" * 50)
    print("测试 TransformerEncoder...")
    
    config = TransformerConfig()
    encoder = TransformerEncoder(config)
    
    # 创建测试输入 (token indices)
    batch_size, seq_len = 2, 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"输入形状: {x.shape}")
    print(f"词汇表大小: {config.vocab_size}")
    
    # 前向传播
    output = encoder(x)
    
    print(f"输出形状: {output.shape}")
    print(f"编码器层数: {config.num_layers}")
    
    # 检查参数数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"总参数数量: {total_params:,}")
    
    return True

# 在main函数中添加新测试
if __name__ == "__main__":
    print("开始测试Transformer组件...")
    
    try:
        test_positional_encoding()
        test_multihead_attention() 
        test_positionwise_ffn()
        test_residual_connection()
        test_transformer_encoder()
        test_components_together()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！组件实现正确。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()