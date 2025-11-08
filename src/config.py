import torch

class TransformerConfig:
    def __init__(self):
        # 模型参数
        self.vocab_size = 66  # Tiny Shakespeare 的字符数
        self.d_model = 128    # 模型维度
        self.n_heads = 4      # 注意力头数
        self.num_layers = 2   # Transformer层数
        self.d_ff = 512       # FFN隐藏层维度
        self.max_seq_len = 256 # 最大序列长度
        self.dropout = 0.1    # Dropout率
        
        # 训练参数
        self.batch_size = 64
        self.learning_rate = 0.001
        self.num_epochs = 20
        self.grad_clip = 1.0

        # 消融实验参数
        self.use_residual = True
        self.use_layernorm = True
        self.use_positional_encoding = True
        self.use_attention = True
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 新增：序列到序列任务相关
        self.use_full_transformer = True  # 是否使用完整的Encoder-Decoder
        
    def __str__(self):
        return f"TransformerConfig: d_model={self.d_model}, n_heads={self.n_heads}, num_layers={self.num_layers}, use_full_transformer={self.use_full_transformer}"

class AblationConfig:
    """消融实验配置"""
    
    @staticmethod
    def get_ablation_variants():
        """返回所有消融实验变体配置"""
        return {
            "baseline": {
                "description": "完整模型（基线）",
                "changes": {}
            },
            "no_residual": {
                "description": "移除残差连接",
                "changes": {"use_residual": False}
            },
            "no_positional": {
                "description": "移除位置编码",
                "changes": {"use_positional_encoding": False}
            },
            "single_head": {
                "description": "单头注意力",
                "changes": {"n_heads": 1}
            },
            "small_ffn": {
                "description": "减小FFN维度",
                "changes": {"d_ff": 128}
            },
            "shallow": {
                "description": "浅层模型", 
                "changes": {"num_layers": 1}
            },
            "no_attention": {
                "description": "无注意力机制",
                "changes": {"use_attention": False}
            }
        }