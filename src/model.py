import torch
import torch.nn as nn
import math
from config import TransformerConfig

class PositionalEncoding(nn.Module):
    """
    正弦位置编码实现
    根据原始论文 "Attention Is All You Need" 中的公式
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, config: TransformerConfig):
        super(PositionalEncoding, self).__init__()
        
        self.config = config
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        
        # 只在启用位置编码时创建相关参数
        if config.use_positional_encoding:
            self.dropout = nn.Dropout(config.dropout)
            
            # 创建位置编码矩阵
            pe = torch.zeros(self.max_seq_len, self.d_model)
            
            # 位置索引 (0 到 max_seq_len-1)
            position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
            
            # 分母项: 10000^(2i/d_model) = exp(2i * -log(10000) / d_model)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2).float() * 
                (-math.log(10000.0) / self.d_model)
            )
            
            # 应用正弦函数到偶数索引
            pe[:, 0::2] = torch.sin(position * div_term)
            # 应用余弦函数到奇数索引
            pe[:, 1::2] = torch.cos(position * div_term)
            
            # 添加批次维度: (1, max_seq_len, d_model)
            pe = pe.unsqueeze(0)
            
            # 将pe注册为buffer（不参与训练的参数）
            self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: 形状为 (batch_size, seq_len, d_model) 的张量
        
        返回:
            添加了位置编码的张量，形状相同
        """
        if self.config.use_positional_encoding:
            seq_len = x.size(1)
            # 添加位置编码（只取前seq_len个位置）
            x = x + self.pe[:, :seq_len, :]
            return self.dropout(x)
        else:
            return x

    def __repr__(self):
        return f"PositionalEncoding(d_model={self.d_model}, max_seq_len={self.max_seq_len}, enabled={self.config.use_positional_encoding})"
    
class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    将输入分割成多个头，分别计算注意力，然后合并
    """
    
    def __init__(self, config: TransformerConfig):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        # 确保d_model可以被n_heads整除
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        # 线性变换层：Q, K, V 和输出
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        缩放点积注意力
        """
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask（如果有）
        if mask is not None:
            # 确保mask是布尔类型，然后转换为与attn_scores相同的数据类型
            if mask.dtype == torch.bool:
                # 对于布尔mask，使用masked_fill
                attn_scores = attn_scores.masked_fill(mask == False, -1e9)
            else:
                # 对于浮点mask，使用原始逻辑
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 应用softmax得到注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
    
    def forward(self, q, k, v, mask=None):
        """
        参数:
            q, k, v: 查询、键、值张量，形状为 (batch_size, seq_len, d_model)
            mask: 可选的注意力mask
        
        返回:
            多头注意力的输出和注意力权重
        """
        batch_size, q_seq_len = q.size(0), q.size(1)
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)
        
        # 线性变换并重塑为多头
        # Q: (batch_size, q_seq_len, d_model) -> (batch_size, q_seq_len, n_heads, d_k) -> (batch_size, n_heads, q_seq_len, d_k)
        q = self.w_q(q).view(batch_size, q_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # K: (batch_size, k_seq_len, d_model) -> (batch_size, k_seq_len, n_heads, d_k) -> (batch_size, n_heads, k_seq_len, d_k)
        k = self.w_k(k).view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # V: (batch_size, v_seq_len, d_model) -> (batch_size, v_seq_len, n_heads, d_k) -> (batch_size, n_heads, v_seq_len, d_k)
        v = self.w_v(v).view(batch_size, v_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算缩放点积注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 转置并合并多头
        # (batch_size, n_heads, q_seq_len, d_k) -> (batch_size, q_seq_len, n_heads, d_k) -> (batch_size, q_seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        
        # 最终线性变换
        output = self.w_o(attn_output)
        
        return output, attn_weights

    def __repr__(self):
        return f"MultiHeadAttention(n_heads={self.n_heads}, d_model={self.d_model}, d_k={self.d_k})"
    
class PositionWiseFFN(nn.Module):
    """
    位置级前馈网络
    由两个线性变换和一个激活函数组成
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, config: TransformerConfig):
        super(PositionWiseFFN, self).__init__()
        
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        
        # 两层线性变换
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        
        # 激活函数和Dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
        
    def forward(self, x):
        """
        参数:
            x: 形状为 (batch_size, seq_len, d_model) 的张量
        
        返回:
            前馈网络的输出，形状相同
        """
        # 第一个线性层 + 激活函数
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 第二个线性层
        x = self.linear2(x)
        
        return x

    def __repr__(self):
        return f"PositionWiseFFN(d_model={self.d_model}, d_ff={self.d_ff})"
    
class ResidualConnection(nn.Module):
    """
    残差连接 + 层归一化
    公式: LayerNorm(x + Sublayer(x))
    """
    
    def __init__(self, config: TransformerConfig):
        super(ResidualConnection, self).__init__()
        
        self.config = config
        
        # 只在启用LayerNorm时创建
        if config.use_layernorm:
            self.layer_norm = nn.LayerNorm(config.d_model)
            
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, sublayer):
        """
        参数:
            x: 输入张量
            sublayer: 子层函数（如注意力或前馈网络）
        
        返回:
            残差连接后的输出
        """
        # 应用残差连接: x + dropout(sublayer(layernorm(x)))
        if self.config.use_residual:
            if self.config.use_layernorm:
                return x + self.dropout(sublayer(self.layer_norm(x)))
            else:
                return x + self.dropout(sublayer(x))
        else:
            if self.config.use_layernorm:
                return self.dropout(sublayer(self.layer_norm(x)))
            else:
                return self.dropout(sublayer(x))

    def __repr__(self):
        return f"ResidualConnection(residual={self.config.use_residual}, layernorm={self.config.use_layernorm})"
    
class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    包含:
    - 多头自注意力 + 残差连接 & 层归一化
    - 前馈网络 + 残差连接 & 层归一化
    """
    
    def __init__(self, config: TransformerConfig):
        super(TransformerEncoderLayer, self).__init__()
        
        self.config = config
        
        # 只在启用注意力时创建注意力层
        if config.use_attention:
            self.self_attention = MultiHeadAttention(config)
            
        self.feed_forward = PositionWiseFFN(config)
        
        # 残差连接
        self.residual1 = ResidualConnection(config)
        self.residual2 = ResidualConnection(config)
        
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入张量，形状 (batch_size, seq_len, d_model)
            mask: 可选的注意力mask
        
        返回:
            编码器层的输出
        """
        # 自注意力子层（如果启用）
        if self.config.use_attention:
            self_attn = lambda x: self.self_attention(x, x, x, mask)[0]
            x = self.residual1(x, self_attn)
        
        # 前馈网络子层
        x = self.residual2(x, self.feed_forward)
        
        return x
        
    def __repr__(self):
        return f"TransformerEncoderLayer(attention={self.config.use_attention})"
    
class TransformerEncoder(nn.Module):
    """
    完整的Transformer编码器
    包含:
    - 输入嵌入层
    - 位置编码
    - 多个编码器层
    """
    
    def __init__(self, config: TransformerConfig):
        super(TransformerEncoder, self).__init__()
        
        self.config = config
        
        # 输入嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(config)
        
        # 编码器层堆叠
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_layers)
        ])
        
        # 输出层归一化（如果启用）
        if config.use_layernorm:
            self.layer_norm = nn.LayerNorm(config.d_model)
        
        # 初始化嵌入层权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.config.d_model**-0.5)
        
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入token索引，形状 (batch_size, seq_len)
            mask: 可选的注意力mask
        
        返回:
            编码器的输出，形状 (batch_size, seq_len, d_model)
        """
        # 嵌入层
        x = self.embedding(x) * math.sqrt(self.config.d_model)
        
        # 位置编码
        x = self.positional_encoding(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 最终层归一化（如果启用）
        if self.config.use_layernorm:
            x = self.layer_norm(x)
        
        return x

    def __repr__(self):
        return f"TransformerEncoder(num_layers={self.config.num_layers}, d_model={self.config.d_model})"
    
class TransformerLanguageModel(nn.Module):
    """
    基于Transformer的语言模型
    用于下一个token预测任务
    """
    
    def __init__(self, config: TransformerConfig):
        super(TransformerLanguageModel, self).__init__()
        
        self.config = config
        self.encoder = TransformerEncoder(config)
        
        # 语言模型头：将编码器输出映射到词汇表
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 与嵌入层共享权重（常见做法）
        self.lm_head.weight = self.encoder.embedding.weight
        
        # 应用缩放初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        # 语言模型头已经与嵌入层共享权重，不需要额外初始化
        pass
        
    def forward(self, input_ids, labels=None):
        """
        参数:
            input_ids: 输入token索引，形状 (batch_size, seq_len)
            labels: 目标token索引，形状 (batch_size, seq_len)，可选
        
        返回:
            如果提供labels: (loss, logits)
            如果不提供labels: (logits,)
        """
        # 通过编码器
        hidden_states = self.encoder(input_ids)
        
        # 语言模型头
        logits = self.lm_head(hidden_states)
        
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            # 修复：使用reshape而不是view
            logits_flat = logits.reshape(-1, self.config.vocab_size)
            labels_flat = labels.reshape(-1)
            loss = loss_fct(logits_flat, labels_flat)
            return loss, logits
        
        return (logits,)
    
    def generate(self, input_ids, max_length=50, temperature=1.0):
        """
        简单的生成函数（自回归生成）
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # 前向传播
                logits = self.forward(input_ids)[0]
                
                # 取最后一个位置的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # 应用softmax得到概率
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # 从分布中采样
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到输入序列
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # 如果生成了结束符，可以提前停止（这里我们简单处理）
                
        return input_ids

    def __repr__(self):
        return f"TransformerLanguageModel(vocab_size={self.config.vocab_size}, num_layers={self.config.num_layers})"
    
    
#################################
# 在 model.py 中添加以下类

class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    包含:
    - 掩码多头自注意力 + 残差连接 & 层归一化
    - 编码器-解码器注意力 + 残差连接 & 层归一化  
    - 前馈网络 + 残差连接 & 层归一化
    """
    
    def __init__(self, config: TransformerConfig):
        super(TransformerDecoderLayer, self).__init__()
        
        # 自注意力（带掩码）
        self.self_attention = MultiHeadAttention(config)
        # 编码器-解码器注意力
        self.cross_attention = MultiHeadAttention(config)
        # 前馈网络
        self.feed_forward = PositionWiseFFN(config)
        
        # 三个残差连接
        self.residual1 = ResidualConnection(config)
        self.residual2 = ResidualConnection(config)
        self.residual3 = ResidualConnection(config)
        
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        参数:
            x: 解码器输入，形状 (batch_size, tgt_seq_len, d_model)
            encoder_output: 编码器输出，形状 (batch_size, src_seq_len, d_model)
            self_attn_mask: 自注意力的掩码，用于避免看到未来信息
            cross_attn_mask: 编码器-解码器注意力的掩码
        """
        # 掩码自注意力子层
        self_attn = lambda x: self.self_attention(x, x, x, self_attn_mask)[0]
        x = self.residual1(x, self_attn)
        
        # 编码器-解码器注意力子层
        cross_attn = lambda x: self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)[0]
        x = self.residual2(x, cross_attn)
        
        # 前馈网络子层
        x = self.residual3(x, self.feed_forward)
        
        return x

    def __repr__(self):
        return f"TransformerDecoderLayer()"

class TransformerDecoder(nn.Module):
    """
    完整的Transformer解码器
    包含:
    - 输出嵌入层
    - 位置编码
    - 多个解码器层
    """
    
    def __init__(self, config: TransformerConfig):
        super(TransformerDecoder, self).__init__()
        
        self.config = config
        
        # 输出嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(config)
        
        # 解码器层堆叠
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config) for _ in range(config.num_layers)
        ])
        
        # 输出层归一化
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # 初始化嵌入层权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.config.d_model**-0.5)
        
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        """
        参数:
            x: 目标序列token索引，形状 (batch_size, tgt_seq_len)
            encoder_output: 编码器输出，形状 (batch_size, src_seq_len, d_model)
            self_attn_mask: 自注意力掩码，用于避免看到未来信息
            cross_attn_mask: 编码器-解码器注意力掩码
        
        返回:
            解码器的输出，形状 (batch_size, tgt_seq_len, d_model)
        """
        # 嵌入层
        x = self.embedding(x) * math.sqrt(self.config.d_model)
        
        # 位置编码
        x = self.positional_encoding(x)
        
        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        
        # 最终层归一化
        x = self.layer_norm(x)
        
        return x

    def __repr__(self):
        return f"TransformerDecoder(num_layers={self.config.num_layers}, d_model={self.config.d_model})"

class FullTransformer(nn.Module):
    """
    完整的Transformer模型（Encoder-Decoder）
    用于序列到序列任务
    """
    
    def __init__(self, config: TransformerConfig):
        super(FullTransformer, self).__init__()
        
        self.config = config
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        
        # 输出线性层
        self.output_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 与解码器嵌入层共享权重
        self.output_layer.weight = self.decoder.embedding.weight
        
        # 应用缩放初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        pass
        
    def forward(self, src_input, tgt_input, labels=None, src_mask=None, tgt_mask=None):
        """
        参数:
            src_input: 源序列token索引，形状 (batch_size, src_seq_len)
            tgt_input: 目标序列token索引，形状 (batch_size, tgt_seq_len)
            labels: 目标序列的标签，形状 (batch_size, tgt_seq_len)，可选
            src_mask: 源序列的掩码（用于Encoder）
            tgt_mask: 目标序列的掩码（用于Decoder自注意力，避免看到未来信息）
        
        返回:
            如果提供labels: (loss, logits)
            如果不提供labels: (logits,)
        """
        # 通过编码器
        encoder_output = self.encoder(src_input, src_mask)
        
        # 通过解码器
        decoder_output = self.decoder(tgt_input, encoder_output, tgt_mask)
        
        # 输出层
        logits = self.output_layer(decoder_output)
        
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            # 修复：使用reshape而不是view
            logits_flat = logits.reshape(-1, self.config.vocab_size)
            labels_flat = labels.reshape(-1)
            loss = loss_fct(logits_flat, labels_flat)
            return loss, logits
        
        return (logits,)
    
    def generate(self, src_input, max_length=50, temperature=1.0):
        """
        序列到序列的生成函数
        """
        self.eval()
        batch_size = src_input.size(0)
        
        # 编码源序列
        encoder_output = self.encoder(src_input)
        
        # 初始化目标序列（开始符）
        # 这里我们使用空格作为开始符，实际中可以使用特殊的开始token
        start_token = self.encoder.embedding.weight.shape[0] - 1  # 使用最后一个token作为开始符
        tgt_input = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src_input.device)
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # 前向传播
                logits = self.forward(src_input, tgt_input)[0]
                
                # 取最后一个位置的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # 应用softmax得到概率
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # 从分布中采样
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到目标序列
                tgt_input = torch.cat([tgt_input, next_token], dim=1)
                
        return tgt_input

    def __repr__(self):
        return f"FullTransformer(vocab_size={self.config.vocab_size}, num_layers={self.config.num_layers})"
    
def create_attention_mask(seq_len, device):
    """
    创建因果注意力掩码（下三角矩阵）
    用于防止解码器看到未来的信息
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)

def create_padding_mask(seq, pad_token=0):
    """
    创建padding掩码
    """
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask