# Transformer 实现作业

## 项目描述
手工实现Transformer模型并在小规模数据集上进行训练验证。

## 环境要求
- Python 3.8+
- PyTorch 1.9+
- 其他依赖见 requirements.txt

## 项目结构
transformer/
├── config.py # 模型配置参数
├── model.py # Transformer模型实现
├── data_loader.py # 数据加载和预处理
├── train_only_encoder.py # 仅训练编码器（语言模型）
├── train_seq2seq.py # 完整编码器-解码器训练
├── test_components.py # 组件测试脚本
├── ablation_study.py # 消融实验
├── generate.py # 文本生成（编码器版本）
├── generate_seq2seq.py # 文本生成（序列到序列版本）
├── utils.py # 工具函数
├── requirements.txt # 依赖包
└── results/ # 训练结果和模型保存
├── best_model.pth
├── full_transformer_best.pth
└── ablation/ # 消融实验结果

## 运行方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据准备
项目使用Tiny Shakespeare数据集，运行任何训练脚本会自动下载数据。

### 3. 模型训练
仅训练编码器（语言模型）
```bash
python train_only_encoder.py
```
用途：训练基于Transformer编码器的语言模型
输出：results/best_model.pth

训练完整序列到序列模型
```bash
python train_seq2seq.py
```
用途：训练完整的Encoder-Decoder Transformer
输出：results/full_transformer_best.pth

### 4. 模型测试
使用编码器模型生成文本
```bash
python generate.py #--prompt "Your text here" --max_length 200
```
使用序列到序列模型生成文本
```bash
python generate_seq2seq.py #--prompt "Your text here" --max_length 200
```

### 5. 消融实验
```bash
python ablation_study.py
```
运行完整的消融实验，评估各组件重要性
生成可视化图表和分析报告
结果保存在 results/ablation/ 目录
