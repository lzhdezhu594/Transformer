#!/bin/bash

# 设置随机种子
SEED=42

echo "设置随机种子: $SEED"
export PYTHONHASHSEED=$SEED

# 创建结果目录
mkdir -p results

echo "开始训练Transformer语言模型..."
cd src

# 训练模型
python ../train.py

echo "训练完成！结果保存在 results/ 目录中"