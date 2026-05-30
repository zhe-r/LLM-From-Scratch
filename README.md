# 🚀 LLM-From-Scratch: 手撕 Transformer 核心架构

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)

## 📌 项目简介 (Project Overview)
本项目旨在从零开始（From Scratch）深入理解大语言模型（LLM）的底层基石。不依赖 `nn.Transformer` 等高级封装 API，完全基于 PyTorch 基础张量操作与算子，逐行实现了原汁原味的 Transformer 架构。

本项目是我个人【大模型算法工程师实习备战计划 - 第一阶段】的核心产出。

## 🎯 核心里程碑 (Key Milestones Achieved)
- [x] **纯手写多头注意力机制 (MHA)**：深刻理解 $Q, K, V$ 张量切分、维度置换（Transpose/Contiguous）与注意力分数计算。
- [x] **因果掩码机制 (Causal/Look-ahead Masking)**：通过下三角矩阵实现 Decoder 的防作弊机制，打通了 GPT 等生成式模型的核心原理。
- [x] **完整 Encoder-Decoder 结构**：实现带残差连接（Residual Connection）与层归一化（LayerNorm）的堆叠 Block，以及交叉注意力（Cross-Attention）的数据流转。
- [x] **端到端训练与推理闭环**：接入 HuggingFace 真实翻译数据集，自定义 `collate_fn` 实现动态 Padding，并实现了基于贪心搜索（Greedy Search）的自回归生成推理。

## 📂 核心代码结构 (Repository Structure)
- `transformer_blocks.py`: 核心网络结构，包含 `TokenEmbedding`, `PositionalEncoding`, `MultiHeadAttention`, `EncoderBlock`, `DecoderBlock` 及最终的 `Transformer` 宏观类。
- `data_process.py`: 工业级数据流水线，涵盖 HuggingFace 数据集离线下载、Tokenizer 词典加载以及 PyTorch DataLoader 的动态组装。
- `main_train.py`: 模型的端到端微型训练脚本（Teacher Forcing 机制）。
- `inference.py`: 模型的自回归推理脚本，验证模型文本生成能力。

## 🛠️ 快速开始 (Quick Start)
1. 安装依赖：
```bash
pip install torch transformers datasets
```

2. 运行数据处理与验证：
```bash
python data_process.py
```

3. 启动模型训练：
```bash
python main_train.py
```

## 🗺️ 后续计划 (Roadmap)
- 🚀 **Q2计划**: 迁移至云端 4090D 算力，探索 Llama/Qwen 等开源大模型的 SFT (指令微调) 及 RAG (检索增强生成) 知识库系统搭建。
