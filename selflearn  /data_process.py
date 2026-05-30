import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer

# 【必杀技】走国内镜像，防止下载超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ===================================================================
# 1. 离线数据集与词典准备
# ===================================================================
local_data_dir = "./opus100_en_zh_local"
if not os.path.exists(local_data_dir):
    print("⏳ 正在下载数据集...")
    dataset = load_dataset("opus100", "en-zh")
    dataset.save_to_disk(local_data_dir)
    print("💾 数据集已成功永久保存到本地！")
else:
    print("⚡ 发现本地缓存，秒速加载数据集！")
    dataset = load_from_disk(local_data_dir)

# 加载你本地的 Tokenizer 文件夹
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


# ===================================================================
# 2. 编写大厂规范的 PyTorch Dataset 类
# ===================================================================
class TranslationDataset(Dataset):
    def __init__(self, hf_dataset_split):
        # 传入数据集的某一侧，比如 dataset['train']
        self.data = hf_dataset_split

    def __len__(self):
        # 告诉 PyTorch 数据集一共有多少条
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引 idx，取出一条中文和一条英文文本
        item = self.data[idx]['translation']
        return item['zh'], item['en']


# ===================================================================
# 3. 终极面试考点：collate_fn (动态打包与 Padding 小助手)
# ===================================================================
def collate_fn(batch):
    """
    batch 进来的是一个列表，里面包含了当前 Batch 大小的 (zh_text, en_text) 元组。
    这个函数负责：把文本转成 ID、找出最长的一句话、用 pad_id(0) 把短句子补齐！
    """
    zh_list, en_list = [], []
    for zh, en in batch:
        zh_list.append(zh)
        en_list.append(en)

    # tokenizer() 会非常智能地把这一批文本全部转成 ID
    # padding=True 会自动找出这批句子里最长的那句，把短句用 0 补齐
    # return_tensors="pt" 表示直接返回 PyTorch 的 Tensor 张量
    src_features = tokenizer(zh_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
    tgt_features = tokenizer(en_list, padding=True, truncation=True, max_length=128, return_tensors="pt")

    # 提取出补齐后的 ID 矩阵
    src_ids = src_features['input_ids']
    tgt_ids = tgt_features['input_ids']

    return src_ids, tgt_ids


# ===================================================================
# 4. 流水线组装与数据流验证
# ===================================================================
if __name__ == "__main__":
    print("\n========== 开始组装 PyTorch 数据流水线 ==========")
    # 实例化我们的 Dataset (我们先拿前 100 条小试牛刀，防止本地内存压力大)
    # 你可以随时换成整个 dataset['train']
    mini_train_data = dataset['train'].select(range(100))
    train_dataset = TranslationDataset(mini_train_data)

    # 实例化 DataLoader
    # batch_size=4 代表一次打包 4 句话；shuffle=True 代表随机打乱顺序
    # collate_fn=collate_fn 是灵魂，它接管了如何把 4 句文本拼成一个规范矩阵的底层操作
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # 倒出一批数据，看看从生产线上流下来的究竟长什么样！
    for src_ids, tgt_ids in train_loader:
        print("\n🚀 成功从 DataLoader 中倒出一批真实数据！")
        print(f"源语言 (中文 src_ids) 的 Tensor 形状: {src_ids.shape}")
        print("中文 ID 矩阵内容 (注意看右边的 0 填充):")
        print(src_ids)

        print(f"\n目标语言 (英文 tgt_ids) 的 Tensor 形状: {tgt_ids.shape}")
        print("英文 ID 矩阵内容:")
        print(tgt_ids)

        print("\n==================================================")
        break  # 我们只看第一批，看懂了就立马收工
