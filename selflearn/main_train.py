import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from datasets import load_from_disk

from transformers import BertTokenizer

from transformer_blocks import (
    Transformer,
    create_padding_mask,
    create_look_ahead_mask
)

# =====================================
# 超参数
# =====================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 16

D_MODEL = 256
N_HEADS = 8
NUM_LAYERS = 4
D_FF = 1024

MAX_LEN = 128

EPOCHS = 10

LR = 1e-4

MODEL_SAVE_PATH = "transformer_en_zh.pth"

# =====================================
# Tokenizer
# =====================================

src_tokenizer = BertTokenizer.from_pretrained(
    "bert-base-chinese"
)

tgt_tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

SRC_VOCAB_SIZE = src_tokenizer.vocab_size
TGT_VOCAB_SIZE = tgt_tokenizer.vocab_size

# =====================================
# Dataset
# =====================================

dataset = load_from_disk(
    "./opus100_en_zh_local"
)

train_split = dataset["train"].select(
    range(10000)
)


class TranslationDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]["translation"]

        return item["zh"], item["en"]


def collate_fn(batch):

    zh_list = []
    en_list = []

    for zh, en in batch:
        zh_list.append(zh)
        en_list.append(en)

    src = src_tokenizer(
        zh_list,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    tgt = tgt_tokenizer(
        en_list,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    return (
        src["input_ids"],
        tgt["input_ids"]
    )


train_dataset = TranslationDataset(
    train_split
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# =====================================
# 模型
# =====================================

model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF
).to(DEVICE)

# =====================================
# Loss
# =====================================

criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_tokenizer.pad_token_id
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

# =====================================
# 开始训练
# =====================================

print("========== 开始训练 ==========")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_idx, (src_ids, tgt_ids) in enumerate(train_loader):

        src_ids = src_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)

        # Decoder输入
        decoder_input = tgt_ids[:, :-1]

        # Decoder标签
        decoder_target = tgt_ids[:, 1:]

        src_mask = create_padding_mask(
            src_ids
        ).to(DEVICE)

        tgt_mask = create_look_ahead_mask(
            decoder_input.size(1)
        ).to(DEVICE)

        optimizer.zero_grad()

        output = model(
            src_ids,
            decoder_input,
            src_mask,
            tgt_mask
        )

        vocab_size = output.size(-1)

        loss = criterion(
            output.reshape(-1, vocab_size),
            decoder_target.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Batch [{batch_idx}] "
                f"Loss={loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader)

    print(
        f"\nEpoch {epoch+1} "
        f"Average Loss={avg_loss:.4f}\n"
    )

    torch.save(
        model.state_dict(),
        MODEL_SAVE_PATH
    )

print("训练结束")
