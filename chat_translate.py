import torch

from transformers import BertTokenizer

from transformer_blocks import (
    Transformer,
    create_padding_mask,
    create_look_ahead_mask
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 中文tokenizer
src_tokenizer = BertTokenizer.from_pretrained(
    "bert-base-chinese"
)

# 英文tokenizer
tgt_tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = Transformer(
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size,
    d_model=256,
    n_heads=8,
    num_layers=4,
    d_ff=1024
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "transformer_en_zh.pth",
        map_location=DEVICE
    )
)

model.eval()

print("模型加载完成")

def translate(text, max_len=50):

    src = src_tokenizer(
        text,
        return_tensors="pt"
    )["input_ids"].to(DEVICE)

    src_mask = create_padding_mask(
        src
    ).to(DEVICE)

    generated = [
        tgt_tokenizer.cls_token_id
    ]

    for _ in range(max_len):

        tgt = torch.tensor(
            [generated],
            dtype=torch.long
        ).to(DEVICE)

        tgt_mask = create_look_ahead_mask(
            tgt.size(1)
        ).to(DEVICE)

        with torch.no_grad():

            output = model(
                src,
                tgt,
                src_mask,
                tgt_mask
            )

        next_token = output[:, -1, :]

        next_id = torch.argmax(
            next_token,
            dim=-1
        ).item()

        generated.append(next_id)

        if next_id == tgt_tokenizer.sep_token_id:
            break

    result = tgt_tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return result

while True:

    text = input("\n中文> ")

    if text.lower() == "q":
        break

    result = translate(text)

    print("英文>", result)