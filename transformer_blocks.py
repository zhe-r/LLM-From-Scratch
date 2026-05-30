import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # x 也就是我们的输入数据，它的维度通常是: (batch_size, seq_len, d_model)
        batch_size = q.size(0)

        # 1. 经过线性映射，生成 Q, K, V
        # 这里的 Q, K, V 维度依然是 (batch_size, seq_len, d_model)
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # 2. 拆分多头并交换维度
        # 原维度: (batch_size, seq_len, d_model)
        # view 之后: (batch_size, seq_len, n_heads, d_k)
        # transpose(1, 2) 之后: (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力分数 (Attention Scores)
        # Q 的维度: (batch_size, n_heads, seq_len, d_k)
        # K 转置后的维度: (batch_size, n_heads, d_k, seq_len)
        # scores 的维度: (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        # 4. 掩码 (Mask) 拦截逻辑 (兼容 Encoder 和 Decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. 计算 Softmax，得到真正的注意力权重 (Attention Weights)
        # dim=-1 表示在最后一个维度（即用来比对的那个序列长度维度）上求概率分布
        attn_weights = torch.softmax(scores, dim=-1)

        # 5. 将注意力权重与 V 相乘，得到加权后的结果
        # attn_weights 维度: (batch_size, n_heads, seq_len, seq_len)
        # V 维度: (batch_size, n_heads, seq_len, d_k)
        # output 维度: (batch_size, n_heads, seq_len, d_k)
        output = torch.matmul(attn_weights, V)

        # 6. 把 8 个头的工作成果拼接回来
        # 先 transpose(1, 2) 把维度变回: (batch_size, seq_len, n_heads, d_k)
        # 再用 contiguous() 在内存里“粘”牢固
        # 最后用 view 展平成: (batch_size, seq_len, d_model)
        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        # 7. 经过最后一个线性层，完美收官
        output = self.W_o(output)

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff = 2048):
        super(FeedForwardNetwork, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        # 1. 核心算子：nn.Embedding
        # 相当于一个 shape 为 (vocab_size, d_model) 的大矩阵
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x 是我们输入的词 ID 序列，维度通常是 (batch_size, seq_len)
        # 乘以 math.sqrt(d_model) 是论文原作者加的一个工程细节
        # 目的是把 Embedding 的数值稍微放大，防止后续加上位置编码时，词本身的语义被位置信息淹没
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEmbedding, self).__init__()
        # 1. 准备一个全是 0 的空白画布，准备往里填位置编码数据
        pe = torch.zeros(max_seq_len, d_model)

        # 2. 生成每个词的位置编号 pos: (0, 1, 2, ..., max_seq_len - 1)
        # unsqueeze(1) 是把它变成一列，维度变成 (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 3. 计算公式括号里的缩放项分母: 10000^(2i/d_model)
        # 这里用了指数和对数的数学变换来加速计算，属于 PyTorch 工程套路
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 4. 偶数维度 (0, 2, 4...) 用 sin 函数
        pe[:, 0::2] = torch.sin(position * div_term)

        # 5. 奇数维度 (1, 3, 5...) 用 cos 函数
        pe[:, 1::2] = torch.cos(position * div_term)

        # 6. 加一个 batch 的维度备用，变成 (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # 7. 终极技巧：register_buffer
        # 因为位置编码是固定的公式算出来的，不需要被模型训练（不需要梯度下降）
        # 把它注册成 buffer，PyTorch 就会自动保存它，但不会去更新它
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1), :]

        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048):
        super(EncoderBlock, self).__init__()
        # 1. 实例化核心组件
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        # 2. 定义层归一化 (Layer Normalization)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x, mask=None):
        # 第一部分：Multi-Head Attention + 残差连接 + LayerNorm
        # x + self.attention(x) 就是经典的残差操作
        attn_output = self.attention(q=x, k=x, v=x, mask=mask)
        x = self.norm1(x + attn_output)

        # 第二部分：FFN + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, look_ahead_mask=None,padding_mask=None):
        attn_output1 = self.masked_attention(q=x, k=x, v=x, mask=look_ahead_mask)
        x = self.norm1(x + attn_output1)

        attn_output2 = self.cross_attention(q=x, k=enc_output, v=enc_output, mask=padding_mask)
        x = self.norm2(x + attn_output2)

        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, num_layers,d_ff=2048,max_seq_len=5000):
        super(Transformer, self).__init__()

        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)

        self.pe = PositionalEmbedding(d_model, max_seq_len)

        self.encoders = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff) for _ in range(num_layers)]
        )

        self.decoders = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # ================== 左半边：编码器 (Encoder) 流程 ==================
        # 1. 输入通过嵌入层和位置编码
        enc_out = self.pe(self.src_embedding(src))

        # 2. 数据一层一层穿过 Nx 个 EncoderBlock
        for encoder in self.encoders:
            enc_out = encoder(enc_out, mask=src_mask)

        # ================== 右半边：解码器 (Decoder) 流程 ==================
        # 3. 目标输入(如当前已生成的英文)通过嵌入层和位置编码
        dec_out = self.pe(self.tgt_embedding(tgt))

        # 4. 数据一层一层穿过 Nx 个 DecoderBlock
        for decoder in self.decoders:
            # 注意：dec_out 不断迭代更新，但 enc_out 在每层都是固定的！
            dec_out = decoder(x=dec_out,
                              enc_output=enc_out,
                              look_ahead_mask=tgt_mask,
                              padding_mask=src_mask)

        # 5. 经过最后的线性层，输出对词表中每个词的打分
        output = self.fc_out(dec_out)

        return output


# ===================================================================
# Mask 生成小助手
# ===================================================================
def create_padding_mask(seq, pad_idx=0):
    # seq 维度: (batch_size, seq_len)
    # 我们希望非 0 的地方是 1 (可以看)，等于 0 的地方是 0 (遮挡)
    # 增加维度变成 (batch_size, 1, 1, seq_len)，为了和 scores 矩阵 (batch, heads, seq, seq) 广播对齐
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(seq_len):
    # 生成一个下三角矩阵，1 表示可以看，0 表示不能看 (遮挡未来)
    # 维度: (seq_len, seq_len)
    def create_look_ahead_mask(seq_len):
        mask = torch.tril(
            torch.ones(seq_len, seq_len)
        ).bool()

        return mask.unsqueeze(0).unsqueeze(0)


