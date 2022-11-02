import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

EMBEDDING_SIZE = 512  # 每个token Embedding的维度
FC_SIZE = 2048  # 前馈神经网络中Linear层映射到多少维度
ENCODER_LAYERS = 6  # 6个encoder
N_HEADS = 8  # multi_head num
D_k = D_v = 64  # dimension of K(=Q), V

# ----------------------------------#
# ScaledDotProductAttention的实现
# ----------------------------------#
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # 输入进来的维度分别是 [batch_size x N_HEADS x len_q x D_k]  K： [batch_size x N_HEADS x len_k x D_k]
        # V: [batch_size x N_HEADS x len_k x D_v]
        # 首先经过matmul函数得到的scores形状是: [batch_size x N_HEADS x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(D_k)

        # 然后最关键的地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对其他单词就不会起作用
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# -------------------------#
# MultiHeadAttention的实现
# -------------------------#
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 输入进来的QKV是相等的，我们会使用linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(EMBEDDING_SIZE, D_k * N_HEADS)
        self.W_K = nn.Linear(EMBEDDING_SIZE, D_k * N_HEADS)
        self.W_V = nn.Linear(EMBEDDING_SIZE, D_v * N_HEADS)
        self.linear = nn.Linear(N_HEADS * D_v, EMBEDDING_SIZE)
        self.layer_norm = nn.LayerNorm(EMBEDDING_SIZE)

    def forward(self, Q, K, V, attn_mask):

        # 这个多头注意力机制分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        # 输入进来的数据形状： Q: [batch_size x len_q x EMBEDDING_SIZE], K: [batch_size x len_k x EMBEDDING_SIZE],
        # V: [batch_size x len_k x EMBEDDING_SIZE]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # 下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        # q_s: [batch_size x N_HEADS x len_q x D_k]
        q_s = self.W_Q(Q).view(batch_size, -1, N_HEADS, D_k).transpose(1, 2)
        # k_s: [batch_size x N_HEADS x len_k x D_k]
        k_s = self.W_K(K).view(batch_size, -1, N_HEADS, D_k).transpose(1, 2)
        # v_s: [batch_size x N_HEADS x len_k x D_v]
        v_s = self.W_V(V).view(batch_size, -1, N_HEADS, D_v).transpose(1, 2)

        # 输入进来的attn_mask形状是batch_size x len_q x len_k，然后经过下面这个代码得到
        # 新的attn_mask: [batch_size x N_HEADS x len_q x len_k]，就是把pad信息重复到了n个头上
        attn_mask = attn_mask.unsqueeze(1).repeat(1, N_HEADS, 1, 1)

        # 然后我们运行ScaledDotProductAttention这个函数
        # 得到的结果有两个：context: [batch_size x N_HEADS x len_q x D_v], attn: [batch_size x N_HEADS x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size x len_q x N_HEADS * D_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, N_HEADS * D_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x EMBEDDING_SIZE]


# ----------------------------#
# PoswiseFeedForwardNet的实现
# ----------------------------#
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=EMBEDDING_SIZE, out_channels=FC_SIZE, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=FC_SIZE, out_channels=EMBEDDING_SIZE, kernel_size=1)
        self.layer_norm = nn.LayerNorm(EMBEDDING_SIZE)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, EMBEDDING_SIZE]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


# --------------------------#
# get_attn_pad_mask的实现：
# --------------------------#
# 比如说，我现在的句子长度是5，在后面注意力机制的部分，我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状len_input * len*input
# 代表每个单词对其余包含自己的单词的影响力。所以这里我需要有一个同等大小形状的矩阵，告诉我哪个位置是PAD部分，之后在计算softmax之前会把这里置
# 为无穷大；一定需要注意的是这里得到的矩阵形状是batch_size x len_q x len_k，我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没必要。
# seq_q和seq_k不一定一致，在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边的pad符号信息就可以，解码端的pad信息在交互注意力层是
# 没有用到的；
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k





# ------------------------------#
# Positional Encoding的代码实现
# ------------------------------#
class PositionalEncoding(nn.Module):
    def __init__(self, EMBEDDING_SIZE, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面的代码只是其中的一种实现方式；
        # 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        # pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        # 假设我的EMBEDDING_SIZE是512，2i以步长2从0取到了512，那么i对应取值就是0,1,2...255
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, EMBEDDING_SIZE)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, EMBEDDING_SIZE, 2).float() * (-math.log(10000.0) / EMBEDDING_SIZE))
        pe[:, 0::2] = torch.sin(position * div_term)  # 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，步长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，步长为2，其实代表的就是奇数位置

        # 下面这行代码之后，我们得到的pe形状是：[max_len * 1 * EMBEDDING_SIZE]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  # 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, EMBEDDING_SIZE]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



# ---------------------------------------------------#
# EncoderLayer：包含两个部分，多头注意力机制和前馈神经网络
# ---------------------------------------------------#
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x EMBEDDING_SIZE]，需要注意的是最初始的QKV矩阵是等同于这个
        输入的，去看一下enc_self_attn函数.
        """

        # enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs: [batch_size x len_q x EMBEDDING_SIZE]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


# -----------------------------------------------------------------------------#
# Encoder部分包含三个部分：词向量embedding，位置编码部分，自注意力层及后续的前馈神经网络
# -----------------------------------------------------------------------------#
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 位置编码，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        self.pos_emb = PositionalEncoding(EMBEDDING_SIZE)
        # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(ENCODER_LAYERS)])

    def forward(self, inputs, inputs_mask):


        # 这行是位置编码，把两者相加放到了pos_emb函数里面
        outputs = self.pos_emb(inputs.transpose(0, 1)).transpose(0, 1)

        enc_self_attn_mask = get_attn_pad_mask(inputs_mask, inputs_mask)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()  # 编码层
        self.linear = nn.Linear(EMBEDDING_SIZE, 1)

    def forward(self, inputs, inputs_mask):
        """
        这里有两个数据进行输入，一个是enc_inputs，形状为[batch_size, src_len]，主要是作为编码端的输入，一个是dec_inputs，
        形状为[batch_size, tgt_len]，主要是作为解码端的输入.
        enc_inputs作为输入，形状为[batch_size, src_len]，输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，
        可以是特定每一层的输出，也可以是中间某些参数的输出；
        """

        # enc_outputs就是编码端的输出，enc_self_attns这里没记错的话是QK转置相乘经softmax之后的矩阵值，代表
        # 的是每个单词和其他单词的相关性，即相关性矩阵；
        outputs, outputs_attns = self.encoder(inputs, inputs_mask)

        # dec_outputs是decoder的主要输出，用于后续的linear映射； dec_self_attns类比于enc_self_attns，
        # 是查看每个单词对decoder中输入的其余单词的相关性；dec_enc_attns是decoder中每个单词对encoder中每
        # 个单词的相关性；
        #dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_outputs做映射到词表大小
        # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        #dec_logits = self.projection(dec_outputs)
        #return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
      
        outputs = self.linear(outputs[:, 0, :])

        return outputs, outputs_attns


if __name__ == '__main__':
    inputs_x = torch.randn(10, 20, EMBEDDING_SIZE)  #batch_size * seq_len * EMBEDDING_SIZE
    inputs_y = torch.randn(10, 1) #batch_size * seq_len * 1
    inputs_mask_x = torch.ones(10, 20)


    model = Transformer()
    criterion = nn.MSELoss()# 定义损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)# 定义优化器
    
    for epoch in range(20):
        optimizer.zero_grad()
        outputs, outputs_attns = model(inputs_x, inputs_mask_x)
        loss = criterion(outputs, inputs_y)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()


