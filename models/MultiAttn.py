import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SoftHGRLoss import SoftHGRLoss

'''
Bidirectional cross-attention layers.
'''
class BidirectionalCrossAttention(nn.Module):

    def __init__(self, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.query_matrix = nn.Linear(model_dim, Q_dim)
        self.key_matrix = nn.Linear(model_dim, K_dim)
        self.value_matrix = nn.Linear(model_dim, V_dim)


    def bidirectional_scaled_dot_product_attention(self, Q, K, V):
        score = torch.bmm(Q, K.transpose(-1, -2))
        scaled_score = score / (K.shape[-1]**0.5)
        attention = torch.bmm(F.softmax(scaled_score, dim = -1), V)

        return attention


    def forward(self, query, key, value):
        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)
        attention = self.bidirectional_scaled_dot_product_attention(Q, K, V)

        return attention



'''
Multi-head bidirectional cross-attention layers.
'''
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [BidirectionalCrossAttention(model_dim, Q_dim, K_dim, V_dim) for _ in range(self.num_heads)]
        )
        self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)


    def forward(self, query, key, value):
        heads = [self.attention_heads[i](query, key, value) for i in range(self.num_heads)]
        multihead_attention = self.projection_matrix(torch.cat(heads, dim = -1))

        return multihead_attention



'''
A feed-forward network, which operates as a key-value memory.
'''
class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    

    def forward(self, x):
        return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))



'''
Residual connection to smooth the learning process.
'''
class AddNorm(nn.Module):

    def __init__(self, model_dim, dropout_rate):
        super().__init__()

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, x, sublayer):
        output = self.layer_norm(x + self.dropout(sublayer(x)))

        return output


'''
MultiAttn is a multimodal fusion model which aims to capture the complicated interactions and 
dependencies across textual, audio and visual modalities through bidirectional cross-attention layers.
MultiAttn is made up of three sub-components:
1. MultiAttn_text: integrate the textual modality with audio and visual information;
2. MultiAttn_audio: incorporate the audio modality with textual and visual information;
3. MultiAttn_visual: fuse the visual modality with textual and visual cues.
'''
class MultiAttnLayer(nn.Module):

    def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        Q_dim = K_dim = V_dim = model_dim // num_heads
        self.attn_1 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
        self.add_norm_1 = AddNorm(model_dim, dropout_rate)
        self.attn_2 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
        self.add_norm_2 = AddNorm(model_dim, dropout_rate)
        self.ff = Feedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_3 = AddNorm(model_dim, dropout_rate)

    def forward(self, query_modality, modality_A, modality_B):
        attn_output_1 = self.add_norm_1(query_modality, lambda query_modality: self.attn_1(query_modality, modality_A, modality_A))
        attn_output_2 = self.add_norm_2(attn_output_1, lambda attn_output_1: self.attn_2(attn_output_1, modality_B, modality_B))
        ff_output = self.add_norm_3(attn_output_2, self.ff)
        return ff_output

    # def forward(self, query_modality, modality_A):
    #     attn_output_1 = self.add_norm_1(query_modality, lambda query_modality: self.attn_1(query_modality, modality_A, modality_A))
    #     # attn_output_2 = self.add_norm_2(attn_output_1, lambda attn_output_1: self.attn_2(attn_output_1, modality_B, modality_B))
    #     ff_output = self.add_norm_3(attn_output_1, self.ff)
    #     return ff_output
'''
Stacks of MultiAttn layers.
'''
class MultiAttn(nn.Module):

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        self.multiattn_layers = nn.ModuleList([
            MultiAttnLayer(num_heads, model_dim, hidden_dim, dropout_rate) for _ in range(num_layers)])

    def forward(self, query_modality, modality_A, modality_B):
        for multiattn_layer in self.multiattn_layers:
            query_modality = multiattn_layer(query_modality, modality_A, modality_B)
    # def forward(self, query_modality, modality_A):
    #     for multiattn_layer in self.multiattn_layers:
    #         query_modality = multiattn_layer(query_modality, modality_A)

        return query_modality


def same_init(layer1, layer2, layer3):
    weight = torch.empty_like(layer1.weight)
    nn.init.xavier_uniform_(weight)  # 使用 Xavier 均匀分布初始化
    # nn.init.kaiming_uniform_(weight, a= math.sqrt(5))  # 使用 PyTorch 的初始化逻辑
    layer1.weight.data = weight.clone()
    layer2.weight.data = weight.clone()
    layer3.weight.data = weight.clone()
    if layer1.bias is not None:
        nn.init.constant_(layer1.bias, 0)
        layer2.bias.data = layer1.bias.data.clone()
        layer3.bias.data = layer1.bias.data.clone()

class MultiAttnModel(nn.Module):

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, out_feature, dropout_rate):
        super().__init__()

        self.multiattn_text = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_audio = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_visual = MultiAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.fc = nn.Linear(model_dim, out_feature)
        self.HGR = SoftHGRLoss()

        self.weight_t = nn.Linear(model_dim, 1)
        self.weight_a = nn.Linear(model_dim, 1)
        self.weight_v = nn.Linear(model_dim, 1)
        same_init(self.weight_t, self.weight_a, self.weight_v)
        # self.weight = nn.Linear(model_dim, 1)


    def forward(self, text_features, audio_features, visual_features):
        f_t = self.multiattn_text(text_features, audio_features, visual_features)
        f_a = self.multiattn_visual(audio_features, text_features, visual_features)
        f_v = self.multiattn_visual(visual_features, text_features, audio_features)

        # loss = self.HGR(f_t, f_a, f_v)
        loss = 0

        w_t = self.weight_t(f_t)  # (batch_size, max_sen_len, 1)
        w_a = self.weight_a(f_a)  # (batch_size, max_sen_len, 1)
        w_v = self.weight_v(f_v)  # (batch_size, max_sen_len, 1)

        # 将权重变成 0 到 1 之间，并确保它们的和为1
        weights = torch.cat([w_t, w_a, w_v], dim=-1)  # (batch_size, max_sen_len, 3)
        weights = F.softmax(weights, dim=-1)  # (batch_size, max_sen_len, 3)
        # weights = (weights + 1.0 / 3) / 2  # 将权重平滑到更接近均匀分布

        # 计算均方误差 (MSE) 正则化损失，使权重分配更均匀
        # reg_loss = self.compute_mse_regularization_loss(weights)
        # print(weights)

        # mean_values = torch.mean(weights, dim=1)
        # print(mean_values)

        # 将权重分割回三个模态
        w_t, w_a, w_v= torch.split(weights, 1, dim=-1)  # 分割为三个权重张量
        # similarity(f_t, f_a, f_v)
        # 计算每个模态的权重

        # similarity(f_t, f_a, f_v)
        # 对每个模态的特征进行加权
        # weighted_sum = w_t * f_t + w_a * f_a + w_v * f_v  # (batch_size, max_sen_len, dim)
        weighted_sum = w_t * f_t + w_a * f_a + w_v * f_v
        # visualize_weights(w_t, w_a, w_v, 1)
        # return self.fc(weighted_sum * 3)
        return self.fc(weighted_sum * 3), loss

    def compute_mse_regularization_loss(self, weights):
        """
        计算均方误差正则化损失，使得权重更加均匀
        :param weights: (batch_size, max_sen_len, 3)
        :return: 正则化损失
        """
        # 计算每对模态权重之间的差异，并平方求和
        mse_loss = torch.mean((weights[:, :, 0] - weights[:, :, 1]) ** 2) + \
                   torch.mean((weights[:, :, 0] - weights[:, :, 2]) ** 2) + \
                   torch.mean((weights[:, :, 1] - weights[:, :, 2]) ** 2)
        return mse_loss




# class MultiAttnModel_1(nn.Module):
#     def __init__(self, model_dim, num_heads, dropout_rate):
#         super().__init__()
#
#         # 为每个模态定义自己的多头注意力层
#         self.multiattn_text = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout_rate,
#                                                     batch_first=True)
#         self.multiattn_audio = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout_rate,
#                                                      batch_first=True)
#         self.multiattn_visual = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout_rate,
#                                                       batch_first=True)
#
#         # 最后的全连接层
#         self.fc = nn.Linear(model_dim, 768)
#
#     def forward(self, text_features, audio_features, visual_features):
#         # text_features, audio_features, visual_features 的 shape 为 (batch_size, sequence_length, model_dim)
#
#         # 通过自注意力机制计算各模态特征
#         attn_output_t, attn_weights_t = self.multiattn_text(text_features, text_features, text_features)
#         attn_output_a, attn_weights_a = self.multiattn_audio(audio_features, audio_features, audio_features)
#         attn_output_v, attn_weights_v = self.multiattn_visual(visual_features, visual_features, visual_features)
#
#         # 将注意力权重归一化成每个模态的权重
#         weights = torch.cat([attn_weights_t.mean(dim=1, keepdim=True),
#                              attn_weights_a.mean(dim=1, keepdim=True),
#                              attn_weights_v.mean(dim=1, keepdim=True)], dim=1)  # (batch_size, 3, sequence_length)
#
#         weights = F.softmax(weights, dim=1)  # 在模态维度上做 softmax 归一化
#
#         # 提取归一化后的权重
#         w_t = weights[:, 0, :].unsqueeze(-1)  # (batch_size, sequence_length, 1)
#         w_a = weights[:, 1, :].unsqueeze(-1)  # (batch_size, sequence_length, 1)
#         w_v = weights[:, 2, :].unsqueeze(-1)  # (batch_size, sequence_length, 1)
#
#         # 对每个模态的特征进行加权平均
#         weighted_sum = w_t * attn_output_t + w_a * attn_output_a + w_v * attn_output_v  # (batch_size, sequence_length, model_dim)
#
#         # 将加权后的特征通过全连接层
#         output = self.fc(weighted_sum)  # (batch_size, sequence_length, 768)
#
#         return output

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_matrix(matrixs):
    modality_labels = ['text', 'audio', 'vision']
    # 设置画布
    plt.figure(figsize=(5, 5))
    # for i in range(matrixs.shape[0]):
    matrix = matrixs[0].cpu().detach().numpy()
    # 绘制第一个矩阵的热力图
    plt.subplot(2, 2, 1)
    sns.heatmap(matrix, cmap="Reds", annot=False, cbar=True, xticklabels=modality_labels)
    plt.title('emo 1')
    plt.xlabel('modality')
    plt.ylabel('sentence')

    matrix = matrixs[1].cpu().detach().numpy()
    plt.subplot(2, 2, 2)
    sns.heatmap(matrix, cmap="Reds", annot=False, cbar=True, xticklabels=modality_labels)
    plt.title('cause 1')
    plt.xticks(range(3), modality_labels)
    # plt.xlabel('Visual Features of Each Frame')
    # plt.ylabel('Language Features of Each Frame')

    matrix = matrixs[2].cpu().detach().numpy()
    plt.subplot(2, 2, 3)
    sns.heatmap(matrix, cmap="Reds", annot=False, cbar=True, xticklabels=modality_labels)
    plt.title('cause 2')
    # plt.xlabel('Visual Features of Each Frame')
    # plt.ylabel('Language Features of Each Frame')

    matrix = matrixs[3].cpu().detach().numpy()
    plt.subplot(2, 2, 4)
    sns.heatmap(matrix, cmap="Reds", annot=False, cbar=True, xticklabels=modality_labels)
    plt.title('cause 3')
    plt.tight_layout()
    plt.show()


def similarity(f_t, f_a, f_v):
    bs = f_t.shape[0]
    max_sen = f_t.shape[1]
    hz = f_t.shape[2]

    # 计算文本-音频、文本-视频、音频-视频的余弦相似性矩阵
    # 计算时需要将特征展平成二维 (bs*max_sen, hz)，再计算相似性
    f_t_flat = f_t.view(bs * max_sen, hz)
    f_a_flat = f_a.view(bs * max_sen, hz)
    f_v_flat = f_v.view(bs * max_sen, hz)

    # 文本-音频余弦相似性
    cos_sim_ta = F.cosine_similarity(f_t_flat.unsqueeze(1), f_a_flat.unsqueeze(0), dim=-1)
    # 文本-视频余弦相似性
    cos_sim_tv = F.cosine_similarity(f_t_flat.unsqueeze(1), f_v_flat.unsqueeze(0), dim=-1)
    # 音频-视频余弦相似性
    cos_sim_av = F.cosine_similarity(f_a_flat.unsqueeze(1), f_v_flat.unsqueeze(0), dim=-1)

    # 输出相似性矩阵的形状
    print("Cosine Similarity (Text-Audio):", cos_sim_ta)
    print("Cosine Similarity (Text-Video):", cos_sim_tv)
    print("Cosine Similarity (Audio-Video):", cos_sim_av)

# tensor([[[0.3330, 0.3336, 0.3335],
#          [0.4818, 0.2536, 0.2646],
#          [0.4467, 0.2686, 0.2847],
#          [0.4101, 0.2996, 0.2902],
#          [0.4029, 0.2954, 0.3017],
#          [0.3588, 0.3136, 0.3276],
#          [0.5105, 0.2497, 0.2397],
#          [0.3332, 0.3490, 0.3178],
#          [0.3754, 0.3361, 0.2886]]], device='cuda:1')
# tensor([[[0.2519, 0.3741, 0.3740],
#          [0.3879, 0.2996, 0.3125],
#          [0.3649, 0.3083, 0.3268],
#          [0.3284, 0.3412, 0.3305],
#          [0.3903, 0.3017, 0.3081],
#          [0.4102, 0.2885, 0.3014],
#          [0.5715, 0.2186, 0.2099],
#          [0.2835, 0.3751, 0.3415],
#          [0.2570, 0.3997, 0.3432]]], device='cuda:1')
# tensor([[[0.2589, 0.3706, 0.3705],
#          [0.4131, 0.2873, 0.2997],
#          [0.3939, 0.2942, 0.3118],
#          [0.3330, 0.3388, 0.3282],
#          [0.3655, 0.3139, 0.3206],
#          [0.2963, 0.3441, 0.3595],
#          [0.5018, 0.2542, 0.2440],
#          [0.3935, 0.3174, 0.2890],
#          [0.2727, 0.3913, 0.3360]]], device='cuda:1')
# tensor([[[0.4411, 0.2736, 0.2854],
#          [0.3764, 0.3027, 0.3208],
#          [0.3272, 0.3417, 0.3310],
#          [0.3555, 0.3189, 0.3257],
#          [0.2880, 0.3482, 0.3638],
#          [0.4455, 0.2829, 0.2716],
#          [0.3582, 0.3360, 0.3059],
#          [0.5307, 0.2525, 0.2168]]], device='cuda:1')


# draw_matrix(tensor)