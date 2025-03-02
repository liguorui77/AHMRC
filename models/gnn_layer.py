import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F



# class GraphAttentionLayer(nn.Module):
#     """
#     reference: https://github.com/xptree/DeepInf
#     """
#     def __init__(self, att_head, in_dim, out_dim, dp_gnn, device, leaky_alpha=0.2):
#         super(GraphAttentionLayer, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.dp_gnn = dp_gnn
#
#         self.att_head = att_head
#         self.W = nn.Parameter(torch.Tensor(self.att_head, self.in_dim, self.out_dim))
#         self.b = nn.Parameter(torch.Tensor(self.out_dim))
#
#         self.w_src = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
#         self.w_dst = nn.Parameter(torch.Tensor(self.att_head, self.out_dim, 1))
#         self.leaky_alpha = leaky_alpha
#         self.init_gnn_param()
#
#         assert self.in_dim == self.out_dim*self.att_head
#         self.H = nn.Linear(self.in_dim, self.in_dim)
#         init.xavier_normal_(self.H.weight)
#         self.device = device
#
#     def init_gnn_param(self):
#         init.xavier_uniform_(self.W.data)
#         init.zeros_(self.b.data)
#         init.xavier_uniform_(self.w_src.data)
#         init.xavier_uniform_(self.w_dst.data)
#
#     def forward(self, feat_in, adj=None):
#         batch, N, in_dim = feat_in.size()
#         assert in_dim == self.in_dim
#
#         feat_in_ = feat_in.unsqueeze(1)
#         h = torch.matmul(feat_in_, self.W)
#
#         attn_src = torch.matmul(torch.tanh(h), self.w_src)
#         attn_dst = torch.matmul(torch.tanh(h), self.w_dst)
#         attn = attn_src.expand(-1, -1, -1, N) + attn_dst.expand(-1, -1, -1, N).permute(0, 1, 3, 2)
#         attn = F.leaky_relu(attn, self.leaky_alpha, inplace=True)
#
#         adj = torch.FloatTensor(adj).to(self.device)
#         mask = 1 - adj.unsqueeze(1)
#         attn.data.masked_fill_(mask.bool(), -999)
#
#         attn = F.softmax(attn, dim=-1)
#         feat_out = torch.matmul(attn, h) + self.b
#
#         feat_out = feat_out.transpose(1, 2).contiguous().view(batch, N, -1)
#         feat_out = F.elu(feat_out)
#
#         gate = torch.sigmoid(self.H(feat_in))
#         feat_out = gate * feat_out + (1 - gate) * feat_in
#
#         feat_out = F.dropout(feat_out, self.dp_gnn, training=self.training)
#
#         return feat_out
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim*self.att_head) + ')'
#
def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

def exponential_decay(x):
    """
    指数衰减函数：e^(-x)
    :param x: 输入的相对距离
    :return: 指数衰减系数
    """
    # return 138.565691*torch.exp(0.2217*x)
    return torch.exp(x)

def gaussian_function(x, sigma=1.0):
    """
    高斯分布函数：e^(-x^2 / (2 * sigma^2))
    :param x: 输入的相对距离
    :param sigma: 高斯分布的标准差
    :return: 高斯分布系数
    """
    return torch.exp(-x ** 2 / (2 * sigma ** 2))


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, device, concat=True, relation=False, num_relation=-1,
                 relation_coding='hard', relation_dim=50):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.alpha = alpha
        self.concat = concat
        self.relation = relation

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.device = device
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if self.relation:
            if relation_coding == 'hard':
                emb_matrix = torch.eye(num_relation)  # num_relation 只有relation=True时 有效
                self.relation_embedding = torch.nn.Embedding.from_pretrained(emb_matrix,
                                                                             freeze=True)  # 每种关系 用one-hot向量表示 且不训练
                self.a = nn.Parameter(torch.empty(size=(2 * out_features + num_relation, 1)))
            elif relation_coding == 'soft':
                self.relation_embedding = torch.nn.Embedding(num_relation, relation_dim)
                self.a = nn.Parameter(torch.empty(size=(2 * out_features + relation_dim, 1)))
        else:
            self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, h, adj, q_type, pos):
        adj = adj.to(self.device)
        # h (B,N,D_in)
        Wh = torch.matmul(h, self.W)  # (B, N, D_out)

        a_input = self._prepare_attentional_mechanism_input(Wh)  # (B, N, N, 2*D_out)

        if self.relation:
            long_adj = adj.clone().type(torch.LongTensor).to(self.device) #(B,N,N)
            relation_one_hot = self.relation_embedding(long_adj)  # 得到每个关系对应的one-hot 固定表示 (B,N,N,num_relation)

            # print(relation_one_hot.shape)

            a_input = torch.cat([a_input, relation_one_hot], dim=-1)  # （B, N, N, 2*D_out+num_relation）

        # print(a_input.shape)
        # if q_type == "f_emo":
        #     e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3) + pos.to(self.device))
        # else:
        #     e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3) + pos.to(self.device)) # (B, N , N)  所有部分都参与了计算 包括填充和没有关系连接的节点
            # e = self.adjust_attention_with_distance(e, pos, mode='exponential')
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -9e15 * torch.ones_like(e)  # 计算mask
        # print(adj.shape)
        # print(e.shape)
        # TODO: Solve empty graph issue here!
        attention = torch.where(adj > 0, e, zero_vec)  # adj中非零位置 对应e的部分 保留，零位置(填充或没有关系连接)置为非常小的负数
        attention = F.softmax(attention, dim=2)  # B, N, N
        # if q_type == "f_cau":
        #     attention = self.adjust_attention_with_distance(attention, pos, mode='exponential')
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (B,N,N_out)

        h_prime = self.layer_norm(h_prime)

        if self.concat:
            return F.gelu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # N
        B = Wh.size()[0]  # B
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #
        # print('Wh', Wh.shape)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating],
                                            dim=2)  # (B, N*N, 2*D_out)
        # all_combinations_matrix.shape == (B, N * N, 2 * out_features)

        return all_combinations_matrix.view(B, N, N, 2 * self.out_features)

    def adjust_attention_with_distance(self, attention, distance, mode='exponential', sigma=1.0):
        """
        根据相对距离修正注意力矩阵
        :param attention: 初始的注意力矩阵 (B, N, N)
        :param distance: 相对距离矩阵 (B, N, N)
        :param mode: 修正方式 ('exponential' 或 'gaussian')
        :param sigma: 高斯分布的标准差
        :return: 修正后的注意力矩阵
        """
        if mode == 'exponential':
            # 计算指数衰减系数 e^(-x)
            decay_coeff = exponential_decay(distance)
        elif mode == 'gaussian':
            # 计算高斯分布系数
            decay_coeff = gaussian_function(distance, sigma)
        else:
            raise ValueError("Invalid mode. Choose 'exponential' or 'gaussian'.")

        # 将 decay_coeff 应用到注意力矩阵上
        adjusted_attention = attention * decay_coeff.to(self.device)

        # 对修正后的注意力进行 softmax
        adjusted_attention = F.softmax(adjusted_attention, dim=2)

        return adjusted_attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class RGAT(nn.Module):
    def __init__(self, args, nfeat, nhid, dropout=0.2, alpha=0.2, nheads=2, num_relation=-1):
        """Dense version of GAT."""
        super(RGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, device=args.device,concat=True, relation=True,
                                               num_relation=num_relation) for _ in range(nheads)]  # 多头注意力

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, device=args.device, concat=False,
                                           relation=True, num_relation=num_relation)  # 恢复到正常维度

        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = LayerNorm(nhid)

    def forward(self, x, adj, q_type, pos):
        redisual = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, q_type, pos) for att in self.attentions], dim=-1)  # (B,N,num_head*N_out)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.gelu(self.out_att(x, adj, q_type, pos))  # (B, N, N_out)
        x = self.fc(x)  # (B, N, N_out)
        x = x + redisual
        x = self.layer_norm(x)
        return x