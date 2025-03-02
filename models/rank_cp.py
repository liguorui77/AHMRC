from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.GNN import GNN
from models.MultiAttn import MultiAttnModel
from models.SeqContext import SeqContext
from models.almt_layer import HhyperLearningLayer, HhyperLearningEncoder
from models.functions import batch_graphify, split_vision, split_speaker, split_audio
from memory_profiler import profile

from models.gnn_layer import GraphAttentionLayer, RGAT

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert_encoder = BertEncoder(configs)
        # self.gnn = GraphNN(configs)
        # # self.gnn = COGMEN(configs)
        self.gnn = DualGATs(configs)
        self.pred_e = Pre_Predictions(configs)
        self.device = configs.device
        self.fc_audio = nn.Linear(100, configs.av_hidden_size)
        self.fc_vision = nn.Linear(512, configs.av_hidden_size)
        self.fc_text = nn.Linear(768, configs.av_hidden_size)
        self.multimodel = MultiAttnModel(2, 256, 4, 1024, configs.out_feature, 0)
        self.relative_position_encoder = LearnableRelativePositionEmbeddings(40, 1, configs.device)
        # self.h_hyper_layer = HhyperLearningEncoder(256, depth=1, fusion_depth=1, heads=4, dim_head=64, dropout=0)
        # self.fc = nn.Linear(configs.av_hidden_size, 768)
        # self.fc1 = nn.Linear(768, 100)
        # self.rnn = SeqContext(768,512,configs)

    def forward(self, query, query_mask, query_seg, query_len, seq_len, doc_len, q_type, adj, audio, vision, pos):
        text, query_h, mask_feature = self.bert_encoder(query, query_mask, query_seg, query_len, seq_len, doc_len)
        audio = self.fc_audio(audio.to(self.device))
        vision = self.fc_vision(vision.to(self.device))
        text_h = self.fc_text(text)

        doc_sents_h, reg_loss = self.multimodel(text_h, audio, vision)

        position_embeddings = self.relative_position_encoder(doc_sents_h.size(0), max(doc_len))

        # doc_sents_h = torch.cat((text_h,audio,vision),dim=-1)
        # doc_sents_h = self.rnn(torch.cat((text_h, audio, vision),dim=-1))

        doc_sents_h = self.gnn(doc_sents_h, adj, q_type, position_embeddings)
        doc_sents_h = torch.cat((query_h, doc_sents_h), dim=-1)
        pred = self.pred_e(doc_sents_h)
        return pred, reg_loss

    def loss_pre(self, pred, true, mask):
        true = torch.FloatTensor(true.float()).to(self.device)  # shape: batch_size, seq_len
        mask = torch.BoolTensor(mask.bool()).to(self.device)
        pred = pred.masked_select(mask)
        true = true.masked_select(mask)
        # weight = torch.where(true > 0.5, 2, 1)
        criterion = nn.BCELoss()
        # criterion = nn.functional.binary_cross_entropy_with_logits
        return criterion(pred, true)

    def normalize(self, data):
        flat_data = data.reshape(-1, data.size(-1))

        # 对每一列进行归一化
        min_vals = flat_data.min(dim=0, keepdim=True).values
        max_vals = flat_data.max(dim=0, keepdim=True).values
        normalized_data = (flat_data - min_vals) / (max_vals - min_vals + 1e-8)
        normalized_data = normalized_data.reshape(-1, data.shape[-2], data.shape[-1])
        return normalized_data


class BertEncoder(nn.Module):
    def __init__(self, configs):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)
        # self.bert = RobertaModel.from_pretrained(configs.bert_cache_path)
        # self.tokenizer = RobertaTokenizer.from_pretrained(configs.bert_tokenizer)
        self.fc = nn.Linear(768, 1)
        self.fc_query = nn.Linear(768, 1)
        self.device = configs.device
        # self.SentenceEncoder = SentenceEncoder(768)

    def forward(self, query, query_mask, query_seg, query_len, seq_len, doc_len):
        hidden_states = self.bert(input_ids=query.to(self.device),
                                  attention_mask=query_mask.to(self.device),
                                  token_type_ids=query_seg.to(self.device))[0]
        # hidden_states = self.bert(input_ids=query.to(self.device),
        #                           attention_mask=query_mask.to(self.device))[0]
        # print(f"Current memory allocated: {torch.cuda.memory_allocated()} bytes")
        # print(f"Max memory allocated: {torch.cuda.max_memory_allocated()} bytes")

        hidden_states, mask_doc, query_state, mask_query = self.get_sentence_state(hidden_states, query_len, seq_len, doc_len)

        alpha = self.fc(hidden_states).squeeze(-1)  # bs, max_doc_len, max_seq_len
        mask_doc_c = mask_doc
        # mask_doc对单词进行填充 0表示无语义
        mask_doc = 1 - mask_doc # bs, max_doc_len, max_seq_len
        alpha.data.masked_fill_(mask_doc.bool(), -9e5)
        alpha = F.softmax(alpha, dim=-1).unsqueeze(-1).repeat(1, 1, 1, hidden_states.size(-1))
        hidden_states = torch.sum(alpha * hidden_states, dim=2) # bs, max_doc_len, 768

        alpha_q = self.fc_query(query_state).squeeze(-1)  # bs, query_len
        mask_query = 1 - mask_query  # bs, max_query_len
        alpha_q.data.masked_fill_(mask_query.bool(), -9e5)
        alpha_q = F.softmax(alpha_q, dim=-1).unsqueeze(-1).repeat(1, 1, query_state.size(-1))
        query_state = torch.sum(alpha_q * query_state, dim=1)  # bs, 768
        query_state = query_state.unsqueeze(1).repeat(1, hidden_states.size(1), 1)

        # 第一步：沿着 max_seq_len 维度进行合并，求和
        mask_sum = mask_doc_c.sum(dim=2)
        # 第二步：如果合并后的结果大于0，则该句子有效
        mask_feature = (mask_sum > 0).float().unsqueeze(-1)

        # doc_sents_h = torch.cat((query_state, hidden_states), dim=-1)
        return hidden_states.to(self.device), query_state.to(self.device), mask_feature.to(self.device)

    def get_sentence_state(self, hidden_states, query_lens, seq_lens, doc_len):
        # 对问题的每个token做注意力，获得问题句子的向量表示；对文档的每个句子的token做注意力，得到每个句子的向量表示
        sentence_state_all = []
        query_state_all = []
        mask_all = []
        mask_query = []
        max_seq_len = 0
        for seq_len in seq_lens: # 找出最长的一句话包含多少token
            for l in seq_len:
                max_seq_len = max(max_seq_len, l)
        max_doc_len = max(doc_len) # 最长的文档包含多少句子
        max_query_len = max(query_lens)  # 最长的问句包含多少token
        for i in range(hidden_states.size(0)):  # 对每个batch
            # 对query
            query = hidden_states[i, 1: query_lens[i] + 1]
            assert query.size(0) == query_lens[i]
            if query_lens[i] < max_query_len:
                query = torch.cat([query, torch.zeros((max_query_len - query_lens[i], query.size(1))).to(self.device)], dim=0)
            query_state_all.append(query.unsqueeze(0))
            mask_query.append([1] * query_lens[i] + [0] * (max_query_len -query_lens[i]))
            # 对文档sentence
            mask = []
            begin = query_lens[i] + 2  # 2是[cls], [sep]
            sentence_state = []
            for seq_len in seq_lens[i]:
                sentence = hidden_states[i, begin: begin + seq_len]
                begin += seq_len
                if sentence.size(0) < max_seq_len:
                    sentence = torch.cat([sentence, torch.zeros((max_seq_len - seq_len, sentence.size(-1))).to(self.device)],
                                         dim=0)
                sentence_state.append(sentence.unsqueeze(0))
                mask.append([1] * seq_len + [0] * (max_seq_len - seq_len))
            # print(sentence_state)
            sentence_state = torch.cat(sentence_state, dim=0).to(self.device)
            if sentence_state.size(0) < max_doc_len:
                mask.extend([[0] * max_seq_len] * (max_doc_len - sentence_state.size(0)))
                padding = torch.zeros(
                    (max_doc_len - sentence_state.size(0), sentence_state.size(-2), sentence_state.size(-1)))
                sentence_state = torch.cat([sentence_state, padding.to(self.device)], dim=0)
            sentence_state_all.append(sentence_state.unsqueeze(0))
            mask_all.append(mask)
        query_state_all = torch.cat(query_state_all, dim=0).to(self.device)
        mask_query = torch.tensor(mask_query).to(self.device)
        sentence_state_all = torch.cat(sentence_state_all, dim=0).to(self.device)
        mask_all = torch.tensor(mask_all).to(self.device)
        return sentence_state_all, mask_all, query_state_all, mask_query

    # def mean_pooling(self, hidden_states, mask):
    #     # mask 的形状应为 (bs, max_doc_len, max_seq_len)，值为 1 表示有效词，0 表示填充部分
    #
    #     mask = mask.unsqueeze(-1).float()  # (bs, max_doc_len, max_seq_len, 1)
    #     sum_hidden = torch.sum(hidden_states * mask, dim=2)  # (bs, max_doc_len, hidden_size)
    #     lengths = torch.sum(mask, dim=2)  # (bs, max_doc_len, 1)
    #     lengths[lengths == 0] = -9e5
    #     sentence_vectors = sum_hidden / lengths  # (bs, max_doc_len, hidden_size)
    #     return sentence_vectors


class LearnableRelativePositionEmbeddings(nn.Module):
    def __init__(self, max_len, embedding_dim, device):
        """
        初始化可学习的相对位置编码。

        :param max_len: 序列的最大长度
        :param embedding_dim: 嵌入维度
        """
        super(LearnableRelativePositionEmbeddings, self).__init__()
        # 相对位置的嵌入矩阵
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # 创建一个可学习的嵌入矩阵，大小为 (2 * max_len - 1, embedding_dim)
        # 2 * max_len - 1 是因为相对位置的距离可以从 -max_len 到 max_len
        self.position_embeddings = nn.Embedding(2 * max_len - 1, embedding_dim)
        self.device = device


    def forward(self, batch_size, seq_len):
        """
        获取输入序列的相对位置编码。

        :param seq_len: 输入序列的长度
        :return: 相对位置编码
        """
        # 生成相对位置的索引
        relative_positions = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)

        window_size = 5
        relative_positions = torch.clamp(relative_positions, min=-window_size, max=window_size)
        # 将负数的位置映射到嵌入矩阵的索引范围内
        # 相对位置的范围是 [-seq_len + 1, seq_len - 1]
        relative_positions = relative_positions + (self.max_len - 1)

        # 通过嵌入矩阵获取相对位置编码
        relative_position_embeddings = self.position_embeddings(relative_positions.to(self.device))

        relative_positions = relative_position_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
        relative_positions = relative_positions.squeeze(-1)

        return relative_positions

class GraphNN(nn.Module):
    def __init__(self, configs):
        super(GraphNN, self).__init__()
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]  # [1024, 256]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')] # [4]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp, configs.device)
            )

    def forward(self, doc_sents_h, doc_len, adj):
        batch, max_doc_len, _ = doc_sents_h.size()
        assert max(doc_len) == max_doc_len
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            doc_sents_h = gnn_layer(doc_sents_h, adj)
        return doc_sents_h

class ContextualBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        :param input_size: 输入特征的维度（hidden_size）
        :param hidden_size: LSTM 隐藏层的维度
        :param num_layers: LSTM 堆叠的层数
        :param dropout: Dropout 比例
        """
        super(ContextualBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

    def forward(self, x):
        """
        :param x: 输入特征，形状为 (batch_size, max_sentence_num, hidden_size)
        :return: 上下文建模后的特征，形状为 (batch_size, max_sentence_num, hidden_size)
        """
        lstm_out, _ = self.bilstm(x)  # (batch_size, max_sentence_num, hidden_size)
        return lstm_out

class DualGATs(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        SpkGAT = []
        # DisGAT = []
        for _ in range(args.gnn_layers):
            SpkGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=6))
            #DisGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=18))

        self.SpkGAT = nn.ModuleList(SpkGAT)
        # self.DisGAT = nn.ModuleList(DisGAT)

        # self.affine1 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        # nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        # self.affine2 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        # nn.init.xavier_uniform_(self.affine2.data, gain=1.414)
        #
        # self.diff_loss = DiffLoss(args)
        # self.beta = 0.3

        # in_dim = args.hidden_dim * 2 + args.emb_dim
        # # output mlp layers
        # layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        # for _ in range(args.mlp_layers - 1):
        #     layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        # layers += [nn.Linear(args.hidden_dim, num_class)]

        # self.out_mlp = nn.Sequential(*layers)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, utterance_features, semantic_adj, q_type, pos):
        '''
        :param tutterance_features: (B, N, emb_dim)
        :param xx_adj: (B, N, N)
        :return:
        '''
        # semantic_adj = self.multimodal_feat(semantic_adj, doc_len_b)
        # semantic_adj_vt = self.multimodal_feat(semantic_adj, doc_len_b, "vision")
        # feat_ta = torch.cat((utterance_features, audio), dim=1)
        # feat_tv = torch.cat((utterance_features, vision), dim=1)
        # feat_av = torch.cat((audio, vision), dim=1)
        # doc_len = utterance_features.size(1)
        H0 = F.relu(self.fc1(utterance_features))  # (B, N, hidden_dim)
        H = [H0]
        diff_loss = 0
        for l in range(self.args.gnn_layers):
            H1_semantic = self.SpkGAT[l](H[l], semantic_adj, q_type, pos)
            H1_semantic_out = self.drop(H1_semantic) if l < self.args.gnn_layers - 1 else H1_semantic
            H.append(H1_semantic_out)

        H.append(utterance_features)

        H = torch.cat([H[-2], H[-1]], dim=2)  # (B, N, hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
        return H
        # H0_ta = F.relu(self.fc0(feat_ta))  # (B, N, hidden_dim)
        # H_ta = [H0_ta]
        # diff_loss = 0
        # for l in range(self.args.gnn_layers):
        #     H1_semantic = self.SpkGAT[l](H_ta[l], semantic_adj)
        #     H1_semantic_out = self.drop(H1_semantic) if l < self.args.gnn_layers - 1 else H1_semantic
        #     H_ta.append(H1_semantic_out)
        # H_ta.append(feat_ta)
        # H_ta = torch.cat([H_ta[-2], H_ta[-1]], dim=2)  # (B, N, hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
        # text0, audio0 = torch.chunk(H_ta, 2, dim=1)
        #
        # H0_tv = F.relu(self.fc1(feat_tv))  # (B, N, hidden_dim)
        # H_tv = [H0_tv]
        # diff_loss = 0
        # for l in range(self.args.gnn_layers):
        #     H1_semantic = self.SpkGAT[l](H_tv[l], semantic_adj)
        #     H1_semantic_out = self.drop(H1_semantic) if l < self.args.gnn_layers - 1 else H1_semantic
        #     H_tv.append(H1_semantic_out)
        # H_tv.append(feat_tv)
        # H_tv = torch.cat([H_tv[-2], H_tv[-1]], dim=2)  # (B, N, hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
        # text1, vision0 = torch.chunk(H_tv, 2, 1)
        #
        # H0_av = F.relu(self.fc2(feat_av))  # (B, N, hidden_dim)
        # H_av = [H0_av]
        # diff_loss = 0
        # for l in range(self.args.gnn_layers):
        #     H1_semantic = self.SpkGAT[l](H_av[l], semantic_adj)
        #     H1_semantic_out = self.drop(H1_semantic) if l < self.args.gnn_layers - 1 else H1_semantic
        #     H_av.append(H1_semantic_out)
        # H_av.append(feat_av)
        # H_av = torch.cat([H_av[-2], H_av[-1]], dim=2)  # (B, N, hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
        # audio1, vision1 = torch.chunk(H_av, 2, 1)
        #
        # H_t = text0+text1
        # H_a = audio0+audio1
        # H_v = vision0+vision1
        # return (H_t+H_a+H_v)/6

    # def forward(self, utterance_features, audio, vision, semantic_adj, doc_len_b):
    #     '''
    #     :param tutterance_features: (B, N, emb_dim)
    #     :param xx_adj: (B, N, N)
    #     :return:
    #     '''
    #     semantic_adj = self.multimodal_feat(semantic_adj, doc_len_b)
    #     # semantic_adj_vt = self.multimodal_feat(semantic_adj, doc_len_b, "vision")
    #     feat_ta = torch.cat((utterance_features, audio), dim=1)
    #     feat_tv = torch.cat((utterance_features, vision), dim=1)
    #     doc_len = utterance_features.size(1)
    #     # H0 = F.relu(self.fc1(utterance_features))  # (B, N, hidden_dim)
    #     # H = [H0]
    #     # diff_loss = 0
    #     # for l in range(self.args.gnn_layers):
    #     #     H1_semantic = self.SpkGAT[l](H[l], semantic_adj)
    #     #     H1_semantic_out = self.drop(H1_semantic) if l < self.args.gnn_layers - 1 else H1_semantic
    #     #     H.append(H1_semantic_out)
    #     #
    #     # H.append(utterance_features)
    #     #
    #     # H = torch.cat([H[-2], H[-1]], dim=2)  # (B, N, hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
    #     H0_ta = F.relu(self.fc1(feat_ta))  # (B, N, hidden_dim)
    #     H_ta = [H0_ta]
    #     diff_loss = 0
    #     for l in range(self.args.gnn_layers):
    #         H1_semantic = self.SpkGAT[l](H_ta[l], semantic_adj)
    #         H1_semantic_out = self.drop(H1_semantic) if l < self.args.gnn_layers - 1 else H1_semantic
    #         H_ta.append(H1_semantic_out)
    #     H_ta.append(feat_ta)
    #     H_ta = torch.cat([H_ta[-2], H_ta[-1]], dim=2)  # (B, N, hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
    #
    #     H0_tv = F.relu(self.fc1(feat_tv))  # (B, N, hidden_dim)
    #     H_tv = [H0_tv]
    #     diff_loss = 0
    #     for l in range(self.args.gnn_layers):
    #         H1_semantic = self.SpkGAT[l](H_tv[l], semantic_adj)
    #         H1_semantic_out = self.drop(H1_semantic) if l < self.args.gnn_layers - 1 else H1_semantic
    #         H_tv.append(H1_semantic_out)
    #     H_tv.append(feat_tv)
    #     H_tv = torch.cat([H_tv[-2], H_tv[-1]], dim=2)  # (B, N, hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
    #
    #     H = (H_ta + H_tv)*0.5 #缩放因子
    #     return H[:, :doc_len, :]

    def multimodal_feat(self, adj, doc_len_b):
        batch_size = adj.size(0)
        #  [bs,doc_len,hz] -> [bs,doc_len1+doc_len2,hz] 用0填充无连接的句子
        max_len = max(doc_len_b)
        new_adj = torch.zeros(batch_size, max_len + max_len, max_len + max_len, dtype=torch.long)
        new_adj[:, :max_len, :max_len] = adj
        for batch_idx in range(batch_size):
            doc_len = doc_len_b[batch_idx]
            for i in range(doc_len):
                new_adj[batch_idx, i, i + max_len] = 1  # 多模态自循环
                new_adj[batch_idx, i + max_len, i] = 1  # 多模态自循环
                new_adj[batch_idx, i + max_len, i + max_len] = 1
        new_adj[:, max_len:, max_len:] = adj[:, :max_len, :max_len]
        return new_adj


class COGMEN(nn.Module):
    def __init__(self, configs):
        super(COGMEN, self).__init__()

        u_dim = 768
        if configs.rnn == "transformer":
            g_dim = configs.hidden_size
        else:
            g_dim = 200
        h1_dim = configs.hidden_size
        h2_dim = configs.hidden_size
        hc_dim = configs.hidden_size

        self.wp = configs.wp
        self.wf = configs.wf
        self.device = configs.device
        self.n_speakers = configs.n_speakers

        self.rnn = SeqContext(u_dim, g_dim, configs)
        self.gcn = GNN(g_dim, h1_dim, h2_dim, configs)

    def get_rep(self, input_tensor,doc_len,speakers,speaker_tensor):

        # 创建一个空集合来存储不同的元素
        speakers_list = set()

        # 遍历列表，并将每个子列表中的元素添加到集合中
        for speaker in speakers:
            for item in speaker:
                speakers_list.add(item)
        # n_speakers = len(speakers_list)

        #根据说话者生成边类型字典
        edge_type_to_idx={}
        for j in range(self.n_speakers):
            for k in range(self.n_speakers):
                edge_type_to_idx[str(j) + str(k) + "0"] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + "1"] = len(edge_type_to_idx)

        # [batch_size, mx_len, D_g]
        # print("input_tensor",data["input_tensor"].shape)  # d_g 1380
        # print("text_len_tensor",data["text_len_tensor"].shape) [bs]
        # print(data["speaker_tensor"]) #[bs,mx_len]
        node_features = self.rnn(doc_len, input_tensor.float())
        #print("node_features",node_features.shape)  # [batch_size, mx_len, D_g] d_g 100
        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            doc_len,
            speaker_tensor,
            self.wp,
            self.wf,
            edge_type_to_idx,
            self.device,
        )
        # print(features.shape, edge_index.shape, edge_type.shape, edge_index_lengths.shape) #[utt_len, D_g] [2, E]  [E]  [bs]
        graph_out = self.gcn(features, edge_index, edge_type)
        return graph_out, features  #  [节点数量，特征维度(100)]

    def forward(self, input_tensor,doc_len_tensor,speakers,speaker_tensor):
        graph_out, features = self.get_rep(input_tensor,doc_len_tensor,speakers,speaker_tensor)

        #(utt_len,hz)->(bs,mx_len,hz)
        # 根据文档长度划分句子
        doc_len = doc_len_tensor.tolist()
        doc_sents_split = torch.split(graph_out, doc_len)
        # 使用零填充不足长度的文档
        max_doc_len = max(doc_len)
        padded_doc_sents = [torch.cat((doc, torch.zeros(max_doc_len - doc.size(0), doc.size(1), device=self.device)), dim=0) for doc in doc_sents_split]

        # 将文档表示组合成一个张量
        doc_tensor = torch.stack(padded_doc_sents)
        # 调整张量的形状
        doc_tensor = doc_tensor.view(len(doc_len), -1, graph_out.size(-1))

        return doc_tensor.to(self.device)

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        if self.concat_gin_gout:
            loss = self.clf.get_loss(
                torch.cat([features, graph_out], dim=-1),
                data["label_tensor"],
                data["text_len_tensor"],
            )
        else:
            loss = self.clf.get_loss(
                graph_out, data["label_tensor"], data["text_len_tensor"]
            )

        return loss

class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = configs.pre_dim
        self.out_e = nn.Linear(self.feat_dim, 1)

    def forward(self, doc_sents_h):
        pred_e = self.out_e(doc_sents_h).squeeze(-1)  # bs, max_doc_len, 1
        pred_e = torch.sigmoid(pred_e)
        return pred_e # shape: bs ,max_doc_len

# if __name__ == "__main__":
#     def normalize(data):
#         flat_data = data.reshape(-1, data.size(-1))
#
#         # 对每一列进行归一化
#         min_vals = flat_data.min(dim=0, keepdim=True).values
#         max_vals = flat_data.max(dim=0, keepdim=True).values
#         normalized_data = (flat_data - min_vals) / (max_vals - min_vals + 1e-8)
#         normalized_data = normalized_data.reshape(-1, data.shape[-2], data.shape[-1])
#
#         return normalized_data
#
#     data = torch.rand(5, 5, 5) * 20 - 10  # 示例数据在 [-10, 10] 之间
#     print(data)
#     data1 = normalize(data)
#     print(data1)
