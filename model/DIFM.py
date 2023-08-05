import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.time2vec import time2vec


class DIFM(nn.Module):
    #     def __init__(self, vocabulary_size, embedding_dim,hidden_dim, layers, loss_type, epoch,
    #                  batch_size, learning_rate, lamda, keep_prob, optimizer_type,
    #                  batch_norm, activation_function, verbose, early_stop, loss_weight,
    #                  upsample, maxnum_events, categorical_columns, numerical_columns,
    #                  hidden_unit, prefix, module, random_seed=2019):
    #         super.__init__()
    #         # bind params to class
    #         self.batch_size = batch_size
    #         self.embedding_dim = embedding_dim
    #         self.layers = layers
    #         self.loss_type = loss_type
    #         self.vocabulary_size = vocabulary_size
    #         self.lamda = lamda
    #         self.epoch = epoch
    #         self.random_seed = random_seed
    #         self.keep_prob = np.array(keep_prob)
    #         self.no_dropout = np.array([1 for _ in range(len(keep_prob))])
    #         self.optimizer_type = optimizer_type
    #         self.learning_rate = learning_rate
    #         self.batch_norm = batch_norm
    #         self.verbose = verbose
    #         self.activation_function = activation_function
    #         self.early_stop = early_stop
    #         # len(event_seq)+1(payment)
    #         self.maxnum_events = maxnum_events
    #         # number of categorical_columns and numerical_columns
    #         self.categorical_columns = categorical_columns
    #         self.numerical_columns = numerical_columns
    #         self.nb_features_per_event = self.categorical_columns+self.numerical_columns
    #         self.valid_features = self.nb_features_per_event * self.maxnum_events
    #         self.prefix = prefix
    #         self.hidden_unit = hidden_unit
    #         # whether to utilize the three block(0/1)
    #         self.module = module
    #         self.loss_weight = loss_weight
    #         self.upsample = upsample
    def __init__(self, embedding_dim, hidden_dim, tss_dim, vocabulary_size,
                 num_features, num_events, activation, use_bn=False, use_vote=False,
                 drop_out=0.2, use_gpu=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tss_dim = tss_dim
        self.vocabulary_size = vocabulary_size
        self.num_features = num_features
        self.num_events = num_events
        self.use_bn = use_bn
        self.drop_out = drop_out
        self.use_vote = use_vote
        self.use_gpu = use_gpu
        # initialize model
        self.emb_layer = nn.Embedding(self.vocabulary_size + 1, self.embedding_dim)
        self.activation_function = nn.ReLU()
        if activation == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation == 'identity':
            self.activation_function = nn.Identity()
        self.dropout_layer = nn.Dropout(self.drop_out)
        self.time_encoder = time2vec(self.embedding_dim)
        self.fc1 = nn.Linear(in_features=self.embedding_dim * 2 + tss_dim, out_features=self.hidden_dim)
        self.fc2 = nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.time_linear = nn.Linear(in_features=self.embedding_dim, out_features=self.tss_dim)
        if use_bn:
            self.bn = nn.BatchNorm1d(num_features=num_events)

        if use_vote:
            self.deep_vote = nn.Linear(in_features=1, out_features=1)
            self.wide_vote = nn.Linear(in_features=1, out_features=1)

        self.wide_layer = nn.Embedding(self.vocabulary_size, 1)
        self.lr_bias = nn.Parameter(torch.zeros([1]))
        self.init_weights()

    def init_weights(self):
        self.all_weights = dict()
        self.W1_1 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W1_2 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W1_3 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W2_1 = nn.Parameter(torch.randn(self.embedding_dim + self.tss_dim, self.embedding_dim + self.tss_dim))
        self.W2_2 = nn.Parameter(torch.randn(self.embedding_dim + self.tss_dim, self.embedding_dim + self.tss_dim))
        self.W2_3 = nn.Parameter(torch.randn(self.embedding_dim + self.tss_dim, self.embedding_dim + self.tss_dim))

        self.all_weights['attention_w1_1'] = nn.init.normal_(self.W1_1, mean=0.0, std=0.01)
        self.all_weights['attention_w2_1'] = nn.init.normal_(self.W1_2, mean=0.0, std=0.01)
        self.all_weights['attention_w3_1'] = nn.init.normal_(self.W1_3, mean=0.0, std=0.01)
        self.all_weights['attention_w1_2'] = nn.init.normal_(self.W2_1, mean=0.0, std=0.01)
        self.all_weights['attention_w2_2'] = nn.init.normal_(self.W2_2, mean=0.0, std=0.01)
        self.all_weights['attention_w3_2'] = nn.init.normal_(self.W2_3, mean=0.0, std=0.01)
        self.lr_bias = nn.init.normal_(self.lr_bias, mean=0.0, std=0.01)

    def attention(self, inputs, num, mask=None):
        Q = torch.matmul(inputs, self.all_weights['attention_w1_' + str(num)])
        K = torch.matmul(inputs, self.all_weights['attention_w2_' + str(num)])
        V = torch.matmul(inputs, self.all_weights['attention_w3_' + str(num)])
        a = torch.sum(Q * K, dim=-1) / torch.sqrt(torch.tensor(float(inputs.shape[-1])))
        if mask is not None:
            a = a - (1 - mask) * 1e12
        a = torch.softmax(a, axis=1)
        outputs = a.unsqueeze(-1) * V
        return outputs

    def compute_FM(self, inputs, dim):
        feature_sum = torch.sum(inputs, dim=dim)
        feature_sum_square = torch.square(feature_sum)

        feature_square = torch.square(inputs)
        feature_square_sum = torch.sum(feature_square, dim=dim)
        fm = (feature_sum_square - feature_square_sum) * 0.5
        return fm

    def forward(self, x):
        events_feats_ids, events_feats_values, seq_len = x

        embs = self.emb_layer(events_feats_ids)  # (B,T,F,EMB_DIM)
        batch_size = events_feats_ids.shape[0]
        initial_embs = torch.mul(embs, events_feats_values.unsqueeze(-1))  # (B,T,F,EMB_DIM)
        time_embs = self.time_encoder(time_seq.squeeze(-1))  # (B,T,TSS_DIM)
        time_embs = time_embs.unsqueeze(2)  # (B,T,1,TSS_DIM)
        embs = torch.cat([initial_embs, time_embs], dim=2)

        # model event
        fm1 = self.compute_FM(initial_embs, 2)  # (B,T,EMB_DIM)
        time_out = self.time_linear(time_embs).squeeze(2)  # (B,T,TSS_DIM)
        fm1 = torch.cat([fm1, time_out], dim=2)
        emb_his = self.attention(fm1, 2, masks)
        emb_his = torch.sum(emb_his, dim=1)
        # model field
        fm2 = self.compute_FM(embs, 1)
        emb_f = self.attention(fm2, 1, None)
        emb_f = torch.sum(emb_f, dim=1)
        
        
        pred_ids_new = events_feats_ids[
            [i for i in range(batch_size)], torch.tensor(seq_length - 1, dtype=torch.long)]
        pred_values_new = events_feats_values[
            [i for i in range(batch_size)], torch.tensor(seq_length - 1, dtype=torch.long)]
        pred_event_emb = self.emb_layer(pred_ids_new)
        pred_event_emb = torch.mul(pred_event_emb, pred_values_new.unsqueeze(-1))
        pred_event_emb = torch.sum(pred_event_emb, dim=1)

        # deep_in
        deep_in = torch.cat([emb_his, emb_f,pred_event_emb], dim=1)
        fc1 = self.fc1(deep_in)
        fc1 = self.activation_function(fc1)
        fc1 = self.dropout_layer(fc1)
        deep_out = self.fc2(fc1)

        # wide part
        lr_events_feats_ids = events_feats_ids.reshape([batch_size, -1])
        lr_events_feats_values = events_feats_values.reshape([batch_size, -1])
        lr_events_embs = torch.mul(self.wide_layer(lr_events_feats_ids).squeeze(), lr_events_feats_values)
        lr_out = torch.sum(lr_events_embs, dim=1, keepdim=True) + self.lr_bias
        if self.use_vote:
            deep_out = self.deep_vote(deep_out)
            lr_out = self.wide_vote(lr_out)
        output = deep_out + lr_out
        output = F.sigmoid(output).squeeze(-1)
        return output