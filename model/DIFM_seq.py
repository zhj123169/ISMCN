# %load './model/DIFM_seq.py'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.time2vec import time2vec


class DIFM_seq(nn.Module):
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
    def __init__(self, embedding_dim, hidden_dim_1, hidden_dim_2, vocabulary_size,
                 num_events, num_fileds, activation, seq_length, use_bn=False, use_vote=False,
                 drop_out_1=0.2, drop_out_2=0.2, use_gpu=True):
        super(DIFM_seq, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.vocabulary_size = vocabulary_size
        self.num_fileds = num_fileds
        self.num_events = num_events
        self.use_bn = use_bn
        self.drop_out_1 = drop_out_1
        self.drop_out_2 = drop_out_2
        self.seq_length = seq_length
        self.sub_seq_num = int(num_events / seq_length)
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
        self.dropout_layer_1 = nn.Dropout(self.drop_out_1)
        self.dropout_layer_2 = nn.Dropout(self.drop_out_2)
        self.time_encoder = time2vec(self.embedding_dim)
        self.fc1 = nn.Linear(in_features=self.embedding_dim * 3, out_features=self.hidden_dim_1)
        self.fc2 = nn.Linear(in_features=self.hidden_dim_1, out_features=self.hidden_dim_2)
        self.fc3 = nn.Linear(in_features=self.hidden_dim_2, out_features=1)
        self.use_vote = use_vote
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
        self.W2_1 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W3_1 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W1_2 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W2_2 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W3_2 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W1_3 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W2_3 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))
        self.W3_3 = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        self.all_weights['attention_w1_1'] = nn.init.normal_(self.W1_1, mean=0.0, std=0.01)
        self.all_weights['attention_w2_1'] = nn.init.normal_(self.W2_1, mean=0.0, std=0.01)
        self.all_weights['attention_w3_1'] = nn.init.normal_(self.W3_1, mean=0.0, std=0.01)
        self.all_weights['attention_w1_2'] = nn.init.normal_(self.W1_2, mean=0.0, std=0.01)
        self.all_weights['attention_w2_2'] = nn.init.normal_(self.W2_2, mean=0.0, std=0.01)
        self.all_weights['attention_w3_2'] = nn.init.normal_(self.W3_2, mean=0.0, std=0.01)
        self.all_weights['attention_w1_3'] = nn.init.normal_(self.W1_3, mean=0.0, std=0.01)
        self.all_weights['attention_w2_3'] = nn.init.normal_(self.W2_3, mean=0.0, std=0.01)
        self.all_weights['attention_w3_3'] = nn.init.normal_(self.W3_3, mean=0.0, std=0.01)
        self.lr_bias = nn.init.normal_(self.lr_bias, mean=0.0, std=0.01)

    def attention(self, inputs, num, mask=None):
        # for k in self.all_weights.keys():
        #     print("in attention",k,self.all_weights[k].device)
        # if len(self.all_weights.keys()) ==0:
        #     print("AAAAAA")
        # print("inputs.device",inputs.device)
        Q = torch.matmul(inputs, self.all_weights['attention_w1_' + str(num)])
        K = torch.matmul(inputs, self.all_weights['attention_w2_' + str(num)])
        V = torch.matmul(inputs, self.all_weights['attention_w3_' + str(num)])

        a = torch.sum(Q * K, dim=-1) / torch.sqrt(torch.tensor(float(self.embedding_dim)))
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

    def forward(self, x, x1=None, x2=None, x3=None, x4=None, x5=None, x6=None):
        # print("start")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # embs,events_feats_values, masks, time_seq,lr_events_embs,time_embs
        if (type(x) == list):
            if len(x) == 4 and self.emb_layer != None:  # initial
                # print("initial")
                events_feats_ids, events_feats_values, masks, time_seq = x
                # events_feats_ids = torch.tensor(events_feats_ids,dtype =torch.long)
                embs = self.emb_layer(events_feats_ids)
                batch_size = events_feats_ids.shape[0]
                time_embs = None
                lr_events_embs = None

            if len(x) == 4 and self.emb_layer == None:  # lrp
                # print("lrp")
                embs, events_feats_values, lr_events_embs, time_embs = x
                batch_size = events_feats_values.shape[0]
                masks = None
                time_seq = None
                seq_masks = None
            if len(x) == 5:  # IG
                # print("ig")
                embs, events_feats_values, masks, time_seq, lr_events_embs = x
                batch_size = events_feats_values.shape[0]
                time_embs = None
        else:
            if self.emb_layer == None:  # lrp
                # print("lrp2")
                embs, events_feats_values, lr_events_embs, time_embs = x, x1, x2, x3
                batch_size = events_feats_values.shape[0]
                masks = None
                time_seq = None
                seq_masks = None

            elif self.emb_layer != None and x5 == None:  # IG
                print("IG2")
                embs, events_feats_values, masks, time_seq, lr_events_embs = x, x1, x2, x3, x4
                batch_size = events_feats_values.shape[0]
                time_embs = None

            elif self.emb_layer != None and x5 != None and x5.shape != torch.tensor(1).shape:  # Lime
                # print("Lime")
                events_feats_ids, events_feats_values, seq_ids_ap, seq_values_ap, masks, time_seq = x, x1, x2, x3, x4, x5
                # device = events_feats_ids.device
                # print("device1",device )
                # print("begin output")
                # print("inputs:",events_feats_ids.device, events_feats_values.device, masks.device, time_seq.device )
                # for k in self.all_weights.keys():
                #         print(k,self.all_weights[k].device)
                # print("end output")
                # events_feats_ids = torch.tensor(events_feats_ids,dtype =torch.long)
                events_feats_ids = torch.cat([events_feats_ids, seq_ids_ap], dim=2)
                events_feats_values = torch.cat([events_feats_values, seq_values_ap], dim=2)
                embs = self.emb_layer(events_feats_ids)
                batch_size = events_feats_ids.shape[0]
                time_embs = None
                lr_events_embs = None
                # if events_feats_ids.device !=self.all_weights['attention_w1_1'].device:
                #     for k in self.all_weights.keys():
                #         self.all_weights[k] = self.all_weights[k].to(events_feats_ids.device)
                #         print(k,self.all_weights[k].device)

            elif self.emb_layer != None and x5 != None and x5 == 1:  # Lime numerical
                # print("Lime_numerical")
                events_feats_values, events_feats_ids, seq_values_ap, masks, time_seq = x, x1, x2, x3, x4
                events_feats_values = torch.cat([events_feats_values, seq_values_ap], dim=2)
                # events_feats_ids = torch.tensor(events_feats_ids,dtype =torch.long)
                embs = self.emb_layer(events_feats_ids)
                batch_size = events_feats_ids.shape[0]
                time_embs = None
                lr_events_embs = None
            else:
                print("IG3")
                embs, events_feats_values, masks, time_seq, lr_events_embs = x, x1, x2, x3, x4
                batch_size = events_feats_values.shape[0]
                time_embs = None
        # elif len(x)==4 and self.emb_layer == None:
        #     print("lrp in 2")
        #     embs,events_feats_values, lr_events_embs,time_embs = x
        #     batch_size = events_feats_values.shape[0]
        #     seq_masks =None
        # else: #len(x)==5: #IG
        #     embs,events_feats_values, masks, time_seq, lr_events_embs = x
        #     batch_size = events_feats_values.shape[0]
        # else: #lrp
        #     print("lrp in 1")
        #     embs,events_feats_values, lr_events_embs,time_embs = x,events_feats_values,lr_events_embs,time_embs
        #     batch_size = events_feats_values.shape[0]
        #     seq_masks =None
        #     self.emb_layer = None
        # print("device2",device )
        embs = torch.mul(embs, events_feats_values.unsqueeze(-1))
        if time_embs == None:  # lrp
            time_embs = self.time_encoder(time_seq.squeeze(-1))
            time_embs = time_embs.unsqueeze(2)

        embs = torch.cat([embs, time_embs], dim=2)
        # print("embs.device",embs.device)

        # model for each event
        # print("device3",device )
        # print("inputs2:",events_feats_ids.device, events_feats_values.device, masks.device, time_seq.device )
        fm1 = self.compute_FM(embs, 2).to(device)
        # print("device4",device )
        # print("fm1.device",fm1.device)
        emb_his = self.attention(fm1, 1, masks)
        emb_his = torch.sum(emb_his, dim=1)

        fm1_out = fm1.unsqueeze(2)
        fm1_out = fm1_out.view((batch_size, self.sub_seq_num, self.seq_length, self.embedding_dim))
        if masks != None:
            seq_masks = masks.view((batch_size, self.sub_seq_num, self.seq_length))
            seq_masks = torch.sum(seq_masks, dim=2)
            if self.use_gpu:
                seq_masks = torch.where(seq_masks >= 1.0, torch.ones(seq_masks.shape).cuda(),
                                        torch.zeros(seq_masks.shape).cuda())
            else:
                seq_masks = torch.where(seq_masks >= 1.0, torch.ones(seq_masks.shape), torch.zeros(seq_masks.shape))
        # fm1 = fm1.reshape((batch_size,self.seq_length,self.sub_seq_num,self.embedding_dim))

        fm3 = self.compute_FM(fm1_out, 2)
        emb_seq = self.attention(fm3, 3, seq_masks)
        emb_seq = torch.sum(emb_seq, dim=1)

        fm2 = self.compute_FM(embs, 1)
        emb_f = self.attention(fm2, 2, None)
        emb_f = torch.sum(emb_f, dim=1)

        # deep_in
        deep_in = torch.cat([emb_his, emb_f, emb_seq], dim=1)
        fc1 = self.fc1(deep_in)
        fc1 = self.activation_function(fc1)
        fc1 = self.dropout_layer_1(fc1)
        fc2 = self.fc2(fc1)
        fc2 = self.activation_function(fc2)
        fc2 = self.dropout_layer_1(fc2)
        deep_out = self.fc3(fc2)

        # wide part
        if lr_events_embs == None:
            lr_events_feats_ids = events_feats_ids.reshape([batch_size, -1])
            lr_events_feats_values = events_feats_values.reshape([batch_size, -1])
            lr_events_embs = torch.mul(self.wide_layer(lr_events_feats_ids).squeeze(), lr_events_feats_values)

        lr_out = torch.sum(lr_events_embs, dim=1, keepdim=True) + self.lr_bias

        if self.use_vote:
            deep_out = self.deep_vote(deep_out)
            lr_out = self.wide_vote(lr_out)

        output = deep_out + lr_out
        output = F.sigmoid(output).squeeze(-1)
        # print("end")
        return output