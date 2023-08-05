import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Mask_Fusion(nn.Module):
    def __init__(self, num_events, num_features, fusion_type,
                 c_begin, c_end, n_begin, n_end, use_gpu,
                 keep_radio, hidden_dim, mask_record_before):
        super().__init__()
        self.num_events = num_events
        self.num_features = num_features
        self.fusion_type = fusion_type
        self.use_gpu = use_gpu
        self.keep_radio = keep_radio
        self.c_begin = c_begin
        self.c_end = c_end
        self.n_begin = n_begin
        self.n_end = n_end
        self.hidden_dim = hidden_dim
        self.mask_record_before = mask_record_before

        if self.fusion_type == "equal_scale":
            self.categorical_factor = nn.Parameter(torch.randn(1))
            self.numerical_factor = nn.Parameter(torch.randn(1))
        if self.fusion_type == "unequal_scale":
            self.W = nn.Parameter(torch.randn(self.num_events, self.num_features))
            self.b = nn.Parameter(torch.zeros(self.num_events, self.num_features))
        if self.fusion_type == "mlp":
            self.fc1 = nn.Linear(in_features=self.num_events * self.num_features, out_features=self.hidden_dim)
            self.fc2 = nn.Linear(in_features=self.hidden_dim, out_features=self.num_events * self.num_features)
        self.init_para()

    def forward(self, mask):
        # mask: tensor of shape(T,F):(150,30)
        if self.fusion_type == "equal_scale":
            mask_final_categorical = torch.mul(mask[:, self.c_begin:self.c_end], self.categorical_factor)
            mask_final_numerical = torch.mul(mask[:, self.n_begin:self.n_end], self.numerical_factor)
            print('self.categorical_factor', self.categorical_factor)
            print('self.numerical_factor', self.numerical_factor)
            mask_final = torch.cat([mask_final_categorical, mask_final_numerical], dim=1)

        if self.fusion_type == "unequal_scale":
            mask_final = mask * self.W + self.b

        if self.fusion_type == "mlp":
            mask_hidden = self.fc1(mask.reshape(1, -1))
            mask_final = self.fc2(mask_hidden)
            mask_final = mask_final.reshape(self.num_events, self.num_features)

        return mask_final

    def init_para(self):
        print("has init")
        if self.fusion_type == "equal_scale":
            nn.init.normal_(self.categorical_factor, mean=1.0, std=0.01)
            nn.init.normal_(self.numerical_factor, mean=1.0, std=0.01)
            # self.categorical_factor.data = 1.0
            # self.numerical_factor.data = 1.0
        if self.fusion_type == "unequal_scale":
            nn.init.normal_(self.W, mean=1.0, std=0.01)
            nn.init.normal_(self.b, mean=0.0, std=0.01)
        if self.fusion_type == "mlp":
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)








