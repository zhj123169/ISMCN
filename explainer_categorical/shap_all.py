import torch
import numpy as np
import logging
import copy
import itertools
from scipy.special import binom
from sklearn.linear_model import LassoLarsIC
from utils.oss_utils import get_bucket, upload_file_to_oss
from datetime import datetime


class kenel_shap():
    def __init__(self, model, explain_data, bg_data, logger, use_gpu,
                 c_begin, c_end, n_begin, n_end, max_batch_size=2048, mode=0):
        """
            model: model to be interpreted
            data: background data of the dataset
                bg_data: {
                    'seq_ids': matrix [150, 30]
                    'seq_values': matrix [150, 30]
                }
        """
        self.use_gpu = use_gpu
        self.logger = logger
        self.model = model
        self.mode = mode
        self.c_begin = c_begin
        self.c_end = c_end
        self.n_begin = n_begin
        self.n_end = n_end
        self.max_batch_size = max_batch_size

        bg_model_input = self.make_bg_input(bg_data, explain_data)
        self.bg_data_id = bg_model_input[0]

        self.bg_data_val = bg_model_input[1]

        self.num_event_feats = self.bg_data_id.shape[1]

        # self.num_field_feats = self.bg_data_id.shape[2]
        if mode == 0:
            self.num_field_feats = self.c_end - self.c_begin
        else:
            self.num_field_feats = self.bg_data_id.shape[2]
        self.num_field_event_feats = int(self.num_event_feats * self.num_field_feats)

        if self.use_gpu:
            bg_model_input = [f.cuda() for f in bg_model_input]
        with torch.no_grad():
            bg_preds = self.model(bg_model_input).cpu().numpy()
        self.f_bg = bg_preds
        self.expected_v = self.f_bg
        self.M = self.num_field_event_feats

    def shap_values(self, X, sample_per_input=1000000):
        """
        X: a dict of model input;
        """
        explanation = self.explain(X, sample_per_input)
        return explanation

    def explain(self, instance, n_samples):
        t1i = datetime.now()
        model_input = self.make_inst_input(instance)
        if self.use_gpu:
            model_input = [f.cuda() for f in model_input]
        with torch.no_grad():
            self.fx = self.model(model_input).cpu().numpy()
        self.logger.info("background score:{:.4f}".format(float(self.f_bg)))
        self.logger.info("instance prediction score:{:.4f}\n".format(float(self.fx)))
        if n_samples != 0:
            self.n_samples = n_samples
        else:
            self.n_samples = 2 * self.num_field_event_feats + 2 ** 11
        self.max_samples = 2 ** 30

        self.allocate(self.mode)
        # 取权重
        num_subset_sizes = int(np.ceil((self.num_field_event_feats - 1) / 2.0))
        num_paired_subset_sizes = int(np.floor((self.num_field_event_feats - 1) / 2.0))
        weight_vector = np.array([(self.num_field_event_feats - 1.0) / (i * (self.num_field_event_feats - i))
                                  for i in range(1, num_subset_sizes + 1)])
        weight_vector[:num_paired_subset_sizes] *= 2
        weight_vector /= np.sum(weight_vector)
        # 取样本
        num_full_subsets = 0
        num_samples_left = self.n_samples
        group_inds = np.arange(self.M, dtype='int64')
        mask = np.zeros(self.M)
        remaining_weight_vector = copy.copy(weight_vector)
        # 取完整样本
        for subset_size in range(1, num_subset_sizes + 1):
            num_subsets = binom(self.M, subset_size)
            if subset_size <= num_paired_subset_sizes:
                num_subsets *= 2

            if num_samples_left * remaining_weight_vector[subset_size - 1] / num_subsets >= 1.0 - 1e-8:
                num_full_subsets += 1
                num_samples_left -= num_subsets

                # rescale what's left of the remaining weight vector to sum to 1
                if remaining_weight_vector[subset_size - 1] < 1.0:
                    remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                # add all the samples of the current subset size
                w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    w /= 2.0
                for inds in itertools.combinations(group_inds, subset_size):
                    mask[:] = 0.0
                    mask[np.array(inds, dtype='int64')] = 1.0
                    self.addsample(instance, mask, w)
                    if subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)
                        self.addsample(instance, mask, w)
            else:
                break

        nfixed_samples = self.nsamplesAdded
        samples_left = self.n_samples - self.nsamplesAdded
        self.logger.info("samples_added = {0}, samples_left = {1}".format(nfixed_samples, samples_left))
        if num_full_subsets != num_subset_sizes:
            # 求权重

            remaining_weight_vector = copy.copy(weight_vector)
            remaining_weight_vector[:num_paired_subset_sizes] /= 2  # because we draw two samples each below
            remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
            remaining_weight_vector /= np.sum(remaining_weight_vector)

            ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
            ind_set_pos = 0
            used_masks = {}
            # 取剩下的样本
            t_start = datetime.now()
            t1 = datetime.now() - t_start
            t2 = datetime.now() - t_start
            t3 = datetime.now() - t_start
            t4 = datetime.now() - t_start

            while samples_left > 0 and ind_set_pos < len(ind_set):
                # if samples_left%1000 ==0:
                #     self.logger.info("samples left:{}".format(samples_left))
                # if (samples_left-1)%1000 ==0:
                #     self.logger.info("samples left:{}".format(samples_left))
                ta_start = datetime.now()
                mask.fill(0.0)
                ind = ind_set[ind_set_pos]  # we call np.random.choice once to save time and then just read it here
                ind_set_pos += 1
                subset_size = ind + num_full_subsets + 1
                mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                mask_tuple = tuple(mask)
                new_sample = False
                ta_end = datetime.now()
                t1 = t1 + ta_end - ta_start
                tb_start = datetime.now()
                if mask_tuple not in used_masks:
                    new_sample = True
                    used_masks[mask_tuple] = self.nsamplesAdded
                    samples_left -= 1
                    td_start = datetime.now()
                    self.addsample(instance, mask, 1.0)
                    td_end = datetime.now()
                    t4 = t4 + td_end - td_start
                else:
                    self.kernelWeights[used_masks[mask_tuple]] += 1.0
                tb_end = datetime.now()
                t2 = t2 + tb_end - tb_start
                tc_start = datetime.now()
                if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                    mask[:] = np.abs(mask - 1)
                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    if new_sample:
                        samples_left -= 1
                        te_start = datetime.now()
                        self.addsample(instance, mask, 1.0)
                        te_end = datetime.now()
                        t4 = t4 + te_end - te_start
                    else:
                        # we know the compliment sample is the next one after the original sample, so + 1
                        self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0
                tc_end = datetime.now()
                t3 = t3 + tc_end - tc_start

            t_end = datetime.now()
            t_all = t_end - t_start

            weight_left = np.sum(weight_vector[num_full_subsets:])
            self.logger.info("all time:{},   t1:{},     t2:{},     t3:{}      t4:{}    ".format(t_all, t1, t2, t3, t4))
            self.logger.info("samples added:{}".format(self.n_samples))
            self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()
        t2i = datetime.now()
        self.run(instance)
        t3i = datetime.now()
        phi, phi_var = self.solve(self.n_samples / self.max_samples)
        print("sample shap cost {}s, run instance cost {}s".format((t2i - t1i), (t3i - t2i)))
        return phi

    def addsample(self, x, m, w):
        # self.addsample(instance, mask, w)
        offset = self.nsamplesAdded

        mask = torch.tensor(m.reshape(self.num_event_feats, self.num_field_feats))
        self.synth_data_ids[offset, :, self.c_begin:self.c_end] = torch.where(mask == 1.0,
                                                                              x['seq_ids'][:, :,
                                                                              self.c_begin:self.c_end],
                                                                              self.synth_data_ids[offset, :,
                                                                              self.c_begin:self.c_end])
        # 下面原有方案
        # mask = m == 1.0
        # groups = np.arange(self.M, dtype='int64')[mask]
        # evaluation_ids = x['seq_ids'][:, :, self.c_begin:self.c_end].reshape((-1,self.num_field_event_feats))[:, mask]

        # temp_id = self.synth_data_ids[offset, :, self.c_begin:self.c_end].reshape((-1,self.num_field_event_feats))
        # temp_id[:, groups] = evaluation_ids
        # self.synth_data_ids[offset, :, self.c_begin:self.c_end] = temp_id.reshape(self.num_event_feats, self.num_field_feats)

        # self.synth_data_vals[offset, :, groups] = evaluation_vals
        if offset < self.seq_masks.shape[0]:
            seq_mask = x['seq_mask']
            seq_len = x['seq_lengths']
            seq_tss = x['seq_tss']
            self.seq_masks[offset] = seq_mask
            self.seq_lengths[offset] = seq_len
            self.seq_tss[offset] = seq_tss

        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def allocate(self, mode):
        self.synth_data_ids = self.bg_data_id.repeat(self.n_samples, 1, 1)

        self.synth_data_vals = self.bg_data_val.repeat(self.max_batch_size, 1, 1)
        self.seq_masks = torch.zeros((self.max_batch_size, self.num_event_feats))
        self.seq_lengths = torch.zeros((self.max_batch_size))
        self.seq_tss = torch.zeros((self.max_batch_size, self.num_event_feats))

        self.maskMatrix = np.zeros((self.n_samples, self.M))
        self.kernelWeights = np.zeros(self.n_samples)
        self.ey = np.zeros((self.n_samples))
        self.nsamplesAdded = 0

    def run(self, instance, batch_size=1024):
        modelOut = []
        for model_input in self.make_disturbed_input(batch_size):
            if self.use_gpu:
                model_input = [f.cuda() for f in model_input]
            with torch.no_grad():
                out = self.model(model_input).cpu().numpy()
            modelOut.append(out)
        self.ey = np.concatenate(modelOut, axis=0)

    def solve(self, fraction_evaluated):
        eyAdj = self.ey - self.f_bg
        s = np.sum(self.maskMatrix, 1)

        nonzero_inds = np.arange(self.M)
        w_aug = np.hstack((self.kernelWeights * (self.M - s), self.kernelWeights * s))
        w_sqrt_aug = np.sqrt(w_aug)
        eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.fx - self.f_bg)))
        eyAdj_aug *= w_sqrt_aug
        mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((self.maskMatrix, self.maskMatrix - 1))))

        nonzero_inds = np.nonzero(LassoLarsIC(criterion="aic").fit(mask_aug, eyAdj_aug).coef_)[0]
        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)

        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (self.fx - self.f_bg)
        etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[:-1]]) - self.maskMatrix[:, nonzero_inds[-1]])

        tmp = np.transpose(np.transpose(etmp) * np.transpose(self.kernelWeights))
        etmp_dot = np.dot(np.transpose(tmp), etmp)
        try:
            tmp2 = np.linalg.inv(etmp_dot)
        except np.linalg.LinAlgError:
            tmp2 = np.linalg.pinv(etmp_dot)
        w = np.dot(tmp2, np.dot(np.transpose(tmp), eyAdj2))

        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.fx - self.f_bg) - sum(w)
        # self.logger.info("phi = {0}".format(phi))

        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))

    def make_disturbed_input(self, batch_size):
        # model_input = [
        #             data['seq_ids'],
        #             data['seq_values'],
        #             data['seq_mask'],
        #             data['seq_tss']
        #         ]
        used_sample = 0
        while used_sample < self.nsamplesAdded:
            # if batch_cnt % 100 == 0:
            if used_sample < self.nsamplesAdded and used_sample + batch_size > self.nsamplesAdded:
                model_input = [

                    self.synth_data_ids[used_sample: self.nsamplesAdded],
                    self.synth_data_vals[0: self.nsamplesAdded - used_sample],
                    self.seq_masks[0: self.nsamplesAdded - used_sample],
                    self.seq_tss[0: self.nsamplesAdded - used_sample]
                ]

                used_sample = self.nsamplesAdded
                yield model_input
            else:
                s = used_sample
                t = used_sample + batch_size
                model_input = [
                    self.synth_data_ids[s: t],
                    self.synth_data_vals[0: batch_size],
                    self.seq_masks[0: batch_size],
                    self.seq_tss[0: batch_size]
                ]

                used_sample += batch_size
                yield model_input

    def save_log(self, oss_path):
        oss_bucket = get_bucket()
        upload_file_to_oss(oss_bucket, oss_path + '/explainer.log', './shap.log')

    def make_bg_input(self, bg_data, explain_data, mask_size=150):
        if self.mode == 0:
            seq_ids = bg_data['seq_ids'].unsqueeze(0)[:, :, self.c_begin:self.c_end]
            explain_seq_ids = explain_data['seq_ids'][:, :, self.n_begin:self.n_end]
            seq_ids = torch.cat([seq_ids, explain_seq_ids], dim=2)

            seq_values = bg_data['seq_values'].unsqueeze(0)[:, :, self.c_begin:self.c_end]
            explain_seq_values = explain_data['seq_values'][:, :, self.n_begin:self.n_end]
            seq_values = torch.cat([seq_values, explain_seq_values], dim=2)
            mask = explain_data['seq_mask']
            ts = explain_data['seq_tss']

            # add a batch dim
            model_input = [seq_ids, seq_values, mask, ts]
            return model_input

    def make_inst_input(self, data):
        model_input = [
            data['seq_ids'],
            data['seq_values'],
            data['seq_mask'],
            data['seq_tss']
        ]
        return model_input


