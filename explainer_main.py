import torch
from explainer_numerical.mask import Mask
from explainer_numerical.perturbation import GaussianBlur, FadeMovingAverage
from utils.run_utils import create_loader, make_model_input, load_checkpoint, init_model, compute_concat_score, \
    compute_random_score, load_all_case, compute_final_radio, compute_final_CE
from utils.oss_utils import get_bucket, upload_file_to_oss, upload_folder_to_oss
from utils.shap_utils import load_bg_data, choose_sample, sample2item, save_explain_result, load_bg_data_all, \
    save_input, save_config, save_cases_result
import config.data_config as data_config
import config.model_config as model_config
import config.run_config as run_config
import config.explain_config as explain_config
from alps.io.base import OdpsConf
from datetime import datetime
import os
import pickle as pkl
import warnings
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from explainer_categorical.shap_all import kenel_shap
from explainer_categorical.discretization import Discretization
from explainer_final.explainer import Mask_Fusion
from utils.losses import mse, log_loss_target, sort_loss, size_loss, feature_loss, feature_loss_after, grad_match, \
    reg_loss, feature_loss_BSE_after
from torch.optim import Adam, SGD
from pypai.utils import env_utils
import json
from baseline_all import baseline_contrast
import pickle as pkl
import time

warnings.filterwarnings("ignore")


def explain_all(model, case_index, oss_root_path, before_sample=None):
    logger = logging.getLogger('shap')
    handler = logging.FileHandler(f"./shap.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    use_gpu = explain_config.ENABLE_GPU
    model_type = explain_config.MODEL_TYPE
    # oss_ckpt_path = explain_config.OSS_PRE_PATH
    fusion_type = explain_config.FUSION_TYPE
    oss_result_path = oss_root_path + "/" + model_type + '_explainer_case_' + str(
        case_index) + "_" + datetime.now().strftime('%Y%m%d_%H_%M') + fusion_type
    num_samples = explain_config.SAMPLES_NUM
    field_end = explain_config.CATEGORICAL_FIELD_END
    field_start = explain_config.CATEGORICAL_FIELD_START
    disc_radio = explain_config.DISCRETIZATION_RADIO
    disc_lr = explain_config.DISCRETIZATION_LR
    save_to_oss = explain_config.SAVE_TO_OSS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c_begin = data_config.CATEGORICAL_FIELD_START
    c_end = data_config.CATEGORICAL_FIELD_END
    n_begin = data_config.NUMERICAL_FIELD_START
    n_end = data_config.NUMERICAL_FIELD_END
    value_index = data_config.VALUE_INDEX

    experiment_configs = {
        'explain_config': {k: v for k, v in vars(explain_config).items() if not k.startswith('__')}
    }

    t0 = time.time()
    # model = init_model(model_type,use_gpu)
    # model, ckpt_dict = load_checkpoint(model, oss_ckpt_path)

    if use_gpu:
        model = model.cuda()
    model.eval()
    # print("##### Load model checkpoint over #####")
    # get categorical mask
    bg_data = load_bg_data()
    # print("##### Load background data over #####")
    # print("##### compute categorical mask #####")
    for i in range(num_samples):
        logger.info("##### Sample {}: #####".format(i + 1))
        if before_sample == None:
            sample = choose_sample(model, 2048, use_gpu)
        else:
            sample = before_sample
        seq_feats_item = sample2item(sample)

    # print("-"*20+"baseline begin")
    # kernel_shap_all_score,kernel_shap_numerical_score=baseline_contrast(model,sample)
    # print("-"*20+"baseline ok")

    explainer_shap = kenel_shap(model, sample, bg_data, logger, use_gpu, c_begin, c_end, n_begin, n_end, )
    shap_values = explainer_shap.shap_values(sample, explain_config.SAMPLE_NUM_PER_INPUT)
    shap_values = shap_values.reshape(explain_config.SEQ_LEN, field_end - field_start)
    field_list = [i for i in range(field_end)]
    time_list = [i for i in range(explain_config.SEQ_LEN)]
    df1 = pd.DataFrame(np.transpose(shap_values), index=field_list, columns=time_list)

    mask_np = df1.to_numpy().transpose()
    # mask_np = np.maximum(-mask_np,mask_np)

    mask_np_final = mask_np
    # mask = torch.tensor(mask_np)
    # disc = Discretization(use_gpu,device)
    # disc.fit(mask, keep_ratio=disc_radio,learning_rate=disc_lr)
    # mask_np_final = disc.mask_tensor.detach().numpy()

    t1 = time.time()

    df2 = pd.DataFrame(mask_np_final.transpose())
    # print("##### get categorical mask #####")
    # get numerical  mask
    # print("##### compute numerical mask #####")
    numerical_start = data_config.NUMERICAL_FIELD_START
    numerical_end = data_config.NUMERICAL_FIELD_END
    seq_len = data_config.SEQ_LEN
    numerical_keep_radio = explain_config.KEEP_RADIO
    numerical_lr = explain_config.LEARNING_RATE
    numerical_epoch = explain_config.EPOCH

    model_input, labels = make_model_input(sample)
    if use_gpu:
        model_input = [f.cuda() for f in model_input]
        labels = labels.cuda()
    with torch.no_grad():
        Y = model(model_input)

    def f(x):
        with torch.no_grad():
            out = model(x)
        out = out[-1]
        return out

    pert = FadeMovingAverage(device)  # This is the perturbation operator
    mask_saliency = torch.zeros(size=(num_samples, seq_len, (numerical_end - numerical_start)), dtype=torch.float32,
                                device=device)  # This is the mask saliency map

    for k, x_input in enumerate(torch.unbind(model_input[1])):
        # print(f"Now working with sample {k + 1}/{num_samples}.")
        # Fit the mask:
        mask = Mask(pert, device, task="classification", verbose=False, deletion_mode=True)
        mask.fit(
            X=x_input[:, numerical_start:numerical_end],
            f=f,
            loss_function=log_loss_target,
            keep_ratio=numerical_keep_radio,
            target=Y[k],
            learning_rate=numerical_lr,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=10000,
            initial_mask_coeff=0.5,
            n_epoch=numerical_epoch,
            momentum=1.0,
            time_reg_factor=0,
            model_input=model_input,
            batch_index=k,
            n_begin=n_begin,
            n_end=n_end,
            value_index=value_index,
        )
        mask_saliency[k, :, :] = mask.mask_tensor

    mask_saliency_np = mask_saliency.clone().detach().cpu().numpy()
    numerical_field_list = [i for i in range((numerical_end - numerical_start))]
    time_list = [i for i in range(explain_config.SEQ_LEN)]
    df3 = pd.DataFrame(np.transpose(mask_saliency_np[0, :, :]), index=numerical_field_list, columns=time_list)
    # print("##### get numerical mask #####")
    t2 = time.time()

    model_input_save = [i.clone().detach().cpu().numpy() for i in model_input]
    if save_to_oss:
        save_explain_result(df1, 'explain_{}_categorical.csv'.format(1), oss_result_path)
        save_explain_result(df2, 'explain_{}_categorical_disc.csv'.format(1), oss_result_path)
        save_explain_result(seq_feats_item, 'sample_{}.csv'.format(1), oss_result_path)
        save_explain_result(df3, 'explain_{}_numerical.csv'.format(1), oss_result_path)
        save_input(model_input_save, "input_{}.pkl".format(1), oss_result_path)
        save_config(experiment_configs, "explain_config.json", oss_result_path)

    if save_to_oss:
        explainer_shap.save_log(oss_result_path)

    catergorical_time = t1 - t0
    numerical_time = t2 - t1
    return model, mask_np_final, mask_saliency_np[0, :, :], model_input, oss_result_path, sample[
        "seq_lengths"], catergorical_time, numerical_time


def mask_fusion(model, catergorical_mask, numerical_mask, instance, oss_result_path, sample_length):
    seq_len = data_config.SEQ_LEN
    field_num = data_config.ID_FIELD_NUM
    use_gpu = explain_config.ENABLE_GPU
    fusion_type = explain_config.FUSION_TYPE
    fusion_radio = explain_config.FUSION_RADIO
    c_begin = data_config.CATEGORICAL_FIELD_START
    c_end = data_config.CATEGORICAL_FIELD_END
    n_begin = data_config.NUMERICAL_FIELD_START
    n_end = data_config.NUMERICAL_FIELD_END
    hidden_dim = explain_config.HIDDEN_DIM
    epoch_num = explain_config.FINAL_EPOCH
    size_factor = explain_config.SIZE_LOSS_FACTOR
    sort_factor = explain_config.SORT_LOSS_FACTOR
    feature_factor = explain_config.FEATURE_LOSS_FACTOR
    save_to_oss = explain_config.SAVE_TO_OSS
    model_type = explain_config.MODEL_TYPE
    learning_rate = explain_config.FUSION_LR
    weight_decay = explain_config.WEIGTH_DECAY
    candidata_radio = explain_config.CANDIDATE_RADIO
    optimizer_type = explain_config.OPTIMIZER_TYPE
    use_bse_loss = explain_config.BSE_LOSS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bg_data = load_bg_data_all()

    mask_concat = torch.cat([catergorical_mask, numerical_mask], dim=1)

    if explain_config.CATEGORICAL_ZERO:
        catergorical_mask = torch.clamp(catergorical_mask, min=0.0)
    else:
        catergorical_mask = torch.abs(catergorical_mask)
    catergorical_mask = catergorical_mask / torch.max(catergorical_mask)

    mask = torch.cat([catergorical_mask, numerical_mask], dim=1)
    mask[sample_length:, :] = 0.0
    if use_gpu:
        model = model.cuda()
        mask_concat = mask_concat.cuda()
        mask = mask.cuda()
        instance = [f.cuda() for f in instance]

    initial_out, random_out = compute_random_score(model, mask, instance, bg_data, fusion_radio, c_end - c_begin,
                                                   n_end - n_begin)

    _, not_norm_concat_out = compute_concat_score(model, mask_concat, instance, bg_data, fusion_radio, c_end - c_begin,
                                                  n_end - n_begin, False)
    mask_record_before, norm_concat_out = compute_concat_score(model, mask, instance, bg_data, fusion_radio,
                                                               c_end - c_begin, n_end - n_begin, True)
    mask_record_before = mask_record_before.cpu()
    mask_fusion = Mask_Fusion(seq_len, field_num, fusion_type, c_begin, c_end, n_begin, n_end,
                              use_gpu, fusion_radio, hidden_dim, mask_record_before)
    # print("fusion_type is {}".format(fusion_type))

    if optimizer_type == "SGD":
        optimizer = SGD(mask_fusion.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        optimizer = Adam(mask_fusion.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = SGD(mask_fusion.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if use_gpu:
        model = model.cuda()
        mask_fusion = mask_fusion.cuda()
        mask = mask.cuda()
        instance = [f.cuda() for f in instance]
        catergorical_mask = catergorical_mask.cuda()
        numerical_mask = numerical_mask.cuda()
        for k in bg_data.keys():
            bg_data[k] = bg_data[k].cuda()

    after_score = []
    min_after_score = 1.0

    for i in range(epoch_num):
        mask_fusion.train()
        model.train()
        # print("initial mask sum:::",torch.sum(mask))
        mask.requires_grad = False
        mask_final = mask_fusion(mask)
        # print("mask_final.sum",torch.sum(mask_final))
        optimizer.zero_grad()
        if use_gpu:
            mask_final = mask_final.cuda()

        final_sort_loss = sort_loss(catergorical_mask, numerical_mask, mask_final,
                                    seq_len, c_end - c_begin, n_end - n_begin)
        final_size_loss = size_loss(mask_final, seq_len, field_num, fusion_radio, device)
        final_reg_loss = reg_loss(mask_final, mask)
        instance_copy = [f.clone() for f in instance]
        # final_feature_loss, initial_out, after_out, mask_final_sort_idx,initial_input_after = feature_loss(model, mask_final, instance_copy, bg_data, fusion_radio, c_end-c_begin, n_end-n_begin)
        if use_bse_loss:
            final_feature_loss, mask_grad_record, threshold_replace, threshold_candidata, after_out, after_out_less = feature_loss_BSE_after(
                model, mask_final, instance_copy, bg_data, fusion_radio, c_end - c_begin, n_end - n_begin,
                candidata_radio)
        else:
            final_feature_loss, mask_grad_record, threshold_replace, threshold_candidata, after_out, after_out_less = feature_loss_after(
                model, mask_final, instance_copy, bg_data, fusion_radio, c_end - c_begin, n_end - n_begin,
                candidata_radio)

        after_score.append(after_out.cpu())
        if after_out < min_after_score:
            mask_final_get = torch.ones(mask_final.shape)
            mask_final_get = mask_final.clone()

        mask_final_replace_grad, mask_final_candidata_grad = grad_match(final_feature_loss, mask_final,
                                                                        mask_grad_record, threshold_replace,
                                                                        threshold_candidata, after_out_less)
        mask_feature_grad = mask_final_replace_grad + mask_final_candidata_grad

        # loss = size_factor * final_size_loss + sort_factor * final_sort_loss + feature_factor * final_feature_loss
        # loss = size_factor * final_size_loss + feature_factor * final_feature_loss

        # loss  = feature_factor * final_feature_loss
        # print('epoch {} final_feature_loss:{}, final_size_loss:{}, final_sort_loss:{}, loss:{}'.format(
        #     epoch_num, final_feature_loss, final_size_loss, final_sort_loss, loss
        # ))
        # loss.backward()
        mask_size_grad = torch.autograd.grad(outputs=final_size_loss * 0.2, inputs=mask_final, only_inputs=True)[0]
        mask_reg_grad = torch.autograd.grad(outputs=final_reg_loss * 0.7, inputs=mask_final, only_inputs=True)[0]
        mask_feature_grad = mask_feature_grad * 0.001

        mask_final_grad = mask_size_grad + mask_feature_grad
        mask_final.backward(mask_final_grad)
        optimizer.step()

    # print("minimum score is",min(after_score))
    mask_fusion_out = min(after_score)
    field_list = [i for i in range(field_num)]
    time_list = [i for i in range(seq_len)]
    mask_final_np = mask_final.detach().cpu().numpy()
    mask_final_get_np = mask_final_get.detach().cpu().numpy()
    df1 = pd.DataFrame(np.transpose(mask_final_np), index=field_list, columns=time_list)
    df2 = pd.DataFrame(np.transpose(mask_final_get_np), index=field_list, columns=time_list)
    if save_to_oss:
        save_explain_result(df1, 'explain_{}_all.csv'.format(1), oss_result_path)
        save_explain_result(df2, 'explain_best_all.csv', oss_result_path)
    return initial_out, random_out, not_norm_concat_out, norm_concat_out, mask_fusion_out


def init_final_map():
    final_score = {}
    final_score['initial'] = []
    final_score['random'] = []
    final_score['concat'] = []
    final_score['mask_fusion'] = []
    return final_score


if __name__ == "__main__":
    oss_case_path = explain_config.OSS_CASE_PATH
    cases = load_all_case(oss_case_path)
    print("main mask is explainer_main")
    cases_start_index = explain_config.CASE_START_INDEX
    cases_end_index = explain_config.CASE_END_INDEX
    fusion_type = explain_config.FUSION_TYPE
    fusion_radio = explain_config.FUSION_RADIO
    final_score = init_final_map()
    sample_num = explain_config.SAMPLE_NUM_PER_INPUT
    if sample_num == 250000 and fusion_type == "unequal_scale":
        oss_root_path = 'sigir/explainer_gamble_{}_{}/'.format(fusion_type,
                                                               str(fusion_radio)) + 'cases_{}_to_{}_'.format(
            cases_start_index, cases_end_index) + datetime.now().strftime('%Y%m%d_%H_%M')
    else:
        oss_root_path = 'sigir/explainer_gamble_{}_{}_{}/'.format(fusion_type, str(fusion_radio),
                                                                  sample_num) + 'cases_{}_to_{}_'.format(
            cases_start_index, cases_end_index) + datetime.now().strftime('%Y%m%d_%H_%M')
    # init model
    oss_ckpt_path = explain_config.OSS_PRE_PATH
    use_gpu = explain_config.ENABLE_GPU
    model_type = explain_config.MODEL_TYPE
    model_initial = init_model(model_type, use_gpu)
    if use_gpu:
        model_initial = model_initial.cuda()
    model_initial, ckpt_dict = load_checkpoint(model_initial, oss_ckpt_path)
    time_categorical_all = 0.0
    time_numerical_all = 0.0
    time_combining_all = 0.0

    for case_index in range(cases_start_index, cases_end_index):
        print("gamble case {} begin".format(case_index))
        t0 = time.time()
        model, catergorical_mask, numerical_mask, instance, oss_result_path, sample_length, catergorical_time, numerical_time = explain_all(
            model_initial, case_index, oss_root_path, cases[case_index])
        t1 = time.time()
        # print("----"*10+"has get mask"+"----"*10)
        initial_out, random_out, not_norm_concat_out, norm_concat_out, mask_fusion_out = mask_fusion(model,
                                                                                                     torch.tensor(
                                                                                                         catergorical_mask,
                                                                                                         dtype=torch.float32),
                                                                                                     torch.tensor(
                                                                                                         numerical_mask,
                                                                                                         dtype=torch.float32),
                                                                                                     instance,
                                                                                                     oss_result_path,
                                                                                                     sample_length)
        t2 = time.time()

        time_categorical_all = time_categorical_all + catergorical_time
        time_numerical_all = time_numerical_all + numerical_time
        time_combining_all = time_combining_all + t2 - t1
        final_score['initial'].append(float(initial_out))
        final_score['random'].append(float(random_out))
        final_score['concat'].append(float(max(not_norm_concat_out, norm_concat_out)))
        final_score['mask_fusion'].append(float(mask_fusion_out))
        print("gamble case {} end".format(case_index))
        final_radio = compute_final_radio(final_score)
        save_cases_result(final_score, final_radio, "final_score.pkl", "final_dr.pkl", oss_root_path)
        # print(initial_out,random_out,not_norm_concat_out,norm_concat_out,mask_fusion_out,kernel_shap_all_score,kernel_shap_numerical_score)
    print("main::time_categorical_all::", time_categorical_all / (cases_end_index - cases_start_index))
    print("main::time_numerical_all::", time_numerical_all / (cases_end_index - cases_start_index))
    print("main::time_combining_all::", time_combining_all / (cases_end_index - cases_start_index))

    final_radio = compute_final_radio(final_score)
    # final_CE = compute_final_CE(final_score)
    print("DR::\n", final_radio)
    # print("CE::\n",final_CE)
    save_cases_result(final_score, final_radio, "final_score.pkl", "final_dr.pkl", oss_root_path)

