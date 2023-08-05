import os
import json
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from datetime import datetime
from model.DIFM_for_fp import DIFM_for_fp
from loader.fp_loader import FPDataset
import config.data_config as data_config
import config.model_config as model_config
import config.run_config as run_config
from utils.metrics import cal_auc_score, cal_f1_score, metrics, cal_auc_score_section
from utils.run_utils import make_model_input, create_loader, save_checkpoint, load_checkpoint
from utils.oss_utils import get_bucket, upload_file_to_oss
from alps.io.base import OdpsConf
from pypai.utils import env_utils
import warnings

warnings.filterwarnings("ignore")



def train():
    model_type = "DIFM_for_fp"
    embedding_dim = model_config.EMBEDDING_DIM
    vocabulary_size = model_config.VOCABULARY_SIZE_FP
    activation = model_config.ACTIVATION
    seq_len = model_config.SEQ_LENGTH
    num_features = data_config.FIELD_NUM
    num_events = data_config.SEQ_LEN
    use_vote = model_config.USE_VOTE
    use_bn = model_config.USE_BN
    tss_dim = model_config.TSS_DIM
    hidden_dim_1 = model_config.HIDDEN_DIM_1
    hidden_dim_2 = model_config.HIDDEN_DIM_2
    drop_out_1 = model_config.DROP_OUT_1
    drop_out_2 = model_config.DROP_OUT_2
    use_gpu = run_config.ENABLE_GPU
    epoch_num = run_config.EPOCH_NUM
    batch_size = run_config.BATCH_SIZE
    lr = run_config.LEARNING_RATE
    phase = run_config.PHASE
    save_to_oss = run_config.SAVE_TO_OSS
    is_pretrained = run_config.PRETRAIN
    weight_decay = run_config.WEIGTH_DECAY

    oss_ckpt_path = run_config.OSS_PRE_PATH
    label_weight = run_config.LABEL_WEIGHT

    drop_human_feats = run_config.DROP_HUMAN_FEATS
    drop_feat_list = run_config.DROP_LIST

    print("Experiment Info: FD: model:{}, max_epoch:{}, use gpu:{}".format(
        model_type, epoch_num, use_gpu
    ))

    # exp_id = model_name + '_' + datetime.now().strftime('%m%d-%H:%M')
    exp_id = model_type + '_dropfeats_0.8_2.5' + datetime.now().strftime('%Y%m%d_%H_%M_%S')
    local_path = 'tmp_results/' + exp_id
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    oss_path = exp_id



    train_loader = DataLoader(FPDataset(
        odps, train_table, cols, drop_human_feats, drop_feat_list), batch_size=batch_size, num_workers=15)
    val_loader = DataLoader(FPDataset(
        odps, val_table, cols, drop_human_feats, drop_feat_list), batch_size=batch_size, num_workers=15)
    test_loader = DataLoader(FPDataset(
        odps, test_table, cols, drop_human_feats, drop_feat_list), batch_size=batch_size, num_workers=15)
    print('FP Dataset Ready, useing colums:{}'.format(cols))

    if not drop_human_feats:
        model = DIFM_for_fp(embedding_dim, hidden_dim_1, tss_dim, vocabulary_size + 9 * 2 + 9, num_events, num_features,
                            activation, use_bn, use_vote, drop_out_1)
    else:
        model = DIFM_for_fp(embedding_dim, hidden_dim_1, tss_dim, vocabulary_size + 6 * 2 + 9, num_events, num_features,
                            activation, use_bn, use_vote, drop_out_1)

    if use_gpu:
        model = model.cuda()
    BCE_loss = torch.nn.BCELoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15], 0.1)

    best_auc = 0
    if is_pretrained or phase == 'test':
        model, info_dict = load_checkpoint(model, oss_ckpt_path)
        if 'best_auc' in info_dict.keys():
            best_auc = info_dict['best_auc']

    if phase == 'test':
        print("-" * 20, "Test start", "-" * 20)
        _ = valid(model, test_loader, use_gpu)

    elif phase == 'train':
        print("-" * 20, "Training start", "-" * 20)
        for epoch in range(epoch_num):
            model.train()
            y_trues = []
            y_preds = []
            y_scores = []
            total_loss = 0
            # t3 = 0
            for batch_data in train_loader:
                # batch_data = down_sample(batch_data, ratio=4.0)
                if batch_data is None:
                    continue

                model_input = [
                    batch_data['seq_mask'],
                    batch_data['seq_ids'],
                    batch_data['seq_values'],
                    batch_data['seq_tss'],
                    batch_data['pred_evt_ids'],
                    batch_data['pred_evt_values'],
                    batch_data['seq_lengths']
                ]
                labels = batch_data['labels']
                label_mask = torch.where(
                    labels == 0, torch.ones(labels.shape[0]) * label_weight[0],
                    torch.ones(labels.shape[0]) * label_weight[1])

                if use_gpu:
                    model_input = [f.cuda() for f in model_input]
                    labels = labels.cuda()
                    label_mask = label_mask.cuda()

                optimizer.zero_grad()
                out = model(model_input)
                loss = BCE_loss(out, labels)
                loss = torch.mean(label_mask * loss)
                loss.backward()
                optimizer.step()

                out = out.detach().cpu()
                y_scores.append(out)
                y_preds.append(torch.where(out > 0.5, torch.ones(out.shape), torch.zeros(out.shape)))
                y_trues.append(labels.cpu())
                total_loss += loss.detach().cpu() * labels.shape[0]

            y_scores = torch.cat(y_scores, dim=0)
            y_preds = torch.cat(y_preds, dim=0)
            y_trues = torch.cat(y_trues, dim=0)
            total_loss /= y_trues.shape[0]

            print('epoch {} trained samples:{}, best_auc:{}, loss:{}'.format(
                epoch, y_trues.shape[0], best_auc, total_loss
            ))

            if epoch % 1 == 0:
                valid_auc = valid(model, val_loader, use_gpu)
                if best_auc < valid_auc:
                    best_auc = valid_auc

                if save_to_oss:
                    save_checkpoint(model, local_path, oss_path,
                                    info_dict={
                                        'best_auc': best_auc,
                                        'epoch': epoch,
                                        'loss': total_loss
                                    },
                                    str_dict={
                                        'epoch': epoch,
                                        'auc': valid_auc
                                    })
            lr_scheduler.step()


def valid(model, loader, use_gpu):
    BCE_loss = torch.nn.BCELoss()
    model.eval()
    with torch.no_grad():
        y_trues = []
        y_preds = []
        y_scores = []
        total_loss = 0
        for batch_data in loader:
            model_input = [
                batch_data['seq_mask'],
                batch_data['seq_ids'],
                batch_data['seq_values'],
                batch_data['seq_tss'],
                batch_data['pred_evt_ids'],
                batch_data['pred_evt_values'],
                batch_data['seq_lengths']
            ]
            labels = batch_data['labels']

            if use_gpu:
                model_input = [f.cuda() for f in model_input]
                labels = labels.cuda()

            out = model(model_input)
            loss = BCE_loss(out, labels)

            out = out.detach().cpu()
            y_scores.append(out)
            y_preds.append(torch.where(out > 0.5, torch.ones(out.shape), torch.zeros(out.shape)))
            y_trues.append(labels.cpu())
            total_loss += loss.detach().cpu() * labels.shape[0]

        y_scores = torch.cat(y_scores, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        total_loss /= y_trues.shape[0]

        f1_score = cal_f1_score(y_trues, y_preds)
        auc_score = cal_auc_score(y_trues, y_scores)
        auc_score_section = cal_auc_score_section(y_trues, y_preds)

        acc = torch.sum(torch.eq(y_preds, y_trues)) / y_trues.shape[0]
        print("valid samples:", y_trues.shape)
        print('Validation: acc:{}, f1_score:{}, auc_score:{},auc_score_section:{}, loss:{}\n\n'.format(
            acc, f1_score, auc_score, auc_score_section, total_loss
        ))
        return auc_score


if __name__ == "__main__":
    train()