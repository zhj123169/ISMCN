import torch
from torch.optim import Adam
from loader.torch_loader import GambleDataset
from loader.odps_loader import OdpsDataset
from utils.run_utils import create_loader, make_model_input, save_checkpoint, load_checkpoint, down_sample, init_model
import config.data_config as data_config
import config.model_config as model_config
import config.run_config as run_config
from model.time2vec import time2vec
from model.DIFM import DIFM
from model.DIFM_seq import DIFM_seq
from utils.metrics import cal_auc_score, cal_f1_score, cal_auc_score_section
import time
from alps.io.base import OdpsConf
import json
from datetime import datetime
from utils.oss_utils import get_bucket, upload_file_to_oss, upload_folder_to_oss
from utils.pic_utils import line_chart
from utils.logger import log_key_hyperpara
import os
import warnings
from pypai.utils import env_utils

warnings.filterwarnings("ignore")


def train():
    t0 = time.time()
    max_epoch = run_config.EPOCH_NUM
    model_name = run_config.MODEL
    val_interval = run_config.VALID_INTERVAL
    learning_rate = run_config.LEARNING_RATE
    weight_decay = run_config.WEIGTH_DECAY
    use_gpu = run_config.ENABLE_GPU
    save_to_oss = run_config.SAVE_TO_OSS
    is_pretrained = run_config.SAVE_TO_OSS
    enable_sample = run_config.ENABLE_SAMPLE
    sample_radio = run_config.SAMPLE_RIDIO
    print("Experiment Info: model:{}, max_epoch:{}, valid interval:{}, use gpu:{}".format(
        model_name, max_epoch, val_interval, use_gpu
    ))

    exp_id = model_name + '_' + datetime.now().strftime('%Y%m%d_%H_%M_%S')
    local_path = 'tmp_results/' + exp_id
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    oss_path = exp_id

    experiment_configs = {
        'run_config': {k: v for k, v in vars(run_config).items() if not k.startswith('__')},
        'data_config': {k: v for k, v in vars(data_config).items() if not k.startswith('__')},
        'model_config': {k: v for k, v in vars(model_config).items() if not k.startswith('__')}
    }
    log_key_hyperpara(experiment_configs, 0)
    print(experiment_configs)
    with open(local_path + '/experiment_config.json', 'w') as f:
        json.dump(experiment_configs, f, indent=2)
    if save_to_oss:
        bucket = get_bucket()
        upload_file_to_oss(bucket, oss_path + '/experiment_config.json', local_path + '/experiment_config.json')

    model = init_model(model_name, use_gpu)

    data_table_train = data_config.ODPS_table_train
    cols = ['seq_ids', 'seq_values', 'seq_mask', 'user_labels']
    train_loader = create_loader(odps, "ant_prot_dev." + data_table_train, cols, batch_size=run_config.BATCH_SIZE)
    # loader = GambleDataset(data_table,data_config.SEQ_LEN,data_config.FIELD_NUM,batch_size=run_config.BATCH_SIZE)
    print('Gambling Dataset Ready, batch_size:{}'.format(run_config.BATCH_SIZE))

    if use_gpu:
        model = model.cuda()
    BCE_loss = torch.nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50], 0.5)

    best_auc = 0.0
    train_loss_list = []
    train_f1_list = []
    train_auc_list = []
    val_loss_list = []
    val_f1_list = []
    val_auc_list = []
    epoch_list = [(i + 1) for i in range(max_epoch)]
    print("-" * 20, "Training start", "-" * 20)
    for epoch in range(max_epoch):
        print("-" * 20, "{}th epoch".format(epoch + 1), "-" * 20)
        model.train()
        y_trues = []
        y_preds = []
        y_scores = []
        total_loss = 0
        for data in train_loader:
            if enable_sample:
                data = down_sample(data, ratio=sample_radio)
            if data is None:
                continue
            model_input, labels = make_model_input(data)

            if use_gpu:
                model_input = [f.cuda() for f in model_input]
                labels = labels.cuda()
            optimizer.zero_grad()
            out = model(model_input)
            loss = BCE_loss(out, labels)
            loss.backward()
            optimizer.step()

            out = out.detach().cpu()
            y_scores.append(out)
            y_preds.append(torch.where(out > 0.5, torch.ones(out.shape), torch.zeros(out.shape)))
            y_trues.append(labels)
            total_loss += loss.detach().cpu() * labels.shape[0]

        y_scores = torch.cat(y_scores, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        total_loss /= y_trues.shape[0]

        f1_score = cal_f1_score(y_trues, y_preds)
        auc_score = cal_auc_score(y_trues, y_scores)
        auc_score_section = cal_auc_score_section(y_trues, y_scores)
        train_loss_list.append(total_loss.item())
        train_f1_list.append(f1_score.item())
        train_auc_list.append(auc_score.item())
        print('epoch {} f1_score:{}, auc_score:{}, auc_score_section:{}, loss:{}'.format(
            epoch, f1_score, auc_score, auc_score_section, total_loss
        ))

        if epoch % val_interval == 0:
            val_loss, val_f1, val_auc = valid(model, use_gpu)
            val_loss_list.append(val_loss.item())
            val_f1_list.append(val_f1.item())
            val_auc_list.append(val_auc.item())
            if val_auc > best_auc:
                best_auc = val_auc
                if save_to_oss:
                    save_checkpoint(model, local_path, oss_path,
                                    info_dict={
                                        'best_auc': best_auc,
                                        'epoch': epoch,
                                        'f1_score': f1_score,
                                        'loss': total_loss
                                    },
                                    str_dict={
                                        'epoch': epoch,
                                        'auc': best_auc
                                    })
    print("best auc is {}".format(best_auc))

    ###todo

    picture_list_x = [epoch_list, epoch_list, epoch_list, epoch_list, epoch_list, epoch_list]
    picture_list_y = [train_loss_list, train_f1_list, train_auc_list, val_loss_list, val_f1_list, val_auc_list]

    picture_list_name = ["train_loss", "train_f1", "train_auc", "val_loss", "val_f1", "val_auc"]
    picture_path = local_path + "/pictures"
    if not os.path.exists(picture_path):
        os.makedirs(picture_path)
    line_chart(picture_list_x, picture_list_y, picture_path, picture_list_name)
    if save_to_oss:
        bucket = get_bucket()
        upload_folder_to_oss(bucket, oss_path, picture_path)


def valid(model, use_gpu):
    print("___________vaild__________")
    data_table_val = data_config.ODPS_table_val
    cols = ['seq_ids', 'seq_values', 'seq_mask', 'user_labels']
    val_loader = create_loader(odps, "ant_prot_dev." + data_table_val, cols, batch_size=run_config.BATCH_SIZE)
    BCE_loss = torch.nn.BCELoss()
    y_trues = []
    y_preds = []
    y_scores = []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for data in val_loader:

            model_input, labels = make_model_input(data)
            if use_gpu:
                model_input = [f.cuda() for f in model_input]
                labels = labels.cuda()

            out = model(model_input)
            loss = BCE_loss(out, labels)

            out = out.detach().cpu()
            y_scores.append(out)
            y_preds.append(torch.where(out > 0.5, torch.ones(out.shape), torch.zeros(out.shape)))
            y_trues.append(labels)
            total_loss += loss.detach().cpu() * labels.shape[0]

        y_scores = torch.cat(y_scores, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        total_loss /= y_trues.shape[0]

        f1_score = cal_f1_score(y_trues, y_preds)
        auc_score = cal_auc_score(y_trues, y_scores)
        auc_score_section = cal_auc_score_section(y_trues, y_scores)
        print('Validation: f1_score:{}, auc_score:{}, auc_score_section:{}, loss:{}\n\n'.format(
            f1_score, auc_score, auc_score_section, total_loss
        ))
        return total_loss, f1_score, auc_score


if __name__ == "__main__":
    train()