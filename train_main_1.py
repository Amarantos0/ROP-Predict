import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error

from tf_model import TransAm
import torch.nn as nn
import torch
import torch.optim as optim
import train_data
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime
from pyplot_make_dataset import *
from utility import *

USE_MULTI_GPU = True
# 设置默认的CUDA设备
torch.cuda.set_device(0)

# 初始化CUDA环境
torch.cuda.init()

# # 检测机器是否有多张显卡
# if USE_MULTI_GPU and torch.cuda.device_count() > 1:
#     MULTI_GPU = True
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # 设置所有六张显卡的编号
#     device_ids = list(range(6))  # 设置所有六张显卡的编号
# else:
#     MULTI_GPU = False
# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# 检测机器是否有多张显卡
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置所有六张显卡的编号
    device_ids = ['0']  # 设置所有六张显卡的编号
else:
    MULTI_GPU = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_mask(seq_len, num_unmasked, num_masked):
    # 创建一个全为False的掩码矩阵
    mask = torch.zeros(seq_len, seq_len)
    # 对后 num_masked 列的未来数据进行掩盖
    for i in range(seq_len):
        mask[i, i + num_unmasked:] = float('-inf')  # 这里我们对未来时间步进行掩盖
    return mask


def train(TModel, loader):
    epoch_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        enc_inputs = X.permute([1, 0, 2])  # [seq,batch,feature_size]
        y = y.permute([1, 0, 2])

        # 创建掩盖矩阵
        attn_mask = create_mask(seq_len=enc_inputs.shape[0], num_unmasked=4, num_masked=8)
        attn_mask = attn_mask.to(device)

        key_padding_mask = torch.zeros(enc_inputs.shape[1], enc_inputs.shape[0], dtype=torch.float32)
        key_padding_mask = key_padding_mask.to(device)

        optimizer.zero_grad()
        output = TModel(enc_inputs, key_padding_mask, y, attn_mask)

        output = output[-pre_len:, :, :]
        y = y[-pre_len:, :, :]
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(TModel.parameters(), 0.10)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss


def test(TModel, tf_loader):
    epoch_loss = 0
    y_pre = []
    y_true = []
    true_size = []
    y_depth = []

    for x, y in tf_loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)

            enc_inputs = x.permute([1, 0, 2])  # [seq,batch,feature_size]
            y = y.permute([1, 0, 2])

            # 创建掩盖矩阵
            attn_mask = create_mask(seq_len=enc_inputs.shape[0], num_unmasked=4, num_masked=8)
            attn_mask = attn_mask.to(device)

            key_padding_mask = torch.zeros(enc_inputs.shape[1], enc_inputs.shape[0], dtype=torch.float32)
            key_padding_mask = key_padding_mask.to(device)

            output = TModel(enc_inputs, key_padding_mask, y, attn_mask)

            output = output[-pre_len:, :, :]
            y = y[-pre_len:, :, :]
            de = enc_inputs[-pre_len:, :, :][:, :, 0].unsqueeze(-1)

            loss = criterion(output, y)
            epoch_loss += loss.item()

            pres = output.detach().cpu().numpy()
            pres = pres.transpose(0, 1, 2)
            pres = np.squeeze(pres, axis=2)
            pres_ = extract_anti_diagonal_blocks(pres)
            pres_ = np.array(pres_)
            pres_ = pres_[:, np.newaxis]

            tru = y.detach().cpu().numpy()
            tru = tru.transpose(0, 1, 2)
            tru = np.squeeze(tru, axis=2)
            tru_ = extract_anti_diagonal_blocks(tru)
            tru_ = np.array(tru_)
            tru_ = tru_[:, np.newaxis]

            de = de.detach().cpu().numpy()
            de = de.transpose(0, 1, 2)
            de = np.squeeze(de, axis=2)
            de_ = extract_anti_diagonal_blocks(de)
            de_ = np.array(de_)
            de_ = de_[:, np.newaxis]

            y_pre.append(pres_)
            y_true.append(tru_)
            y_depth.append(de_)

            true_size.append(y)

    pre = np.concatenate(y_pre, axis=0)
    true = np.concatenate(y_true, axis=0)
    depth = np.concatenate(y_depth, axis=0)

    pre = np.nan_to_num(pre, nan=0.0)
    true = np.nan_to_num(true, nan=0.0)
    depth = np.nan_to_num(depth, nan=0.0)

    acc = r2_score(true, pre)
    mse = mean_squared_error(true, pre)
    mae = mean_absolute_error(true, pre)

    epoch_loss = np.array([epoch_loss])
    acc = np.array([acc])
    mse = np.array([mse])
    mae = np.array([mae])
    return acc, epoch_loss, true, pre, depth, mse, mae


print(MULTI_GPU)
deviceCount = torch.cuda.device_count()
torch.cuda.set_device(device)
print(deviceCount)
# if MULTI_GPU:
#     model = nn.DataParallel(model, device_ids=device_ids)
train_loader, train_feature_size, train_out_size = train_data.dataset()
model = TransAm(train_feature_size, train_out_size).to(device)
#
# model = lstm_model.lstm_uni_attention(input_size=11, hidden_size=256, num_layers=2,
#                                              pre_length=50, seq_length=50).to(device)
# model.load_state_dict(
#     torch.load('result/train_model_300_150/Model_volve.pkl', map_location=torch.device('cuda')))
criterion = nn.MSELoss()  # 忽略 占位符 索引为0.9
optimizer = optim.SGD(model.parameters(), lr=tf_lr, momentum=0.99)
# optimizer = optim.Adam(model.parameters(), lr=tf_lr, weight_decay=0.001)
# if MULTI_GPU:
#     optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
print(device)


def initiate():
    global train_de, train_t, train_p
    train_acc_size = []
    test_acc_size = []

    train_loss_size = []
    test_loss_size = []

    test_mse_size = []
    train_mse_size = []

    test_mae_size = []
    train_mae_size = []

    start = datetime.now()
    # model = train_model_single_step.train()
    for epoch in range(5000):
        model.train()
        train(model, train_loader)
        model.eval()

        train_acc, train_loss, true_train, pre_train, train_depth, train_mse, train_mae = test(model, train_loader)
        train_acc_size.append(train_acc)
        train_loss_size.append(train_loss)
        train_mse_size.append(train_mse)
        train_mae_size.append(train_mae)

        print('Epoch:', '%04d' % epoch, 'loss =', '{:.6f}'.format(train_loss.item()), ' acc =',
              '{:.6f}'.format(train_acc.item()),
              ' mse =', '{:.6f}'.format(train_mse.item()), ' mae =', '{:.6f}'.format(train_mae.item()), 'time = ',
              start)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'result/train_model_300_150/volve_well_10.pkl')

            train_de = pd.DataFrame(np.concatenate(train_depth, axis=0), columns=['train_depth'])
            train_t = pd.DataFrame(np.concatenate(true_train, axis=0), columns=['train_true'])
            train_p = pd.DataFrame(np.concatenate(pre_train, axis=0), columns=['train_pre'])

            csv_rel_pre = pd.concat([train_de, train_t, train_p], axis=1)
            csv_rel_pre.to_csv('./result/data_300_150/rel_pre/f_rel_pre_model+well_10.csv', sep=",", index=True)

            # 使用局部变量存储和拼接结果
            train_loss = pd.DataFrame(train_loss_size, columns=['train_loss'])
            train_acc = pd.DataFrame(train_acc_size, columns=['train_acc'])
            train_mse = pd.DataFrame(train_mse_size, columns=['train_mse'])
            train_mae = pd.DataFrame(train_mae_size, columns=['train_mae'])

            csv_train_loss_acc_mse_mae = pd.concat(
                [train_loss, train_acc, train_mae, train_mse], axis=1)
            csv_train_loss_acc_mse_mae.to_csv(
                './result/data_300_150/loss_acc/f_loss_acc_mse_mae_model+well_10.csv', sep=",", index=True)

        if (epoch + 1) % 10 == 0:
            train_loss_acc_plot(train_loss_size, 'Loss', 'loss',
                                './result/png/loss/loss_300_150/f_loss_model+well_10.png')
            train_loss_acc_plot(train_acc_size, 'Acc', 'acc',
                                './result/png/acc/acc_300_150/f_acc_model+well_10.png')
            rel_pre_plot(train_de, train_t, train_p, 'm/hr', 'train',
                         './result/png/rel&pre/rel&pre_300_150/f_train_model+well_10.png')


initiate()
