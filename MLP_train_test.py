from datetime import datetime

import math
import warnings
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import TensorDataset, DataLoader
from pylab import mpl
import os
import torch
import pandas as pd

warnings.filterwarnings('ignore')
# 设置matplotlib的配置
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# volve
model_pre_len = 5
model_seq_len = 300
model_tf_lr = 0.0005  # 0.0005
model_batch = 32  # 128
model_feature_size = 5
model_d_model = 128  # 512
model_num_layers = 1  # 1
model_dropout = 0.4  # 0.01

USE_MULTI_GPU = True
# 设置默认的CUDA设备
torch.cuda.set_device(0)
# 初始化CUDA环境
torch.cuda.init()
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # 设置所有六张显卡的编号
    device_ids = ['0', '1', '2', '3', '4', '5', ]  # 设置所有六张显卡的编号
else:
    MULTI_GPU = False
    device_ids = ['0']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(MULTI_GPU)
deviceCount = torch.cuda.device_count()
torch.cuda.set_device(device)
print(deviceCount)
print(device)


class MLPModel(nn.Module):
    def __init__(self, feature_size=model_feature_size, hidden_size=512, dropout=0.4):
        super(MLPModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # 输出维度为1
        )

    def forward(self, src):
        # src: [batch_size, seq_len, feature_size]
        out = self.model(src)  # 输出: [batch_size, seq_len, 1]
        return out.squeeze(-1)  # squeeze掉最后一维，返回 [batch_size, seq_len]


def true_test_plot(depth, true_data, predicted_data, type_, path):
    plt.figure(figsize=(20, 6))
    plt.plot(depth, true_data, label='true_data', color='blue', linewidth=1)
    plt.plot(depth, predicted_data, label='test_data', color='green', linewidth=1)
    plt.ylabel("rop", fontsize=18)
    plt.xlabel('depth', fontsize=18)
    plt.title(f'{type_}:True rop and Predicted rop')
    path_ = f'{path}'
    plt.grid()
    plt.savefig(path_)
    # plt.show()


data_path_4 = './data/volve/S_F4.csv'
data_path_5 = './data/volve/S_F5.csv'
data_path_10 = './data/volve/S_F10.csv'
data_path_14 = './data/volve/S_F14.csv'
data_path_15 = './data/volve/MLP_volve_15.csv'
data_path_15A = './data/volve/S_F15A.csv'
data_path_15_no_reaction = '/home/user2/LiuProject/MYH/reaction_parameter/data_process_no_reaction/no_reaction_F_15A.csv'


def averages(matrix):  # 计算平均值
    matrix = np.array(matrix)
    row_count, col_count = matrix.shape
    max_diagonal = row_count + col_count - 1
    diagonals = np.zeros(max_diagonal)
    counts = np.zeros(max_diagonal, dtype=int)
    for i in range(row_count):
        for j in range(col_count):
            num = matrix[i, j]
            diagonal_index = i + j
            diagonals[diagonal_index] += num
            counts[diagonal_index] += 1
    averages = diagonals / counts
    return averages


# 定义保存路径和模型参数
model_pre_len_values = [1, 3, 4, 5, 7, 9, 11]
base_path = "./out_xj/test/300_{}"
model_path = "./out_xj/model/Model_xj.pkl"

# 创建汇总结果的字典
summary_results = {
    "model_pre_len": [],
    "R2": [],
    "mse": [],
    "mae": [],
}


# 定义测试函数
def initiate(model_pre_len):
    # 更新保存路径
    save_path = base_path.format(model_pre_len)
    os.makedirs(save_path, exist_ok=True)
    test_r2_size = []
    test_mse_size = []
    test_mae_size = []
    start = datetime.now()

    model.eval()

    test_r2, test_mse, test_mae, test_epoch_loss, test_true, test_pre, test_depth = test(model, data_dataloader, y_max,
                                                                                         y_min, de_max, de_min)

    test_mse_size.append(test_mse)
    test_mae_size.append(test_mae)

    test_r2_size.append(test_r2)
    print(' r2 =', '{:.6f}'.format(test_r2), ' mse =', '{:.6f}'.format(test_mse), ' mae =', '{:.6f}'.format(test_mae),
          'time = ', start)
    loss_acc_mse_mae_dict = {'test_r2': test_r2_size, 'test_mse': test_mse_size, 'test_mae': test_mae_size, }
    loss_acc_mse_mae = pd.DataFrame(loss_acc_mse_mae_dict)

    test_de = pd.DataFrame(test_depth, columns=['test_depth'])
    test_t = pd.DataFrame(test_true, columns=['test_true'])
    test_p = pd.DataFrame(test_pre, columns=['test_pre'])
    csv_test = pd.concat([test_de, test_t, test_p], axis=1)

    loss_acc_mse_mae.to_csv(os.path.join(save_path, "acc_mse_mae_.csv"), sep=",", index=True)
    csv_test.to_csv(os.path.join(save_path, "rel_pre_test_.csv"), sep=",", index=True)

    # 绘制测试结果图
    true_test_plot(csv_test['test_depth'], csv_test['test_true'], csv_test['test_pre'], 'test',
                   os.path.join(save_path, "xj1.png"))

    # 将结果汇总到字典中
    summary_results["model_pre_len"].append(model_pre_len)
    summary_results["R2"].append(test_r2)
    summary_results["mse"].append(test_mse)
    summary_results["mae"].append(test_mae)


# 主循环：运行不同的 model_pre_len
for model_pre_len in model_pre_len_values:
    def test(TModel, tf_loader, y_max, y_min, de_max, de_min):
        epoch_loss = 0
        y_pre = []
        y_true = []
        y_depth = []

        for x, y in tf_loader:
            with torch.no_grad():
                label = y[:, -model_pre_len:, -1].detach().reshape(1, len(y[:, -model_pre_len:,
                                                                          -1]) * model_pre_len).squeeze()
                label = label * (y_max - y_min) + y_min
                label = label.numpy().tolist()
                y_true += label

                de = y[:, -model_pre_len:, 0].detach().reshape(1,
                                                               len(y[:, -model_pre_len:, 0]) * model_pre_len).squeeze()
                de = de * (de_max - de_min) + de_min
                de = de.numpy().tolist()
                y_depth += de

                x, y = x.to(device), y.to(device)

                output = TModel(x)

                pre_out = output[:, -model_pre_len:]

                loss = criterion(pre_out, y[:, -model_pre_len:, -1])

                epoch_loss += loss.item()

                hat = pre_out.cpu().detach().reshape(1, len(y[:, -model_pre_len:, -1]) * model_pre_len).squeeze()
                hat = hat * (y_max - y_min) + y_min
                hat = hat.numpy().tolist()
                y_pre += hat

        label = np.array(y_true)
        predict = np.array(y_pre)
        dep = np.array(y_depth)

        seq_label = label.reshape(int(len(label) / model_pre_len), model_pre_len)
        seq_predict = predict.reshape(int(len(predict) / model_pre_len), model_pre_len)
        seq_depth = dep.reshape(int(len(dep) / model_pre_len), model_pre_len)

        true = np.concatenate((seq_label[:-1, 0], seq_label[-1, :]), axis=0)
        depth = np.concatenate((seq_depth[:-1, 0], seq_depth[-1, :]), axis=0)
        pre = averages(seq_predict)

        r2 = r2_score(true, pre)
        mse = mean_squared_error(true, pre)

        mae = mean_absolute_error(true, pre)

        return r2, mse, mae, epoch_loss, true, pre, depth


    def test_data_load(data_path):
        well_3 = pd.read_csv(data_path)
        data_all = well_3

        data = data_all.astype('float32')
        data.dropna(inplace=True)
        data = data.values

        data_ = torch.tensor(data[:len(data)])
        maxc, _ = data_.max(dim=0)
        minc, _ = data_.min(dim=0)
        y_max = maxc[-1]
        y_min = minc[-1]
        de_max = maxc[0]
        de_min = minc[0]
        data_ = (data_ - minc) / (maxc - minc)

        data_last_index = data_.shape[0] - model_seq_len

        data_X = []
        data_Y = []

        for i in range(0, data_last_index - model_seq_len - model_pre_len + 1):
            data_x = np.expand_dims(data_[i:i + model_seq_len], 0)
            data_y = np.expand_dims(data_[i + model_seq_len:i + model_seq_len + model_pre_len], 0)
            data_X.append(data_x)
            data_Y.append(data_y)

        data_X = np.concatenate(data_X, axis=0)
        data_Y = np.concatenate(data_Y, axis=0)

        process_data = torch.from_numpy(data_X).type(torch.float32)
        process_label = torch.from_numpy(data_Y).type(torch.float32)

        data_feature_size = process_data.shape[-1]

        dataset_train = TensorDataset(process_data, process_label)

        data_dataloader = DataLoader(dataset_train, batch_size=model_batch, shuffle=False)
        return data_dataloader, y_max, y_min, de_max, de_min


    # 加载数据
    data_dataloader, y_max, y_min, de_max, de_min = test_data_load(data_path_well_1_no)

    # 初始化模型
    # model = MLPModel().to(device)
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.MSELoss()

    #    plt.show()
    print(f"Running with model_pre_len = {model_pre_len}")
    initiate(model_pre_len)

# 将汇总结果保存到TXT文件
summary_df = pd.DataFrame(summary_results)
summary_df.set_index("model_pre_len", inplace=True)
summary_df.to_csv("./out/test/summary_results_.txt", sep="\t", index=True, float_format="%.4f")

print("All experiments completed and results saved.")
