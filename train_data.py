import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utility import *


def single_data():
    # data_4 = pd.read_csv(data_path_4)
    # data_5 = pd.read_csv(data_path_5)
    # data_7 = pd.read_csv(data_path_7)
    # data_9 = pd.read_csv(data_path_9)
    # data_9A = pd.read_csv(data_path_9A)
    # data_10 = pd.read_csv(data_path_10)
    # data_12 = pd.read_csv(data_path_12)
    # data_14 = pd.read_csv(data_path_14)
    # data_15A = pd.read_csv(data_path_15A)
    # data_all = pd.concat([data_4, data_5, data_7, data_9, data_10, data_12, data_14, data_15A])
    # data_all = data_all.astype('float32')
    # x_hat = data_all.values
    # zero_size = []

    data_10 = pd.read_csv(data_path_10)
    data_14 = pd.read_csv(data_path_14)
    data_4 = pd.read_csv(data_path_4)
    data_all = pd.concat([data_4])
    data_all = data_all.iloc[::3, :]
    data_all = data_all.astype('float32')
    x_hat = data_all.values
    zero_size = []

    # well_4 = pd.read_csv(data_path_4)
    # well_4_half = well_4[:int(len(well_4)/2)]
    # data_all = well_4_half
    # data_all = data_all.astype('float32')
    # x_hat = data_all.values
    # zero_size = []

    # data_all = pd.read_csv(data_path_10)
    # data_all = data_all.astype('float32')
    # x_hat = data_all.values
    # zero_size = []

    # x_T 中第二个元素（可能是某种序列数据）中值为0的位置索引，并将这些索引值保存到 zero_size 列表中。
    x_T = x_hat.T
    for index, elem in enumerate(x_T):  # 同时返回索引和元素值
        if index == 1:
            for i, e in enumerate(elem):
                if e == 0:
                    zero_size.append(i)
        minVals = elem.min(0)
        maxVals = elem.max(0)
        # 当列的数据差距过大做归一化
        if maxVals - minVals > 10000:
            elem = noramlization(elem)
            x_T[index] = elem

    # 做倒置
    # x = np.flipud(x_hat)
    x = x_hat
    # X = data1.drop(columns=['Churn', 'customerID', 'TotalCharges'])
    # y = data1['Churn']
    y = x[:, -1]

    if len(y.shape) < 2:
        y = np.expand_dims(y, 1)
    x = np.nan_to_num(x, nan=0.0)
    return x, y  # [none,feature_size]  [none,feature_size]默认out_size为1


def data_load(seq_len):
    x, y = single_data()
    len = x.shape[0]
    data_last_index = len - seq_len
    X = []
    Y = []

    for i in range(0, data_last_index):
        # for i in range(0, data_last_index, interval):
        data_x = np.expand_dims(x[i:i + seq_len], 0)  # [1,seq,feature_size]
        data_y = np.expand_dims(y[i:i + seq_len], 0)  # [1,seq,out_size]
        # data_y=np.expand_dims(y[,0)   #[1,seq,out_size]
        X.append(data_x)
        Y.append(data_y)

    # del X[-interval:]
    # del Y[0:interval]
    data_x = np.concatenate(X, axis=0)
    data_y = np.concatenate(Y, axis=0)
    data = torch.from_numpy(data_x).type(torch.float32)
    label = torch.from_numpy(data_y).type(torch.float32)
    return data, label  # [num_data,seq,feature_size]  [num_data,seq] 默认out_size为1


def dataset():
    X, Y = data_load(seq_len)
    feature_size = X.shape[-1]
    out_size = Y.shape[-1]

    # 确保返回的 feature_size 是整数
    assert isinstance(feature_size, int), f"Expected feature_size to be int, got {type(feature_size)}"

    dataset_train = TensorDataset(X, Y)
    dataloader = DataLoader(dataset_train, batch_size=batch, shuffle=False)
    return dataloader, feature_size, out_size
