import numpy as np
import pandas as pd

# from sklearn.metrics import mean_absolute_error

# 数据处理层
# 新疆录井

# # 202
data_path_4 = '/home/user2/LiuProject/MYH/new_tf/train/data_process/S_F4.csv'
data_path_5 = '/home/user2/LiuProject/MYH/new_tf/train/data_process/S_F5.csv'
data_path_7 = '/home/user2/LiuProject/MYH/new_tf/train/data_process/S_F7.csv'
data_path_9 = '/home/user2/LiuProject/MYH/new_tf/train/data_process/S_F9.csv'
data_path_9A = '/home/user2/LiuProject/MYH/new_tf/train/data_process/S_F9A.csv'
data_path_10 = '/home/user2/LiuProject/MYH/new_tf/train/data_process/S_F10.csv'
data_path_12 = '/home/user2/LiuProject/MYH/new_tf/train/data_process/S_F12.csv'
data_path_14 = '/home/user2/LiuProject/MYH/new_tf/train/data_process/S_F14.csv'
data_path_15A = '/home/user2/LiuProject/MYH/reaction_parameter/data_process_volve/S_F15A.csv'

# reaction parameters
batch = 128
test_batch = 128
pre_len = 200
seq_len = 300  # 400 xj scatter
interval = 1  # 20
tf_lr = 0.0005


# 做标准化归一化
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals) / ranges
    return normData


def dep_label(matrix):
    matrix = np.array(matrix)
    row_count, col_count = matrix.shape
    x=[]
    for i in range(row_count):
        if i==0 :
            x+=list(matrix[i,:])
        else:
            x+=list(matrix[i,-1:])
    return np.array(x)


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


#  预测重复值计算
def anti_diagonal_averages(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    result = []

    for diag in range(rows + cols - 1):
        sum_diag = 0
        count = 0
        for i in range(max(0, diag - cols + 1), min(rows, diag + 1)):
            j = diag - i
            sum_diag += matrix[i][j]
            count += 1
        result.append(sum_diag / count)

    return result


def extract_anti_diagonal_blocks(matrix):
    rows, cols = matrix.shape
    blocks = []

    # 分割所有的数据块 不足的补0
    sub_blocks = []
    # for start_row in range(0, rows, interval):
    for start_row in range(0, rows):
        for col in range(cols):
            block = []
            for i in range(interval):
                if start_row + i < rows:
                    block.append(matrix[start_row + i, col])
                else:
                    block.append(-999)  # 填充0
            sub_blocks.append((start_row // interval, col, block))

    # 获取所有反对角线的位置
    anti_diagonals = {}
    for r, c, block in sub_blocks:
        if (r + c) not in anti_diagonals:
            anti_diagonals[(r + c)] = []
        anti_diagonals[(r + c)].append(block)

    # 转换反对角线变成数据块
    for key in sorted(anti_diagonals.keys()):
        blocks.append(anti_diagonals[key])

    result = []
    for arr in blocks:
        if len(arr) == 1:
            result.extend([x for x in arr[0] if x != -999])
        else:
            num_elements = len(arr[0])
            sums = np.zeros(num_elements)
            counts = np.zeros(num_elements)
            for sub_arr in arr:
                for i, value in enumerate(sub_arr):
                    if value != -999:
                        sums[i] += value
                        counts[i] += 1

            for i in range(num_elements):
                if counts[i] > 0:
                    result.append(sums[i] / counts[i])
                else:
                    result.append(-999)

    return result


# 将数据变为有监督学习
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    # i: n_in, n_in-1, ..., 1
    # 代表t-n_in, ... ,t-1
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(data.columns[j] + '(t-%d)' % i) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(data.columns[j] + '%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [(data.columns[j] + '(t+%d)' % i) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

