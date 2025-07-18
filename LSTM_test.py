from skimage.metrics import mean_squared_error
from LSTM_model import LSTMModel
from MLP_model import MLPModel
import torch.nn as nn
import torch
import torch.optim as optim
from train_data import *
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime
from test_data import *
from train_test_plot import *
from utility import *
import os

USE_MULTI_GPU = True
# 设置默认的CUDA设备
torch.cuda.set_device(0)
# 初始化CUDA环境
torch.cuda.init()
# 检测机器是否有多张显卡
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 设置所有六张显卡的编号
    device_ids = ['0']  # 设置所有六张显卡的编号
else:
    MULTI_GPU = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(MULTI_GPU)
deviceCount = torch.cuda.device_count()
torch.cuda.set_device(device)
print(deviceCount)


def test(TModel, tf_loader, y_max, y_min, de_max, de_min):
    epoch_loss = 0
    y_pre = []
    y_true = []
    y_depth = []

    for x, y in tf_loader:
        with torch.no_grad():
            label = y[:, -model_pre_len:, -1].detach().reshape(1,
                                                               len(y[:, -model_pre_len:, -1]) * model_pre_len).squeeze()
            label = label * (y_max - y_min) + y_min
            label = label.numpy().tolist()
            y_true += label

            de = y[:, -model_pre_len:, 0].detach().reshape(1, len(y[:, -model_pre_len:, 0]) * model_pre_len).squeeze()
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

    print("NaNs in true:", np.isnan(true).sum())
    print("NaNs in pre:", np.isnan(pre).sum())

    r2 = r2_score(true, pre)
    mse = mean_squared_error(true, pre)

    mae = mean_absolute_error(true, pre)

    return r2, mse, mae, epoch_loss, true, pre, depth


test_loader, test_y_max, test_y_min, test_de_max, test_de_min = test_data_load(data_path_well_2)

model = LSTMModel().to(device)
model.load_state_dict(torch.load('./out_xj/fulldata/model/Model_xj.pkl', map_location=torch.device('cuda')))
criterion = nn.MSELoss()  # 忽略 占位符 索引为0.9
#optimizer = optim.SGD(model.parameters(), lr=model_tf_lr, momentum=0.99)
optimizer = optim.Adam(model.parameters(), lr=model_tf_lr, weight_decay=0.0001)  # volve 是0.001


def initiate():
    train_r2_size = []
    train_mse_size = []
    train_mae_size = []
    train_loss_size = []

    test_r2_size = []
    test_mse_size = []
    test_mae_size = []
    test_loss_size = []

    start = datetime.now()
    model.eval()

    test_r2, test_mse, test_mae, test_loss, true_test, pre_test, test_depth = test(model, test_loader,
                                                                                       test_y_max, test_y_min,
                                                                                       test_de_max, test_de_min)
    test_mse_size.append(test_mse)
    test_mae_size.append(test_mae)

    test_r2_size.append(test_r2)
    test_loss_size.append(test_loss)
    print('loss =', '{:.6f}'.format(test_loss),
          ' r2 =', '{:.6f}'.format(test_r2),
          ' mse =', '{:.6f}'.format(test_mse),
          ' mae =', '{:.6f}'.format(test_mae),
          'time = ', start)
    loss_acc_mse_mae_dict = {'test_loss': test_loss_size,
                             'test_r2': test_r2_size,
                             'test_mse': test_mse_size,
                             'test_mae': test_mae_size, }
    loss_acc_mse_mae = pd.DataFrame(loss_acc_mse_mae_dict)

    test_de = pd.DataFrame(test_depth, columns=['test_depth'])
    test_t = pd.DataFrame(true_test, columns=['test_true'])
    test_p = pd.DataFrame(pre_test, columns=['test_pre'])

    csv_test = pd.concat([test_de, test_t, test_p], axis=1)

    loss_acc_mse_mae.to_csv('./out_xj/fulldata/data/well_2/loss_acc_mse_mae_300.csv', sep=",", index=True)

    csv_test.to_csv('./out_xj/fulldata/data/well_2/rel_pre_test_300.csv', sep=",", index=True)

    # acc_loss_plot_two(loss_acc_mse_mae['train_loss'], loss_acc_mse_mae['test_loss'], 'loss',
    #                   './out_xj/png/loss.png')
    # acc_loss_plot_two(loss_acc_mse_mae['train_r2'], loss_acc_mse_mae['test_r2'], 'r2',
    #                   './out_xj/png/acc.png')
    # true_test_plot(csv_test['test_depth'], csv_test['test_true'], csv_test['test_pre'], 'test',
    #                './out_xj/png/pre_true_test.png')


initiate()
