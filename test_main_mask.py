from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error
import torch.nn as nn
from datetime import datetime
from tf_model import TransAm
from pyplot_make_dataset import *
from utility import *
import test_data
import os
import torch

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


def create_mask(seq_len, num_unmasked, num_masked):
    # 创建一个全为False的掩码矩阵
    mask = torch.zeros(seq_len, seq_len)
    # 对后 num_masked 列的未来数据进行掩盖
    for i in range(seq_len):
        mask[i, i + num_unmasked:] = float('-inf')  # 这里我们对未来时间步进行掩盖
    return mask


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


# 检测设备数量
print(MULTI_GPU)
deviceCount = torch.cuda.device_count()
torch.cuda.set_device(device)
print(deviceCount)

# 加载测试数据
test_loader, test_feature_size, test_out_size = test_data.dataset()

# 加载预训练模型
model = TransAm(test_feature_size, test_out_size).to(device)
model.load_state_dict(
    torch.load('model/xj/mask_reaction_XJ_well_3.pkl', map_location=torch.device('cuda'), weights_only=True))

criterion = nn.MSELoss()


# 开始测试
def initiate_test():
    test_acc_size = []
    test_loss_size = []
    test_mse_size = []
    test_mae_size = []

    start = datetime.now()
    model.eval()

    for epoch in range(1):
        test_acc, test_loss, true_test, pre_test, test_depth, test_mse, test_mae = test(model, test_loader)
        test_acc_size.append(np.array(test_acc))
        test_loss_size.append(np.array(test_loss))
        test_mse_size.append(np.array(test_mse))
        test_mae_size.append(np.array(test_mae))

        print('TEST:', 'loss =', '{:.6f}'.format(test_loss.item()), ' acc =', '{:.6f}'.format(test_acc.item()),
              ' mse =', '{:.6f}'.format(test_mse.item()), ' mae =', '{:.6f}'.format(test_mae.item()), "time = ",
              datetime.now())

        test_de = pd.DataFrame(np.concatenate(test_depth, axis=0), columns=['test_depth'])
        test_t = pd.DataFrame(np.concatenate(true_test, axis=0), columns=['test_true'])
        test_p = pd.DataFrame(np.concatenate(pre_test, axis=0), columns=['test_pre'])

        csv_test_rel_pre = pd.concat([test_de, test_t, test_p], axis=1)
        # csv_test_rel_pre.to_csv('./result/data_XJ/mask_reaction/300/rel_pre/right_rel_pre_well_4_half.csv', sep=",", index=True)
        csv_test_rel_pre.to_csv('./result/lstm/volve/right_rel_pre_10.csv', sep=",", index=True)

        test_loss = pd.DataFrame(test_loss_size, columns=['test_loss'])
        test_acc = pd.DataFrame(test_acc_size, columns=['test_acc'])
        test_mse = pd.DataFrame(test_mse_size, columns=['test_mse'])
        test_mae = pd.DataFrame(test_mae_size, columns=['test_mae'])

        csv_test_loss_acc_mse_mae = pd.concat([test_loss, test_acc, test_mae, test_mse], axis=1)
        # csv_test_loss_acc_mse_mae.to_csv(
            # './result/data_XJ/mask_reaction/300/loss_acc/right_loss_acc_mse_mae_well_4_half.csv', sep=",", index=True)
        csv_test_loss_acc_mse_mae.to_csv(
            './result/lstm/volve/right_loss_acc_mse_mae_10.csv', sep=",", index=True)
        # test_loss_acc_plot(test_loss_size, 'Loss', 'loss',
        #                    './result/png/loss/loss_300_150/loss_model_well_2.png')
        # test_loss_acc_plot(test_acc_size, 'Acc', 'acc',
        #                    './result/png/acc/acc_300_150/acc_model_well_2.png')
        # rel_pre_plot(test_de, test_t, test_p, 'ROP m/hr', 'test',
        #              './result/data_XJ/mask_reaction/300/png/right_test_well_4_half.png')
        rel_pre_plot(test_de, test_t, test_p, 'ROP m/hr', 'test',
                     './result/lstm/volve/right_test_10.png')


initiate_test()
