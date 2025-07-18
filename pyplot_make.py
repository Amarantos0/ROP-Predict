import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from utility import *
from matplotlib import font_manager as fm, rcParams

# fig = plt.figure()
# fig.set_size_inches(10, 4)  # 整个绘图区域的宽度10和高度4

# ax = fig.add_subplot(1, 2, 1)
font = {'weight': '400',
        'size': 15,
        }


#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 显示中文标签
# plt.rcParams['axes.unicode_minus'] = False


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def loss_acc_plot(train, test, label_y, title, path):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(train, label='Train', color='r')
        plt.plot(test, label='Test', color='b')
        plt.xlabel('Epoch')
        plt.ylabel(f'{label_y}')
        plt.title(f'train_test_{title}')
        plt.legend()
        plt.grid()
        ensure_dir_exists(os.path.dirname(path))
        plt.savefig(f'{path}')
        plt.close()
        print(f"Saved plot to {path}")
        # file = open('1.txt', 'a+')
        # file.write(path)
        # file.close()
    except Exception as e:
        print(f"Error saving plot {title}: {e}")


def rel_pre_plot(depth, rel_data, pre_data, label_y, title, path):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(depth, rel_data, label='Train', color='r')
        plt.plot(depth, pre_data, label='Test', color='b')
        plt.xlabel('Depth')
        plt.ylabel(f'{label_y}')
        plt.title(f'{title}_rel_pre')
        plt.grid()
        plt.savefig(f'{path}')
        plt.close()
        print(f"Saved plot to {path}")
    except Exception as e:
        print(f"Error saving0 plot {title}: {e}")


def rop_prediction_plt(true, pre, epoch, path):
    regressor = LinearRegression()
    regressor = regressor.fit(np.reshape(true, (-1, 1)), np.reshape(pre, (-1, 1)))
    print(regressor.coef_, regressor.intercept_)  # 打印拟合结果(参数)
    # 画出数据和拟合直线的图
    plt.scatter(true, pre)
    plt.plot(np.reshape(true, (-1, 1)), regressor.predict(np.reshape(true, (-1, 1))), 'r')
    plt.xlabel("actual value")
    plt.ylabel("predictive value")
    plt.title("Fitting results")
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    plt.close()


def distance_chart_plt(train_acc_size, test_acc_size, path):
    unit = ["MSELoss", "epochs"]
    plt.figure(figsize=(24, 8))
    plt.plot(train_acc_size)
    plt.plot(test_acc_size)
    # plt.plot(depth,col_one_pre)
    # plt.plot(depth,col_one_pre_attn)
    plt.legend(["train_acc", "test_acc"], prop=font)
    plt.ylabel(unit[0], font)
    plt.xlabel(unit[1], font)
    # plt.tick_params(labelsize=20)
    # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
    plt.savefig(os.path.join(path, 'acc.png'))
    plt.close()


def distance_chart_plt_loss(train_acc_size, test_acc_size, path):
    unit = ["MSELoss", "epochs"]
    plt.figure(figsize=(24, 8))
    plt.plot(train_acc_size)
    plt.plot(test_acc_size)
    # plt.plot(depth,col_one_pre)
    # plt.plot(depth,col_one_pre_attn)
    plt.legend(["train_loss", "test_loss"], prop=font)
    plt.ylabel(unit[0], font)
    plt.xlabel(unit[1], font)
    # plt.tick_params(labelsize=20)
    # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.close()


def line_chart_plt(true, pre, md, epoch, path):
    true = true.flatten()
    pre = pre.flatten()
    md = md.flatten()
    unit = ["ROP(m/hr)", "Depth(m)"]
    plt.figure(figsize=(24, 8))
    plt.plot(md, true)
    # plt.plot(depth,col_one_pre)
    # plt.plot(depth,col_one_pre_attn)
    plt.plot(md, pre)
    plt.legend(["real", "pre"], prop=font)
    plt.ylabel(unit[0], font)
    plt.xlabel(unit[1], font)
    plt.tick_params(labelsize=20)
    # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    plt.close()


def rel_error_plt(md, r, epoch, path):
    plt.scatter(r, md)
    # plt.plot(np.reshape(x, (-1, 1)), regressor.predict(np.reshape(x, (-1, 1))), 'r')
    plt.xlabel("relative_error")
    plt.ylabel("md")
    plt.title("Fitting results")
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    plt.close()
