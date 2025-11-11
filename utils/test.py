import os
import torch.optim as optim
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils.data_utils import (
    load_and_preprocess_data,
    split_data_by_chromosome,
    create_data_loaders,
    load_and_preprocess_data_notag
)
from utils.data import NucDataset
from models.model_architectures.sei_model import Sei

def training_classifier_regression(n_epoch, lr, model_dir, train, valid, model, device, result_path, fold_num, early_stopping_patience=20):
    # 返回参数的个数，总的和训练的
    total = sum(p.numel() for p in model.parameters()) # 80亿参数，这么多
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nStart training, parameters total: {}, trainable: {}\n'.format(total, trainable))
    model.train()  # 将 models 的模式改为 train

    criterion = nn.MSELoss()  # 适用于多标签分类
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss, best_loss = 0, 1e5  # 初始化损失值
    no_improve_counter = 0  # 用于计数验证(vaild)损失没有改善的epoch数

    # 用于存储每个 epoch 的损失
    train_losses = []
    valid_losses = []

    # 训练过程
    for epoch in range(n_epoch):
        total_loss = 0
        # Training phase
        model.train()  # 设置为训练模式
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            if torch.isnan(labels).any():
                print('1111111')
            optimizer.zero_grad()

            inputs = inputs.permute(0, 2, 1)  # For sei models
            outputs = model(inputs)
            outputs = outputs.squeeze()  # 去掉外层的 dimension # 1,要把21907改成1，如果用四种实验就是4
            loss = criterion(outputs, labels)  # 计算训练损失 nan值
            if torch.isnan(loss).any():
                print('22222')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # print('[ Epoch{}: {}/{} ] loss:{:.3f} '.format(epoch + 1, i + 1, t_batch, loss.item()), end='\r')

        train_loss = total_loss / t_batch
        train_losses.append(train_loss)  # 记录训练损失
        print('\nTrain | Loss:{:.5f}'.format(train_loss))

        # Validation phase
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            total_loss = 0
            all_labels = []
            all_probs = []

            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                inputs = inputs.permute(0, 2, 1)  # For sei models
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # 存储真实标签和预测概率
                all_labels.append(labels.cpu().numpy().ravel() )
                all_probs.append(outputs.cpu().numpy().ravel() )  # **转换为概率**


            valid_loss = total_loss / v_batch
            valid_losses.append(valid_loss)  # 记录验证损失
            print("Valid | Loss:{:.5f} ".format(valid_loss))

            # # **计算 AUC**
            # all_labels = np.concatenate(all_labels)
            # all_probs = np.concatenate(all_probs)

            # auroc, auprc = plot_auroc_auprc(all_labels, all_probs)
            # print(f"Valid AUROC: {auroc:.4f}")
            # print(f"Valid AUPRC: {auprc:.4f}")

            # Early stopping check
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), model_dir)  # 保存模型参数
                print('Saving models with loss {:.3f}'.format(valid_loss))
                no_improve_counter = 0  # Reset counter if validation loss improves
            else:
                no_improve_counter += 1  # Increment counter if no improvement

            # Check if early stopping is triggered
            if no_improve_counter >= early_stopping_patience:
                print("Early stopping due to no improvement in validation loss")
                break  # Stop training

        print('-----------------------------------------------')

    epochs_trained = len(train_losses)

    # **绘制损失曲线**
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs_trained + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epochs_trained + 1), valid_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 保存损失曲线
    loss_filename = os.path.join(result_path, f'fold_num_{fold_num}_training_validation_loss_plot.png')
    plt.savefig(loss_filename, dpi=300)  # 高质量保存
    print(f"Loss plot saved as {loss_filename}")

    plt.close('all')


def smart_statistical_imputation(y):
    """针对0-0.02范围小值数据的智能填充"""
    y_imputed = y.copy()
    nan_mask = np.isnan(y)

    if np.sum(nan_mask) == 0:
        return y

    # 分析非NaN值的分布 取反
    valid_values = y[~nan_mask]

    if len(valid_values) == 0:
        # 如果全部是NaN，使用0填充
        y_imputed[nan_mask] = 0.0
        return y_imputed

    # 对于小值数据，使用分位数而不是均值
    # 25%分位数对异常值不敏感，能更好地代表数据的"典型"小值。
    fill_value = np.percentile(valid_values, 25)  # 25%分位数，避免异常值影响

    # 如果分位数还是太大，使用更保守的值
    if fill_value > 0.01:  # 如果大于0.01，使用更小的值
        fill_value = np.percentile(valid_values, 10)

    y_imputed[nan_mask] = fill_value

    return y_imputed
def simple_zero_handling(y):
    """
    最简单的0值处理方法
    适用于大多数组蛋白修饰数据场景
    """
    y_processed = y.copy()

    # 统计0值信息
    zero_mask = y == 0
    zero_ratio = np.mean(zero_mask)
    print(f"0值比例: {zero_ratio:.3f}")

    if zero_ratio < 0.1:
        # 少量0值：替换为很小的正数
        non_zero_values = y[~zero_mask]
        if len(non_zero_values) > 0:
            # 使用非0值的最小值的一半
            fill_value = np.min(non_zero_values) * 0.5
        else:
            fill_value = 1e-6  # 极端情况

        y_processed[zero_mask] = fill_value
        print(f"将0值替换为: {fill_value:.6f}")

    else:
        # 较多0值：保留0值，但确保模型能正确处理
        print("0值较多，建议在模型层面处理（如使用两阶段模型）")
        # 这里可以添加小扰动，避免数值问题， 替换成1e-8
        y_processed = np.where(y_processed == 0, 1e-8, y_processed)

    return y_processed



if __name__ == '__main__':
    # 从文件里读取x，y
    x_path = '../data/mouse/input.fasta'
    y_path = '../data/mouse/recomputed_sy_data_600.tsv'

    x_new = load_and_preprocess_data_notag(x_path)
    df = pd.read_csv(y_path, sep='\t')
    y_new = df.iloc[:600, -2].values
    # y_new = torch.tensor(df.iloc[:600, -2].values, dtype=torch.float)
    # 在这处理nan值, 先不要tensor
    y = smart_statistical_imputation(y_new)
    # 处理0值 一个很小的正值
    y = torch.tensor(simple_zero_handling(y), dtype=torch.float)


    # 切分数据集 现在先随便切，把模型先跑起来，正式跑的时候再按染色体来划分
    # x_train, y_train, x_val, y_val, x_test, y_test, _ = split_data_by_chromosome(
    #     x_all, labels, pos, tag_dict
    # )
    all_dataset = NucDataset(x_new, y)
    train_size = int(0.8 * len(all_dataset))
    val_size = (len(all_dataset) - train_size) // 2
    test_size = len(all_dataset) - train_size - val_size

    batch_size = 64
    # 它将一个数据集随机分成多个互不重叠的子集
    #  [train_size, val_size, test_size] 每个子集的大小列表
    train_set, val_set, test_set = torch.utils.data.random_split(
        all_dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    # 跑验证集卡住是因为batch_size大了，显存不足
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    sequence_length = 1024
    n_genomic_features = 1 # 这个参数是什么意思，先按实验数来定义
    model = Sei(sequence_length=sequence_length, n_genomic_features=n_genomic_features)

    epoch = 1
    lr = 1e-4
    device = 'cuda:0'
    result_path = '../results/test/'
    model_prth = '../model_pth/test/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_prth):
        os.makedirs(model_prth)
    model = model.to(device, dtype=torch.float)
    training_classifier_regression(epoch, lr, model_prth, train_loader, val_loader, model, device, result_path, fold_num = 1, early_stopping_patience=20)





    for batch in train_loader:
        print('0000')


    # # 输入的应该都是张量才对，我应该多搞一些，然后用batch size
    # x = torch.randint(0, 1, (1000, 4, 1024))
    # y = torch.randint(0, 1, (1000, 7))
    # # 一共输入了1000条数据，每个数据会会对应1个值，但是这里是7个任务所以对于每一条数据输出了7个值
    #
    # # x_train, x_test, y_train, y_test = split_data_by_chromosome(x, y) # y的格式应该不对，直接改这个函数的内容，所以先不分数据集
    #
    # train_set = NucDataset(x=x, y=y)
    # train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    #
    # for batch in train_loader:
    #     c = 0
    #     print('0000')


