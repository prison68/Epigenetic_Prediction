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
from scripts.evaluate import evaluate_model

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


if __name__ == '__main__':
    # 从文件里读取x，y
    x_path = '../data/mouse/input.fasta'
    y_path = '../data/mouse/recomputed_sy_data_600.tsv'

    # 切分数据集 现在先随便切，把模型先跑起来，正式跑的时候再按染色体来划分
    tag_path = '../models/tag.txt'
    x_all, labels, pos, tag_dict = load_and_preprocess_data(x_path, y_path, tag_path)

    x_train, y_train, x_val, y_val, x_test, y_test, _ = split_data_by_chromosome(
        x_all, labels, pos, tag_dict
    )
    batch_size = 64
    train_loader, val_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, batch_size=batch_size
    )
    # 训练
    sequence_length = 1024
    n_genomic_features = 1  # 这个参数是什么意思，先按实验数来定义
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
    training_classifier_regression(epoch, lr,
                                   model_prth,
                                   train_loader,
                                   val_loader,
                                   model,
                                   device,
                                   result_path,
                                   fold_num=1,
                                   early_stopping_patience=20)

    # 评估
    test_dataset = NucDataset(x=x_test, y=y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    prediction_dir = '../eval/test/'
    t = 'regression'
    # 还是按回归来跑的
    evaluate_model(model, test_loader, device, prediction_dir, type=t)



