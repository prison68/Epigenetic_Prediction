import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader
from utils.data import *
from utils.preprocessing import *
from utils.preprocess import *
import itertools
import numpy as np
from collections import Counter
import random


def load_and_preprocess_data_notag(fasta_file):
    # 读取序列数据和标签文件  在preprocessing脚本里
    seqs, tags, pos = load_data_notag(fasta_file, label=True, strand=False, pos=True) # 只读了前600个

    # 数据预处理：序列数据转为one-hot形式，标签转为数字映射
    nuc_pre = NucPreprocess(seqs)
    X_all = nuc_pre.onehot_for_nuc()

    return X_all

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


# 这个函数处理数据
def load_and_preprocess_data(fasta_file, y_path, tag_file):
    # 读取序列数据和标签文件 先直接跑
    seqs, pos = load_data(fasta_file, label=False, strand=False, pos=True)

    # 加载标签文件，并创建标签字典
    with open(tag_file) as f:
        tag_list = f.readlines()

    tag_dict = {item.strip(): index for index, item in enumerate(tag_list)}
    feature_size = len(tag_dict)
    print("tag_dict:", feature_size)

    # 数据预处理：序列数据转为one-hot形式，标签转为数字映射
    nuc_pre = NucPreprocess(seqs[:2000000])
    X_all = nuc_pre.onehot_for_nuc()
    # 加载label并进行nan值和0值处理，全部封装到这里
    df = pd.read_csv(y_path, sep='\t')
    y_new = df.iloc[:2000000, -2].values
    # y_new = torch.tensor(df.iloc[:600, -2].values, dtype=torch.float)
    # 在这处理nan值, 先不要tensor
    y = smart_statistical_imputation(y_new)
    # 处理0值 一个很小的正值
    y = torch.tensor(simple_zero_handling(y), dtype=torch.float)

    return X_all, y, pos, tag_dict


# utils/data_utils.py
def split_data_by_chromosome_cross_val(X_all, labels, pos, tag_dict, num_folds=5):
    # 提取染色体号
    chromosome_numbers = []
    for item in pos:
        if item.startswith("Chr"):
            item = item[3:]  # 去掉前缀 "Chr"
        chrom_number = int(item.split(':')[0])
        chromosome_numbers.append(chrom_number)

    # 使用 Counter 来统计不同染色体的数量
    chromosome_count = Counter(chromosome_numbers)
    print(f"Chromosome count: {chromosome_count}")

    # 获取所有染色体
    chromosomes = list(chromosome_count.keys())

    # 存储每次的训练集、验证集和测试集组合
    fold_data = []

    # 获取所有可能的染色体组合：从5个染色体中选择1个为测试集，1个为验证集，剩余为训练集
    combinations = list(itertools.combinations(chromosomes, 2))

    # 遍历每一个组合
    for comb in combinations:
        test_chrom = comb[0]  # 选择一个染色体作为测试集
        val_chrom = comb[1]  # 选择一个染色体作为验证集

        # 剩下的染色体作为训练集
        train_chromosomes = [chrom for chrom in chromosomes if chrom not in [test_chrom, val_chrom]]

        # 划分训练集、验证集和测试集的索引
        testing_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom == test_chrom]
        validation_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom == val_chrom]
        training_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom in train_chromosomes]

        # 数据切分
        X_training = [X_all[i] for i in training_index]
        y_training = np.array([labels[i] for i in training_index])

        X_validation = [X_all[i] for i in validation_index]
        y_validation = np.array([labels[i] for i in validation_index])

        X_testing = [X_all[i] for i in testing_index]
        y_testing = np.array([labels[i] for i in testing_index])

        # 存储每次的训练集、验证集和测试集
        fold_data.append(
            (X_training, y_training, X_validation, y_validation, X_testing, y_testing, test_chrom, val_chrom,
             train_chromosomes))
    # 从fold_data中随机选择指定数量的组合（这里选择5对）
    random.seed(42)
    random_folds = random.sample(fold_data, num_folds)

    return random_folds, tag_dict


def split_data_by_chromosome(X_all, labels, pos, tag_dict):
    # 提取染色体号
    chromosome_numbers = []
    for item in pos:
        if item.startswith("Chr"):
            item = item[3:]  # 去掉前缀 "Chr"
        chrom_number = int(item.split('-')[0])
        chromosome_numbers.append(chrom_number)

    # 使用 Counter 来统计不同染色体的数量 这里count记录的是每个start出现的次数
    chromosome_count = Counter(chromosome_numbers)
    print(chromosome_count)

    # 定义测试集和验证集的染色体号
    test_chromosomes = [13]
    validation_chromosomes = [14]

    # 将染色体号转换为集合，提高查找效率
    test_chromosomes_set = set(test_chromosomes)
    validation_chromosomes_set = set(validation_chromosomes)

    # 根据染色体号划分数据集
    testing_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom in test_chromosomes_set]
    validation_index = [index for index, chrom in enumerate(chromosome_numbers) if chrom in validation_chromosomes_set]

    # 通过集合判断是否属于训练集
    validation_and_testing_index = set(testing_index + validation_index)
    training_index = [i for i in range(len(pos)) if i not in validation_and_testing_index]

    # 输出数据集长度
    print(f"Training set length: {len(training_index)}")
    print(f"Testing set length: {len(testing_index)}")
    print(f"Validation set length: {len(validation_index)}")

    # 数据切分
    X_training = [X_all[i] for i in training_index]
    y_training = [labels[i] for i in training_index]

    X_validation = [X_all[i] for i in validation_index]
    y_validation = [labels[i] for i in validation_index]

    X_testing = [X_all[i] for i in testing_index]
    y_testing = [labels[i] for i in testing_index]

    return X_training, y_training, X_validation, y_validation, X_testing, y_testing, tag_dict


# utils/data_utils.py
def create_data_loaders(X_training, y_training, X_validation, y_validation, batch_size=256):
    # 将数据转为 PyTorch 数据集，并定义批量大小
    train_set = NucDataset(x=X_training, y=y_training)
    val_set = NucDataset(x=X_validation, y=y_validation)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader


if __name__ == '__main__':
    X_all, labels, pos, tag_dict = load_and_preprocess_data(
        'D:\\CodeProjectAll\\DeepLearning\\epigenetic_prediction\\SeiPlant\\scripts\\bed\\arabidopsis_thaliana_1024_128.fa',
        'D:\\CodeProjectAll\\DeepLearning\\epigenetic_prediction\\SeiPlant\\models\\histone_modification_tag.txt')
    # X_all, labels, pos, tag_dict = load_and_preprocess_data_notag(
    #     'D:\\CodeProjectAll\\DeepLearning\\epigenetic_prediction\\SeiPlant\\scripts\\bed\\arabidopsis_thaliana_1024_128.fa')
