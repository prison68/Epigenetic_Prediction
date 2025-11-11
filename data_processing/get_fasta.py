import glob
import os
import gzip
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm
import json

def save_str2fasta(chr, seq, path, description):
    records = SeqRecord(Seq(seq), id=chr, description=description)
    SeqIO.write(records, path, "fasta")

def processing_str2fasta_original(path, output_path):
    for dir in tqdm(os.listdir(path), desc="processing str2fasta"):
        if dir.endswith('.gz'):
            file_path = os.path.join(path, dir)

            with gzip.open(file_path, 'rb') as f:
                df = pd.read_csv(file_path, sep='\t', header=None)
                chr =df.iloc[0, 0].split('-')[0]
                seq = ''.join(df.iloc[:, 1].astype(str))

                # 将一条染色体的序列存储为一个fasta文件
                fasta_path = os.path.join(output_path, chr + '.fasta')
                save_str2fasta('chr'+ chr, seq, fasta_path, chr)


def processing_str2fasta(path, output_path):
    # 存每条染色体起始位点
    start={}
    for dir in tqdm(os.listdir(path), desc="processing str2fasta"):
        if dir.endswith('.gz'):
            file_path = os.path.join(path, dir)

            with gzip.open(file_path, 'rb') as f:
                df = pd.read_csv(file_path, sep='\t', header=None)
                chr =df.iloc[0, 0].split('-')[0]
                seq = ''.join(df.iloc[:, 1].astype(str))
                start['Chr' + chr] = int(df.iloc[0, 0].split('-')[1])

                # # 将一条染色体的序列存储为一个fasta文件
                # fasta_path = os.path.join(output_path, chr + '.fasta')
                # save_str2fasta('Chr' + chr, seq, fasta_path, chr) # 先用大写的
    # 直接把字典存到文件
    with open('../data/mouse.start.txt', 'w') as f:
        json.dump(start, f)


def merge_fastas(path, output_path):
    file_list = glob.glob(os.path.join(path, '*.fasta'))
    file_list.sort() # 排序没没排好，1过了是10了，先这样吧

    genome_records = []

    for file_path in tqdm(file_list,desc='making genome'):
        # 提取染色体标识（从文件名）
        record = SeqIO.read(file_path, 'fasta')
        genome_records.append(record)

    SeqIO.write(genome_records, output_path, "fasta")



if __name__=='__main__':
    path = 'D:/CodeProjectAll/DeepLearning/DATA/ED/Mus_musculus_chromosome_splits/chromosome_splits'
    output_path = '../data/fasta/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 将一条染色体的序列存储为一个fasta文件 已完成
    processing_str2fasta(path, output_path)

    # # 将每条染色体的fasta文件拼接成一个基因组fasta
    # output_path2 = '../data/mouse.fasta'
    # merge_fastas(output_path, output_path2)

