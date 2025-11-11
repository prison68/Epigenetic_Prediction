import pandas as pd
import json
import os
import gzip
from tqdm import tqdm

def split(path):
    # 不存维字典，存tsv
    # 这里还是base-1， 存储的时候-1,不行，在main里-1
    # chrom_segments = {}
    path_tsv = '../data/mouse/mouse_chrom_segments_base_1.tsv'
    with open('../data/mouse.start.txt') as f:
        start_pos = json.load(f)

    for dir in tqdm(os.listdir(path), desc="processing split"):
        if dir.endswith('.gz'):
            file_path = os.path.join(path, dir)

            # l = []

            with gzip.open(file_path, 'rb') as f:
                df = pd.read_csv(file_path, sep='\t', header=None)
                chr = 'Chr' + df.iloc[0, 0].split('-')[0]

                start = start_pos[chr]

                for i in range(df.shape[0]):
                    temp1 = df.iloc[i, 0].split('-')
                    if i + 1 >= df.shape[0]:
                        with open(path_tsv, 'a', encoding='utf-8') as f:
                            f.write(chr + '\t' + str(start) + '\t' + str(temp1[2]) + '\n')
                            # l.append((start, temp1[2]))
                        break
                    else:
                        temp2 = df.iloc[i + 1, 0].split('-')
                        if int(temp1[2]) + 1 != int(temp2[1]):
                            with open(path_tsv, 'a', encoding='utf-8') as f:
                                f.write(chr + '\t' + str(start) + '\t' + str(temp1[2]) + '\n')
                                # l.append((start, temp1[2]))
                            start = temp2[1]

                # chrom_segments[chr] = l

    # with open('../data/mouse_chrom_segments.txt', 'w') as f:
    #     json.dump(chrom_segments, f)



if __name__ == "__main__":
    # path = 'D:/CodeProjectAll/DeepLearning/DATA/ED/Mus_musculus_chromosome_splits/chromosome_splits/'
    #
    # split(path)

    path_tsv = '../data/mouse/mouse_chrom_segments_base_1.tsv'
    segs = pd.read_csv(path_tsv, sep='\t', header=None, dtype={0 : str, 1 : int, 2 : int})
    print(segs.head())
    segs[1] -= 1

    path_tsv1 = '../data/mouse/mouse_chrom_segments_base_0.tsv'
    segs.to_csv(path_tsv1, sep='\t', index=False)


    # df = pd.read_csv(file_path, sep='\t', header=None)
    # print(df.head())
    #
    # chr = 'chr1'
    # with open('../data/mouse.start.txt') as f:
    #     start_pos = json.load(f)
    #
    # start = start_pos[chr]
    # chrom_segments = {}
    # l = []
    # for i in tqdm(range(df.shape[0]), desc='loading'):
    #     temp1 = df.iloc[i, 0].split('-')
    #     if i + 1 >= df.shape[0]:
    #         l.append((start, temp1[2]))
    #         break
    #     else:
    #         temp2 = df.iloc[i + 1, 0].split('-')
    #         if int(temp1[2]) + 1 != int(temp2[1]):
    #             l.append((start, temp1[2]))
    #             start = temp2[1]
    #
    # chrom_segments[chr] = l
    # with open('../data/mouse_chrom_segments.txt', 'w') as f:
    #     json.dump(chrom_segments, f)

