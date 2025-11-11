import numpy as np
import pandas as pd
import os
import json

from tqdm import tqdm

if __name__ == '__main__':
    path = '../data/PS_K4_filtered.normalized_with_cpm.tsv'
    df = pd.read_csv(path, sep='\t', dtype={'chr': str})
    total_norm = (df['read_count'] / df['mappability']).sum()
    print(df.head()) # 指定dtype吗

    windows = 1024
    stride = 256 # sei_plant使用的是512

    # 先去拿到以0开始的染色体区间
    seg_path = '../data/mouse/mouse_chrom_segments_base_0.tsv'
    segs = pd.read_csv(seg_path, sep='\t', dtype={0: str, 1: int, 2: int})
    print(segs.head())

    l = segs.iloc[:, 0].unique() # 染色体名称的列表
    print(l)

    # 新开一个文件来存

    header = ["chr", "start", "end", "CPM_normalized_reads", "log2_CPM_normalized_reads", "log10_CPM_normalized_reads"]
    o = []

    # with open(output_path, 'w') as f:
    #     f.write('\t'.join(header) + '\n')

    # 有个问题，我其实我直接就能从实验文件来做
    # 考虑步长没有
    chunk_size = 4
    stride_size = 1 # 1个表示256bp
    for chr in l:
        s = chr[3:]
        sub_tsv = df[df['chr'] == s]
        # print(sub_tsv.head())
        for i in tqdm(range(0, len(sub_tsv) - chunk_size + 1, 1), desc='Processing {}'.format(chr)):
            # if len(o) >= 600:
            #     t = pd.DataFrame(o, columns=header)
            #     output_path1 = '../data/mouse/recomputed_sy_data_600.tsv'
            #     t.to_csv(output_path1, sep='\t', index=False)
            #     break
            chunk = sub_tsv.iloc[i:i+chunk_size]
            if chunk.shape[0] < chunk_size:
                break
            else:
                # 在这处理数据
                start = chunk.iloc[0, 1]
                end = chunk.iloc[3, 2]

                read = chunk['read_count'].sum() # 可能为0
                map = chunk['mappability'].sum()
                norm = None
                cmp = 0
                output_log2 = 0
                output_log10 = 0
                if map < 0.1:
                    cmp = None
                    output_log2 = None
                    output_log10 = None
                else:
                    norm = read / map
                    cmp = norm / total_norm * 1e6 # 算出来数太大了？, 是因为
                    output_log2 = np.log2(cmp + 1)
                    output_log10 = np.log10(cmp + 1)

                # 写到df里
                row = [s, start, end, cmp, output_log2, output_log10]
                o.append(row)

    t = pd.DataFrame(o, columns=header)
    output_path1 = '../data/mouse/recomputed_sy_data.tsv'
    t.to_csv(output_path1, sep='\t', index=False)










