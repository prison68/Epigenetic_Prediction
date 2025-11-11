import json
import heapq
import random
from Bio import SeqIO
import collections
import numpy as np
import sys

Contig = collections.namedtuple('Contig', ['chr', 'start', 'end'])
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])


def split_contigs(genome_file):
    """
    :param genome_file:
    :return: 读取每条染色体的长度
    """
    # 读取每条染色体的起始位置
    with open('../data/mouse.start.txt') as f:
        start_pos = json.load(f)

    chrom_segments = {}

    records = SeqIO.index(genome_file, 'fasta')
    for chrom in records:
        length = len(records[chrom])
        chrom_segments[chrom] = [(0 + int(start_pos[chrom]), length + int(start_pos[chrom]))]  # 有问题，实际的DNA起始不是从0开始的

    return chrom_segments


def break_large_contigs(contigs, break_t, verbose=False):
    """Break large contigs in half until all contigs are under
       the size threshold."""

    # initialize a heapq of contigs and lengths
    contig_heapq = []
    for ctg in contigs:
        ctg_len = int(ctg.end) - int(ctg.start)
        heapq.heappush(contig_heapq, (-ctg_len, ctg))

    ctg_len = break_t + 1
    while ctg_len > break_t:

        # pop largest contig
        ctg_nlen, ctg = heapq.heappop(contig_heapq)
        ctg_len = -ctg_nlen

        # if too large
        if ctg_len > break_t:
            if verbose:
                print('Breaking %s:%d-%d (%d nt)' % (ctg.chr, ctg.start, ctg.end, ctg_len))

            # break in two
            ctg_mid = int(ctg.start) + int(ctg_len) // 2

            try:
                ctg_left = Contig(ctg.genome, ctg.chr, ctg.start, ctg_mid)
                ctg_right = Contig(ctg.genome, ctg.chr, ctg_mid, ctg.end)
            except AttributeError:
                ctg_left = Contig(ctg.chr, ctg.start, ctg_mid)
                ctg_right = Contig(ctg.chr, ctg_mid, ctg.end)

            # add left
            ctg_left_len = int(ctg_left.end) - int(ctg_left.start)
            heapq.heappush(contig_heapq, (-ctg_left_len, ctg_left))

            # add right
            ctg_right_len = int(ctg_right.end) - int(ctg_right.start)
            heapq.heappush(contig_heapq, (-ctg_right_len, ctg_right))

    # return to list
    contigs = [len_ctg[1] for len_ctg in contig_heapq]

    return contigs


def divide_contigs_pct(contigs, test_pct, valid_pct, pct_abstain=0.2):
    """Divide list of contigs into train/valid/test lists,
       aiming for the specified nucleotide percentages."""

    # sort contigs descending by length
    length_contigs = [(int(ctg.end) - int(ctg.start), ctg) for ctg in contigs]
    length_contigs.sort(reverse=True)

    # compute total nucleotides
    total_nt = sum([lc[0] for lc in length_contigs])

    # compute aimed train/valid/test nucleotides
    test_nt_aim = test_pct * total_nt
    valid_nt_aim = valid_pct * total_nt
    train_nt_aim = total_nt - valid_nt_aim - test_nt_aim

    # initialize current train/valid/test nucleotides
    train_nt = 0
    valid_nt = 0
    test_nt = 0

    # initialize train/valid/test contig lists
    train_contigs = []
    valid_contigs = []
    test_contigs = []

    # process contigs
    for ctg_len, ctg in length_contigs:

        # compute gap between current and aim
        test_nt_gap = max(0, test_nt_aim - test_nt)
        valid_nt_gap = max(0, valid_nt_aim - valid_nt)
        train_nt_gap = max(1, train_nt_aim - train_nt)

        # skip if too large
        if ctg_len > pct_abstain * test_nt_gap:
            test_nt_gap = 0
        if ctg_len > pct_abstain * valid_nt_gap:
            valid_nt_gap = 0

        # compute remaining %
        gap_sum = train_nt_gap + valid_nt_gap + test_nt_gap
        test_pct_gap = test_nt_gap / gap_sum
        valid_pct_gap = valid_nt_gap / gap_sum
        train_pct_gap = train_nt_gap / gap_sum

        # sample train/valid/test
        ri = np.random.choice(range(3), 1, p=[train_pct_gap, valid_pct_gap, test_pct_gap])[0]
        if ri == 0:
            train_contigs.append(ctg)
            train_nt += ctg_len
        elif ri == 1:
            valid_contigs.append(ctg)
            valid_nt += ctg_len
        elif ri == 2:
            test_contigs.append(ctg)
            test_nt += ctg_len
        else:
            print('TVT random number beyond 0,1,2', file=sys.stderr)
            exit(1)

    print('Contigs divided into')
    print(' Train: %5d contigs, %10d nt (%.4f)' % \
          (len(train_contigs), train_nt, train_nt / total_nt))
    print(' Valid: %5d contigs, %10d nt (%.4f)' % \
          (len(valid_contigs), valid_nt, valid_nt / total_nt))
    print(' Test:  %5d contigs, %10d nt (%.4f)' % \
          (len(test_contigs), test_nt, test_nt / total_nt))

    return [train_contigs, valid_contigs, test_contigs]


def rejoin_large_contigs(contigs):
    """ Rejoin large contigs that were broken up before alignment comparison."""

    # split list by chromosome
    chr_contigs = {}
    for ctg in contigs:
        chr_contigs.setdefault(ctg.chr, []).append(ctg)

    contigs = []
    for chrm in chr_contigs:
        # sort within chromosome
        chr_contigs[chrm].sort(key=lambda x: int(x.start))

        ctg_ongoing = chr_contigs[chrm][0]
        for i in range(1, len(chr_contigs[chrm])):
            ctg_this = chr_contigs[chrm][i]
            if ctg_ongoing.end == ctg_this.start:
                # join
                # ctg_ongoing.end = ctg_this.end
                ctg_ongoing = ctg_ongoing._replace(end=ctg_this.end)
            else:
                # conclude ongoing
                contigs.append(ctg_ongoing)

                # move to next
                ctg_ongoing = ctg_this

        # conclude final
        contigs.append(ctg_ongoing)

    return contigs


def contig_sequences(contigs, seq_length, stride, snap=1, label=None):
    ''' Break up a list of Contig's into a list of ModelSeq's. '''
    mseqs = []
    for ctg in contigs:
        seq_start = int(np.ceil(int(ctg.start) / snap) * snap)
        seq_end = seq_start + seq_length

        while seq_end <= int(ctg.end):  # 大于的直接丢掉了，是的
            # record sequence
            mseqs.append(ModelSeq(ctg.chr, seq_start, seq_end, label))

            # update
            seq_start += stride
            seq_end += stride

    return mseqs

def write_seqs_bed(bed_file, seqs, labels=False):
  '''Write sequences to BED file.'''
  bed_out = open(bed_file, 'w')
  for i in range(len(seqs)):
    line = '%s\t%d\t%d' % (seqs[i].chr, seqs[i].start, seqs[i].end)
    if labels:
      line += '\t%s' % seqs[i].label
    print(line, file=bed_out)
  bed_out.close()

if __name__ == '__main__':
    out_dir = '../data/mouse'
    genome_path = '../data/mouse.fasta'
    # chrom_segments直接从文件里读，然后再加载成对象就可以了

    with open('../data/mouse_chrom_segments.txt') as f:
        chrom_segments = json.load(f)

    # 创建序列对象 没问题
    contigs = []
    for chrom in chrom_segments:
        contigs += [Contig(chrom, ctg_start, ctg_end) for ctg_start, ctg_end in chrom_segments[chrom]]

    # 处理序列长度
    seq_length = 131072
    crop_bp = 0  # 这里的剪切是指什么
    seq_tlength = seq_length - 2 * crop_bp
    contigs = [ctg for ctg in contigs if int(ctg.end) - int(ctg.start) >= seq_tlength]
    # 将过大的片段做切分
    break_t = 786432
    if break_t is not None:
        contigs = break_large_contigs(contigs, break_t)

    # 划分数据集
    # 不是k-flods，直接按概率来划分
    num_flods = 3
    fold_labels = ['train', 'valid', 'test']
    valid_pct = 0.2
    test_pct = 0.1
    fold_contigs = divide_contigs_pct(contigs, test_pct, valid_pct)

    # 合并每个floa里面的连续区间
    for fi in range(len(fold_contigs)):
        fold_contigs[fi] = rejoin_large_contigs(fold_contigs[fi])

    # 将分好的区间写到文件里 还不是固定长度的输入
    ctg_bed_file = '%s/contigs.bed' % out_dir
    ctg_bed_out = open(ctg_bed_file, 'w')
    for fi in range(len(fold_contigs)):
        for ctg in fold_contigs[fi]:
            line = '%s\t%d\t%d\t%s' % (ctg.chr, int(ctg.start), int(ctg.end), fold_labels[fi])
            print(line, file=ctg_bed_out)
    ctg_bed_out.close()

    # 得到固定长度的DNA序列
    snap = 64 # 是序列的位点对齐
    sample_pct = 1 # 小于1也就是取子集，用部分数据集做训练

    fold_mseqs = []
    for fi in range(len(fold_contigs)):
        if fold_labels[fi] in ['valid', 'test']:
            # 验证、测试的步长
            stride_fold = 256
        else:
            stride_fold = 256  # 训练，能不能设大
        fold_mseqs_fi = contig_sequences(fold_contigs[fi], seq_tlength,
                                         stride_fold, snap, fold_labels[fi])
        fold_mseqs.append(fold_mseqs_fi)

        # shuffle
        random.shuffle(fold_mseqs[fi])

        # down-sample 主要是方便调试，先用小部分的数据集
        if sample_pct < 1.0:
            fold_mseqs[fi] = random.sample(fold_mseqs[fi], int(sample_pct * len(fold_mseqs[fi])))

        # 将所有fold的ModelSeq放到一个列表里
        mseqs = [ms for fm in fold_mseqs for ms in fm]

        seqs_bed_file = '%s/sequences.bed' % out_dir
        write_seqs_bed(seqs_bed_file, mseqs, True)
