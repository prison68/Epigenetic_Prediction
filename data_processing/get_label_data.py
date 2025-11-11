import collections
import h5py
import os
import numpy as np
import pandas as pd
import sys
import scipy.interpolate

ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])

class CovFace:
    def __init__(self, cov_file):
        self.cov_file = cov_file # 文件路径
        self.bigwig = False
        self.bed = False
        self.tsv = True
        self.cov_open = pd.read_csv(self.cov_file, sep='\t',
                             usecols=[0, 1, 2, 7],
                             dtype={'chr': str, 'start': int, 'end': int, 'log2_CPM_normalized_reads': float}) # 这里的float
        print(self.cov_open.head())
        # cov_ext = os.path.splitext(self.cov_file)[1].lower()
        # if cov_ext == '.gz':
        #     cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()
        #
        # if cov_ext in ['.bed', '.narrowpeak']:
        #     self.bed = True
        #     self.preprocess_bed()
        #
        # elif cov_ext in ['.bw', '.bigwig']:
        #     self.cov_open = pyBigWig.open(self.cov_file, 'r')
        #     self.bigwig = True
        #
        # elif cov_ext in ['.h5', '.hdf5', '.w5', '.wdf5']:
        #     self.cov_open = h5py.File(self.cov_file, 'r')
        #
        # else:
        #     print('Cannot identify coverage file extension "%s".' % cov_ext,
        #           file=sys.stderr)
        #     exit(1)

    def preprocess_bed(self):
        # read BED
        bed_df = pd.read_csv(self.cov_file, sep='\t',
                             usecols=range(3),
                             dtype={'chr': str, 'start': int, 'end': int})
        t = bed_df.chr.unique()

        # for each chromosome 为什么要用一样的命名
        self.cov_open = {}
        for chrm in bed_df.chr.unique():
            bed_chr_df = bed_df[bed_df.chr == chrm] #按染色体分组处理，拿到当前这个染色体的所有行

            # find max pos 因为这里取得是整条染色体的最后一个碱基的位置，所以是从0开始的，前面的都是false，有效区间才是true
            pos_max = bed_chr_df.end.max()

            # initialize array chrm=10 zeros在bool就是false
            self.cov_open[chrm] = np.zeros(pos_max, dtype='bool')

            # set peaks
            # chr_df.itertuples每行变为一个命名元组，可通过属性名访问
            for peak in bed_chr_df.itertuples():
                self.cov_open[peak.chr][peak.start:peak.end] = 1

    def read(self, chrm, start, end):
        if self.bigwig:
            cov = self.cov_open.values(chrm, start, end, numpy=True).astype('float16')

        elif self.tsv:
            pass
        else:
            if chrm in self.cov_open:
                cov = self.cov_open[chrm][start:end]

                # handle mysterious inf's
                # 将数组cov中的所有值限制在float16数据类型能够表示的安全范围内
                cov = np.clip(cov, np.finfo(np.float16).min, np.finfo(np.float16).max)

                # pad
                pad_zeros = end - start - len(cov)
                if pad_zeros > 0:
                    cov_pad = np.zeros(pad_zeros, dtype='bool')
                    cov = np.concatenate([cov, cov_pad])

            else:
                print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % \
                      (self.cov_file, chrm, start, end), file=sys.stderr)
                cov = np.zeros(end - start, dtype='float16')

        return cov

    def close(self):
        if not self.bed:
            self.cov_open.close()

def interp_nan(x, kind='linear'):
    '''Linearly interpolate to fill NaN.'''

    # pad zeroes
    xp = np.zeros(len(x) + 2)
    xp[1:-1] = x

    # find NaN
    x_nan = np.isnan(xp)

    if np.sum(x_nan) == 0:
        # unnecessary
        return x

    else:
        # interpolate
        inds = np.arange(len(xp))
        interpolator = scipy.interpolate.interp1d(
            inds[~x_nan],
            xp[~x_nan],
            kind=kind,
            bounds_error=False)

        loc = np.where(x_nan)
        xp[loc] = interpolator(loc)

        # slice off pad
        return xp[1:-1]

if __name__ == '__main__':
    model_seqs= []
    seqs_bed_file = '../data/mouse/sequences.bed'
    seqs_cov_file = '../data/mouse/seqs_cov/0.h5' # 应该是h5对应的路径
    genome_cov_file = '../data/mouse/mouse_testis/PS_K4_filtered.normalized_with_cpm.tsv'

    for line in open(seqs_bed_file):
        a = line.split()
        model_seqs.append(ModelSeq(a[0], int(a[1]), int(a[2]), None))

    num_seqs = len(model_seqs)
    seqs_len_nt = model_seqs[0].end - model_seqs[0].start
    # 为什么这里还要剪切
    crop_bp = 0
    seqs_len_nt -= 2*crop_bp
    pool_width = 256
    # L/分辨率
    target_length = seqs_len_nt // pool_width

    # 初始化文件对象
    seqs_cov_open = h5py.File(seqs_cov_file, 'w')
    targets = []
    # 打开实验文件的类
    genome_cov_open = CovFace(genome_cov_file)
    # 读取指定区间的数值，并对数值进行处理
    # 对每一个序列进行处理
    for si in range(num_seqs):
        mseq = model_seqs[si]
        # 所以这里读了？
        seq_cov_nt = genome_cov_open.read(mseq.chr[3:], mseq.start, mseq.end) # 序列划分文件是用的chr开头的，这里实验文件是用1，没有chr
        seq_cov_nt = seq_cov_nt.astype('float32')

        # 处理null值，线性插值
        interp_nan_e = True
        if interp_nan_e:
            seq_cov_nt = interp_nan(seq_cov_nt)







