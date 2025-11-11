import pandas as pd
import numpy as np


def sliding_window_chunks(df, window_size=4, stride=1):
    """
    使用滑动窗口分割DataFrame，允许重叠

    参数:
        df: 要分割的DataFrame
        window_size: 窗口大小（每个块包含的行数）
        stride: 步长（窗口移动的距离）
    """
    chunks = []
    total_rows = len(df)

    for start in range(0, total_rows - window_size + 1, stride):
        end = start + window_size
        chunk = df.iloc[start:end]
        chunks.append(chunk)
        print(f"窗口 {len(chunks)}: 行 {start}-{end - 1}")

    return chunks


# 使用示例
data = pd.DataFrame({
    'position': range(100, 1000, 100),
    'value': np.random.randn(9)
})

print("原始数据:")
print(data)

# 窗口大小=4，步长=2（50%重叠）
chunks = sliding_window_chunks(data, window_size=4, stride=1)

for i, chunk in enumerate(chunks):
    print(f"\n块 {i + 1}:")
    print(chunk)