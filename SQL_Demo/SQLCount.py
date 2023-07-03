import pandas as pd
import numpy as np
import os
import shutil


def copy_csv_file(csv_file, destination):
    # 读取 CSV 文件
    data = pd.read_csv(csv_file)

    # 复制 CSV 文件到指定目录
    shutil.copyfile(csv_file, destination)

    # 计算统计指标
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        # 如果有数值型的列，计算数值型列的统计指标
        numeric_stats = data[numeric_columns].describe().transpose()
    else:
        # 如果没有数值型的列，将 numeric_stats 设为空 DataFrame
        numeric_stats = pd.DataFrame()

    # 计算 count 统计指标2
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
    non_numeric_stats = data[non_numeric_columns].count().to_frame(name='count')

    # 合并统计指标
    stats = pd.concat([non_numeric_stats, numeric_stats], axis=0)

    # 将统计指标输出到 CSV 文件中
    stats.to_csv('description.csv', header=True, index=True)

    # 返回读取的数据
    return data


if __name__ == '__main__':
    data = copy_csv_file('house_prices1.csv', 'house_prices1_copy.csv')
    print(data)