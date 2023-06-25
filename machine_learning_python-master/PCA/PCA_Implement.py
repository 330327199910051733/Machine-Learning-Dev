""""
功能：主成分分析，Principal Component Analysis（PCA）,对角化协方差矩阵
版本：V2.0
参考文献：
[1]进击的马斯特.浅谈协方差矩阵[DB/OL].http://pinkyjie.com/2010/08/31/covariance/,2010-08-31.
[2]进击的马斯特.再谈协方差矩阵之主成分分析[DB/OL].http://pinkyjie.com/2011/02/24/covariance-pca/,2011-02-24.
"""
from __future__ import division
import numpy as np
import pandas as pd


def jacobi_eig(matrix, tolerance=1e-10, max_iterations=1000):
    """
    使用Jacobi迭代法计算矩阵的特征值和特征向量。

    参数：
    matrix -- 输入矩阵，大小为(n, n)
    tolerance -- 迭代收敛的精度，默认为1e-10
    max_iterations -- 最大迭代次数，默认为1000

    返回：
    eigenvalues -- 特征值向量，大小为(n,)
    eigenvectors -- 特征向量矩阵，大小为(n, n)
    """
    n = matrix.shape[0]
    # 创建一个n行n列的单位矩阵，其中对角线上的元素为1，其余元素为0
    eigenvectors = np.eye(n)

    for i in range(max_iterations):
        # 计算非对角线元素中绝对值最大的位置
        max_idx = np.argmax(np.abs(np.triu(matrix, k=1)))
        row, col = divmod(max_idx, n)

        # 如果非对角线元素都已经很小了，就认为已经收敛了
        if np.abs(matrix[row, col]) < tolerance:
            break

        # 计算Jacobi旋转矩阵
        theta = 0.5 * np.arctan2(2 * matrix[row, col], matrix[col, col] - matrix[row, row])
        c = np.cos(theta)
        s = np.sin(theta)
        jacobi_matrix = np.eye(n)
        jacobi_matrix[row, row] = c
        jacobi_matrix[col, col] = c
        jacobi_matrix[row, col] = s
        jacobi_matrix[col, row] = -s

        # 更新矩阵和特征向量
        matrix = np.dot(jacobi_matrix.T, np.dot(matrix, jacobi_matrix))
        eigenvectors = np.dot(eigenvectors, jacobi_matrix)

    eigenvalues = np.diag(matrix)

    return eigenvalues, eigenvectors

# 根据保留多少维特征进行降维
class PCAcomponent(object):
    def __init__(self, X, N=3):
        self.X = X  # X：原始数据矩阵
        self.N = N  # N：需要降到的维度
        self.variance_ratio = []  # variance_ratio：每个维度所占的方差百分比
        self.low_dataMat = [] # low_dataMat：降维后的数据矩阵

    # 计算降维后的数据矩阵和每个维度的方差百分比
    def _fit(self):
        # 1.计算原始数据矩阵的均值，并将数据矩阵进行中心化
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        # 2.计算协方差矩阵，并使用jacobi方法计算特征值和特征向量
        # 另一种计算协方差矩阵的方法：dataMat.T * dataMat / dataMat.shape[0]
        # 若rowvar非0，一列代表一个样本；为0，一行代表一个样本
        covMat = np.cov(dataMat, rowvar=False)
        # 求特征值和特征向量，特征向量是按列放的，即一列代表一个特征向量
        # eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        eigVal, eigVect = jacobi_eig(np.mat(covMat))
        # 3.取前N个较大的特征值和对应的特征向量，得到N维投影矩阵
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-1:-(self.N + 1):-1]  # 取前N个较大的特征值
        # 4.使用投影矩阵将数据矩阵进行降维，得到降维后的数据矩阵
        small_eigVect = eigVect[:, eigValInd]  # *N维投影矩阵
        self.low_dataMat = dataMat * small_eigVect  # 投影变换后的新矩阵

        # reconMat = (self.low_dataMat * small_eigVect.I) + X_mean  # 重构数据
        # recover_dataMat = low_dataMat * small_eigVect.T + X_mean
        # 5.计算每个维度所占的方差百分比，并将结果存储在variance_ratio属性中
        # 输出每个维度所占的方差百分比
        [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
        return self.low_dataMat

    def fit(self):
        self._fit()
        return self
    # 重构数据
    def recover(self):
        # 1.计算原始数据矩阵的均值，并将数据矩阵进行中心化
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        # 2.计算协方差矩阵，并使用jacobi方法计算特征值和特征向量
        # 另一种计算协方差矩阵的方法：dataMat.T * dataMat / dataMat.shape[0]
        # 若rowvar非0，一列代表一个样本；为0，一行代表一个样本
        covMat = np.cov(dataMat, rowvar=False)
        # 求特征值和特征向量，特征向量是按列放的，即一列代表一个特征向量
        # eigVal, eigVect = np.linalg.eig(np.mat(covMat))
        eigVal, eigVect = jacobi_eig(np.mat(covMat))
        # 3.取前N个较大的特征值和对应的特征向量，得到N维投影矩阵
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-1:-(self.N + 1):-1]  # 取前N个较大的特征值
        # 4.使用投影矩阵将数据矩阵进行降维，得到降维后的数据矩阵
        small_eigVect = eigVect[:, eigValInd]  # *N维投影矩阵
        low_dataMat = dataMat * small_eigVect  # 投影变换后的新矩阵
        # 5.使用投影矩阵的逆矩阵将降维后的数据重新映射到原始高维空间中，得到恢复后的数据矩阵
        recover_dataMat = low_dataMat * small_eigVect.T + X_mean
        return recover_dataMat

# 根据保留多大方差百分比进行降维
class PCApercent(object):
    def __init__(self, X, percentage=0.95):
        self.X = X # X：原始数据矩阵
        self.percentage = percentage # percentage：需要保留的方差百分比
        self.variance_ratio = []  # 每个维度所占的方差百分比
        self.low_dataMat = []  # low_dataMat：降维后的数据矩阵

    # 通过方差百分比选取前n个主成份
    def percent2n(self, eigVal):
        # 1. 将特征值按从大到小排列
        sortVal = np.sort(eigVal)[-1::-1]
        # 2 .依次加上每个特征值，直到累计和大于等于所有特征值之和的指定百分比。记录此时已经加上的特征值个数，即为满足指定方差百分比的最小维度。
        percentSum, componentNum = 0, 0
        for i in sortVal:
            percentSum += i
            componentNum += 1
            if percentSum >= sum(sortVal) * self.percentage:
                break
        return componentNum

    # 计算降维后的数据矩阵和每个维度的方差百分比
    def _fit(self):
        # 1.计算原始数据矩阵的均值，并将数据矩阵进行中心化
        X_mean = np.mean(self.X, axis=0)
        dataMat = self.X - X_mean
        # 2.计算协方差矩阵，并使用jacobi方法计算特征值和特征向量
        covMat = np.cov(dataMat, rowvar=False)
        eigVal, eigVect = jacobi_eig(np.mat(covMat))
        # eigVal, eigVect = np.linalg.eig(np.mat(covMat))

        # 3.取前N个较大的特征值和对应的特征向量，得到N维投影矩阵
        n = self.percent2n(eigVal)
        eigValInd = np.argsort(eigVal)
        eigValInd = eigValInd[-1:-(n + 1):-1]
        n_eigVect = eigVect[:, eigValInd]
        # 4.使用投影矩阵将数据矩阵进行降维，得到降维后的数据矩阵
        self.low_dataMat = dataMat * n_eigVect
        # 5.计算每个维度所占的方差百分比，并将结果存储在variance_ratio属性中
        [self.variance_ratio.append(eigVal[i] / sum(eigVal)) for i in eigValInd]
        return self.low_dataMat

    def fit(self):
        self._fit()
        return self





df = pd.read_csv(r'X.txt', header=None)
data, label = df[range(len(df.columns) - 1)], df[[len(df.columns) - 1]]
data = np.mat(data)
print("Original dataset = {}*{}".format(data.shape[0], data.shape[1]))
pca = PCAcomponent(data, 2)
# pca = PCApercent(data, 0.98)
pca.fit()
# print()
print(pca.low_dataMat)
print(pca.recover())
print(pca.variance_ratio)