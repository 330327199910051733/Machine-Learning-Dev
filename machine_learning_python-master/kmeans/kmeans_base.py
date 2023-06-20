from collections import defaultdict
import time
# from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
from utils.misc_utils import distance, check_random_state, sortLabel

class KMeansBase(object):
    # 初始化参数
    def __init__(self, n_clusters = 8, init = "random", max_iter = 300, random_state = None, n_init = 10, tol = 1e-4):
        self.k = n_clusters # 聚类个数
        self.init = init # 输出化方式
        self.max_iter = max_iter # 最大迭代次数
        self.random_state = check_random_state(random_state) #随机数
        self.n_init = n_init # 进行多次聚类，选择最好的一次
        self.tol = tol # 停止聚类的阈值

    def fit(self, dataset):
        """fit对train建立模型

        Args:
            dataset:聚类的数据集

        Returns:best_labels, best_centers, best_error(最好的一组标签，聚类中心点，及误差的大小)

        """
        self.tol = self._tolerance(dataset, self.tol)
        best_error = None  # 表示最佳聚类结果的误差
        best_centers = None  # 表示聚类中心的坐标
        best_labels = None  # 代表聚类标签，取值范围是0-k-1
        # 执行n_init次K均值聚类算法，每次都随机初始化聚类中心，得到一组聚类结果。在每次聚类完成后，它记录当前的最佳聚类结果
        for i in range(self.n_init):
            labels, centers, error = self._kmeans(dataset)
            if best_error == None or error < best_error:
                best_error = error
                best_centers = centers
                best_labels = labels
        self.centers = best_centers
        return best_labels, best_centers, best_error

    def predict(self, X):
        """predict根据训练好的模型预测新的数据

        Args:
            X: 聚类的数据集

        Returns:

        """
        return self.update_labels_error(X, self.centers)[0]

    def fit_predict(self, dataset):
        """合并fit和predict

        Args:
            dataset: 聚类的数据集

        Returns:预测结果

        """
        self.fit(dataset)
        return self.predict(dataset)

    def _kmeans(self, dataset):
        """kmeans的主要方法，完成一次聚类的过程

        Args:
            dataset: 聚类的数据集

        Returns:best_labels, best_centers, best_error(返回当前的最好的标签，聚类中心，以及误差)

        """
        self.dataset = np.array(dataset)
        best_error = None
        best_centers = None
        best_labels = None
        center_shift_total = 0
        centers = self._init_centroids(dataset)  # 初始化中心点
        for i in range(self.max_iter):
            old_centers = centers.copy()
            labels, error = self.update_labels_error(dataset, centers)
            centers = self.update_centers(dataset, labels)

            if best_error == None or error < best_error:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_error = error

            #  oldCenters和centers的偏移量，判断当前迭代过程中聚类中心是否已经收敛
            center_shift_total = np.linalg.norm(old_centers - centers) ** 2
            if center_shift_total <= self.tol:
                break

        #  由于上面的循环，最后一步更新了centers，所以如果和旧的centers不一样的话，再更新一次labels，error
        if center_shift_total > 0:
            best_labels, best_error = self.update_labels_error(dataset, best_centers)

        return best_labels, best_centers, best_error

    def _init_centroids(self, dataset) -> np.ndarray:
        """k个数据点，随机生成,初始化聚类中心,根据指定的初始化方式从输入的数据集中获取初始聚类中心点

        Args:
            dataset:聚类的数据集

        Returns:聚类的中心

        """
        n_samples = dataset.shape[0]
        centers = []
        if self.init == "random":
            seeds = self.random_state.permutation(n_samples)[:self.k]
            centers = dataset[seeds]
        # kmeans的改进版
        elif self.init == "k-means++":
            pass
        return np.array(centers)

    def _tolerance(self, dataset, tol):
        """把tol和dataset相关联，计算聚类算法的收敛容差参数，计算数据集每个特征的方差，并返回一个阈值，选择对于模型性能较为重要的特征，例如方差较大的特征对于模型的影响会比较大

        Args:
            dataset: 聚类的数据集
            tol:某个阈值可以自己设定来判断停止条件

        Returns:

        """
        variances = np.var(dataset, axis=0)
        return np.mean(variances) * tol

    def update_labels_error(self, dataset, centers):
        """更新每个点的标签，和计算误差

        Args:
            dataset:聚类的数据集
            centers:聚类的中心

        Returns:标签的值及误差的大小

        """
        labels = self.assign_points(dataset, centers)
        new_means = defaultdict(list)
        error = 0
        for assignment, point in zip(labels, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            new_center = np.mean(points, axis=0)
            error += np.sqrt(np.sum(np.square(points - new_center)))

        return labels, error

    def update_centers(self, dataset, labels):
        """更新中心点

        Args:
            dataset: 聚类的数据集
            labels: 数据的标签

        Returns:聚类的中心的数组

        """
        new_means = defaultdict(list)
        centers = []
        for assignment, point in zip(labels, dataset):
            new_means[assignment].append(point)

        for points in new_means.values():
            new_center = np.mean(points, axis=0)
            centers.append(new_center)

        return np.array(centers)

    def assign_points(self, dataset, centers):
        """分配每个点到最近的center

        Args:
            dataset:需要聚类的数据集
            centers:聚类的中心点

        Returns:返回数据集到中心点的距离哪个近,然后分配相应的标签

        """
        labels = []
        for point in dataset:
            shortest = float("inf")  # 正无穷
            shortest_index = 0
            for i in range(len(centers)):
                val = distance(point[np.newaxis], centers[i])
                if val < shortest:
                    shortest = val
                    shortest_index = i
            labels.append(shortest_index)
        return labels


if __name__ == "__main__":
    #读取X.txt的文件要去掉第一行
    # X = np.genfromtxt("../kmeans/X.txt", delimiter=",", skip_header=1)
    # print(X)
    # km = KMeansBase(3)
    # startTime = time.time()
    # labels = km.fit_predict(X)
    # print("km time", time.time() - startTime)
    # print(np.array(sortLabel(labels)))
    # 用iris数据集测试准确度和速度
    # dataset:'data': array([[5.1, 3.5, 1.4, 0.2],
    #        [4.9, 3. , 1.4, 0.2]
    iris = datasets.load_iris()
    print(iris)
    km = KMeansBase(3)
    startTime = time.time()
    labels = km.fit_predict(iris.data)
    print("km time", time.time() - startTime)

    print(np.array(sortLabel(labels)))

