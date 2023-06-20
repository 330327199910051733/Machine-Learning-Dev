import numpy as np
import numbers
from scipy.stats import multivariate_normal

#返回欧几里得距离
def distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2), axis=1))


#检查并返回一个随机数生成器对象，在使用随机数生成器对象时，其随机序列的起始状态是可控的，从而使得实验结果可以被重现。
def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

#  将一组标签（label）按照从小到大的顺序进行排序，并返回排序后的标签
def sortLabel(label):
    # 将label转换为数组
    label = np.array(label)
    # 创建一个空列表，用于存放原始标签
    label_old = []
    # 获取标签的数量
    label_num = len(list(set(label)))
    # 遍历标签
    for i in label:
        # 如果标签不在原始标签列表中，则将其添加到原始标签列表中
        if i not in label_old:
            label_old.append(i)
        # 如果原始标签列表的长度等于标签的数量，则跳出循环
        if len(label_old) == label_num:
            break

    # 对原始标签列表进行排序
    label_new = sorted(label_old)
    # 遍历原始标签列表，将排序后的标签赋值给标签
    for i, old in enumerate(label_old):
        label[label == old] = label_new[i] + 10000
    return label - 10000

def prob(x, mu, cov):
    # 创建一个多重正态分布，均值为mu，协方差为cov
    norm = multivariate_normal(mean=mu, cov=cov)
    # 返回正态分布的概率密度函数
    return norm.pdf(x)

def log_prob(x, mu, cov):
    # 创建一个多重正态分布，均值为mu，协方差为cov
    norm = multivariate_normal(mean=mu, cov=cov)
    # 返回正态分布的概率密度函数的对数函数
    return norm.logpdf(x)


def log_weight_prob(x, alpha, mu, cov):
    # 计算x的概率密度函数
    N = x.shape[0]
    # 返回x的概率密度函数的对数函数
    return np.mat(np.log(alpha) + log_prob(x, mu, cov)).reshape([N, 1])