import numpy as np
from utils.misc_utils import distance, check_random_state, sortLabel

# 定义测试样例
point1 = np.array([[1, 2, 3]])
point2 = np.array([[9, 8, 7]])

# 调用函数计算欧几里得距离
result = distance(point1, point2)

# 打印输出
print(result)


# 定义测试样例
label = [2, 3, 1, 3, 2, 2]

# 调用函数进行排序
sorted_label = sortLabel(label)

# 打印输出结果
print(sorted_label)