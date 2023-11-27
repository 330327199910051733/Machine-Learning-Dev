import argparse
import inspect

import torch


# 定义一个配置类Config，用于管理模型训练的配置参数，并提供从命令行解析参数的功能。通过修改类中的属性的默认值，可以自定义模型训练的各种配置参数。
class Config:
    device = torch.device("cuda:0") #设备类型
    # device = torch.device("cpu")
    train_epochs = 20 # 训练轮数
    batch_size = 1024 # 批次大小
    learning_rate = 0.01 # 学习率
    l2_regularization = 1e-3  # 正则化系数
    learning_rate_decay = 0.99  # 学习率衰减程度

    # dataset_file = 'data/movie-ratings.csv'
    # dataset_file = 'data/Digital_Music_5.json.csv'
    dataset_file = 'data/ratings_Digital_Music.csv'
    # dataset_file = 'data/amazonCSJ.json.csv'
    saved_model = 'best_model.pt'

    embedding_dim = 10

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
