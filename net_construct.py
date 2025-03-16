import torch
import torch.nn as nn


def create_custom_nn(input_size, output_size, hidden_sizes, hidden_layers):
    """
    根据输入变量的特性动态创建具有可变层数和神经元数量的神经网络。

    :param input_size: 输入层大小
    :param output_size: 输出层大小
    :param hidden_sizes: 隐藏层神经元个数
    :param hidden_layers: 隐藏层层数
    :return: 定制的神经网络模块
    """
    class CustomNN(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim):
            super(CustomNN, self).__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.Sigmoid())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)
