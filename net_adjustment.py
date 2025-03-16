import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 定义需要拟合的目标函数
def target_function(x, y, z):
    # return 5*x**2+7*y**2+3*z**2
    return 5*x+2*y+7*z+x*y+3*x*z+9*y*z+10*x*y*z+5
    # return 5*x**2


# 随机生成变量数据,三阶
def generate_data(num_samples):
    x1_data = torch.rand(num_samples, 1)
    x2_data = torch.rand(num_samples, 1)
    x3_data = torch.rand(num_samples, 1)
    target = target_function(x1_data, x2_data, x3_data)   # 只针对自己有定义的对应输入变量和输出的映射关系
    return x1_data, x2_data, x3_data, target

# 定义零模型（输出恒为零）
class ZeroModel(nn.Module):
    def __init__(self):
        super(ZeroModel, self).__init__()

    def forward(self, x):
        return torch.zeros_like(x[:, 0])  # 保持与原模块输出维度一致

# HDMR零阶模块
class HDMR_F0(nn.Module):
    def __init__(self):
        super(HDMR_F0, self).__init__()
        self.f0 = nn.Parameter(torch.tensor(0.0))  # 直接设置一个简单的可学习常数参数

    def forward(self, inputs):
        return self.f0 * torch.ones_like(inputs)  # 对每个数据都有常数偏移


# HDMR一阶模块 一输入一输出mlp网络模型
class HDMR_FirstOrder(nn.Module):
    def __init__(self, hidden_size, hidden_num):
        super(HDMR_FirstOrder, self).__init__()
        self.input_layer = nn.Linear(1, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            ) for _ in range(hidden_num)
        ])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.input_layer(x))
        for layer in self.hidden_layers:
            out = layer(out)
        output = self.output_layer(out)
        return output


# HDMR二阶模块 二输入一输出mlp模型
class HDMR_SecondOrder(nn.Module):
    def __init__(self, hidden_size, hidden_num):
        super(HDMR_SecondOrder, self).__init__()
        self.input_layer = nn.Linear(2, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            ) for _ in range(hidden_num)
        ])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.input_layer(x))
        for layer in self.hidden_layers:
            out = layer(out)
        output = self.output_layer(out)
        return output

# HDMR三阶模块 三输入一输出mlp模型
class HDMR_ThirdOrder(nn.Module):
    def __init__(self, hidden_size, hidden_num):
        super(HDMR_ThirdOrder, self).__init__()
        self.input_layer = nn.Linear(3, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            ) for _ in range(hidden_num)
        ])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.input_layer(x))
        for layer in self.hidden_layers:
            out = layer(out)
        output = self.output_layer(out)
        return output


# 构建HDMR神经网络模型
class HDMR_Network(nn.Module):
    def __init__(self, first_order_params, second_order_params, third_order_params):
        super(HDMR_Network, self).__init__()
        self.f0_module = HDMR_F0()

        # 使用nn.ModuleList来存储各个模块
        self.first_order_modules = nn.ModuleList([HDMR_FirstOrder(hidden_size, hidden_num)
                                                  for hidden_size, hidden_num in first_order_params])

        self.second_order_modules = nn.ModuleList(
            [HDMR_SecondOrder(hidden_size, hidden_num) for hidden_size, hidden_num in second_order_params])

        self.third_order_modules = nn.ModuleList(
            [HDMR_ThirdOrder(hidden_size, hidden_num) for hidden_size, hidden_num in third_order_params])

        # 存储indices
        self.first_order_indices = list(range(num_vars))
        self.second_order_indices = []
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                self.second_order_indices.append((i, j))  # 记录两两配对的索引
        self.third_order_indices = []
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                for k in range(j + 1, num_vars):
                    self.third_order_indices.append((i, j, k))  # 记录三三配对的索引

        # 用于存储各阶输出
        self.f_j = []
        self.f_j1j2 = []
        self.f_j1j2j3 = []

        # 用于存储各阶方差信息
        self.first_order_V = []
        self.second_order_V = []
        self.third_order_V = []

    def forward(self, x):
        """
        前向传播过程，整合各阶模块的输出结果
        """
        # 重新清零
        self.first_order_V = []
        self.second_order_V = []
        self.third_order_V = []
        self.f_j = []
        self.f_j1j2 = []
        self.f_j1j2j3 = []

        # 零阶模块输出
        f0_output = self.f0_module(x[:, 0])
        f0_result = f0_output.unsqueeze(1)

        # 一阶模块输出之和
        first_order_sum = 0
        for index in range(len(self.first_order_modules)):
            i = self.first_order_indices[index]
            f_j_element = self.first_order_modules[index](x[:, i].unsqueeze(1)).squeeze() - f0_output
            self.f_j.append(f_j_element)
            # 计算每个f_j均值
            mu_j = torch.mean(f_j_element)
            # 计算一阶模块方差
            self.first_order_V.append(torch.sum((f_j_element-mu_j) ** 2))
        for element in self.f_j:
            first_order_sum += element
        first_order_output = first_order_sum
        first_order_result = first_order_output.unsqueeze(1)
        # print(f"first_order_result:{first_order_result}")

        # 二阶模块输出之和
        second_order_sum = 0
        for index in range(len(self.second_order_modules)):
            i, j = self.second_order_indices[index]
            input_pairs = torch.cat([x[:, i].unsqueeze(1), x[:, j].unsqueeze(1)], dim=1)
            f_j1j2_element = self.second_order_modules[index](input_pairs).squeeze()-self.f_j[i]-self.f_j[j]-f0_output
            self.f_j1j2.append(f_j1j2_element)
            # 计算f_j1j2均值
            mu_j1j2 = torch.mean(f_j1j2_element)
            # 计算二阶模块方差
            self.second_order_V.append(torch.sum((f_j1j2_element-mu_j1j2) ** 2))
        for element in self.f_j1j2:
            second_order_sum += element
        second_order_output = second_order_sum
        second_order_result = second_order_output.unsqueeze(1)
        # print(f"second_order_result:{second_order_result}")

        # 三阶模块输出之和
        third_order_sum = 0
        for index in range(len(self.third_order_modules)):
            i, j, k = self.third_order_indices[index]
            input_triplets = torch.cat([x[:, i].unsqueeze(1), x[:, j].unsqueeze(1), x[:, k].unsqueeze(1)], dim=1)
            f_j1j2j3_element = (self.third_order_modules[index](input_triplets).squeeze()-self.f_j1j2[i]-self.f_j1j2[j]
                                - self.f_j1j2[k]-self.f_j[i]-self.f_j[j]-self.f_j[k]-f0_output)
            self.f_j1j2j3.append(f_j1j2j3_element)
            # 计算f_j1j2j3均值
            mu_j1j2j3 = torch.mean(f_j1j2j3_element)
            self.third_order_V.append(torch.sum((f_j1j2j3_element-mu_j1j2j3) ** 2))
        for element in self.f_j1j2j3:
            third_order_sum += element
        third_order_output = third_order_sum
        third_order_result = third_order_output.unsqueeze(1)
        # print(f"third_order_result:{third_order_result}")

        # 各阶结果相加
        final_output = f0_result + first_order_result + second_order_result + third_order_result
        # final_output = f0_output + first_order_output + second_order_output + third_order_output
        return final_output

    # 根据灵敏度调整各阶模块参数  threshold: 灵敏度阈值
    def adjust_modules(self, sensitivities, threshold=0.2, size_factor=1.2, num_factor=2, base_size=100, base_num=2):
        # 调整一阶模块
        for i in range(len(self.first_order_modules)):
            sensitivity = sensitivities[0][i]
            if sensitivity > threshold:
                new_size = min(200, int(base_size * sensitivity * size_factor))
                new_num = min(3, int(base_num + sensitivity * num_factor))
            else:
                new_size = max(10, int(base_size * sensitivity))
                new_num = base_num
            print(f'first new_size{new_size}new_num{new_num}')
            # 动态修改网络结构
            self.first_order_modules[i] = HDMR_FirstOrder(new_size, new_num)
            if sensitivity < 0.001:
                self.first_order_modules[i] = ZeroModel()

        # 调整二阶模块
        for i in range(len(self.second_order_modules)):
            sensitivity = sensitivities[1][i]
            if sensitivity > threshold:
                new_size = min(200, int(base_size * sensitivity * size_factor))
                new_num = min(3, int(base_num + sensitivity * num_factor))
            else:
                new_size = max(10, int(base_size * sensitivity))
                new_num = base_num
            print(f'second new_size{new_size}new_num{new_num}')
            # 动态修改网络结构
            self.second_order_modules[i] = HDMR_SecondOrder(new_size, new_num)
            if sensitivity < 0.001:
                self.second_order_modules[i] = ZeroModel()
        # 调整三阶模块
        for i in range(len(self.third_order_modules)):
            sensitivity = sensitivities[2][i]
            if sensitivity > threshold:
                new_size = min(200, int(base_size * sensitivity * size_factor))
                new_num = min(3, int(base_num + sensitivity * num_factor))
            else:
                new_size = max(10, int(base_size * sensitivity))
                new_num = base_num
            print(f'third new_size{new_size}new_num{new_num}')
            # 动态修改网络结构
            self.third_order_modules[i] = HDMR_ThirdOrder(new_size, new_num)
            if sensitivity < 0.001:
                self.third_order_modules[i] = ZeroModel()



# 设置训练模型及参数
def train_model(model, num_epochs, num_samples):
    """
    训练模型函数
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    mse = []

    # print("Model parameters:")
    # for param in model.parameters():
    #     print(param.shape)

    # 生成输入数据
    x1_data, x2_data, x3_data, target = generate_data(num_samples)
    inputs = torch.cat([x1_data, x2_data, x3_data], dim=1)
    # print(inputs)

    sensitivities_sum = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        mse.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        # 计算目标函数的方差
        mu_outputs = torch.mean(outputs)
        outputs_V = torch.sum((outputs - mu_outputs) ** 2)
        # print(outputs_V)
        # 计算各阶灵敏度指标
        first_order_sensitivities = [var_f_j / outputs_V for var_f_j in model.first_order_V]
        second_order_sensitivities = [var_f_j1j2 / outputs_V for var_f_j1j2 in model.second_order_V]
        third_order_sensitivities = [var_f_j1j2j3 / outputs_V for var_f_j1j2j3 in model.third_order_V]
        sensitivities = [
            first_order_sensitivities,
            second_order_sensitivities,
            third_order_sensitivities
        ]
        # 求灵敏度之和
        result = torch.tensor(0.0)  # 先初始化结果为0张量
        for list in [first_order_sensitivities, second_order_sensitivities, third_order_sensitivities]:
            for element in list:
                result += element
                # 将每次计算的 result 值添加到列表中
        sensitivities_sum.append(result.item())
        if epoch % 100 == 0:
            print(f'Epoch {epoch+1}: Loss = {loss.item()}')
            print(first_order_sensitivities)
            print(second_order_sensitivities)
            print(third_order_sensitivities)
            # print(outputs_V)
            print(f"灵敏度之和{result}")
            print("-------------------------------------------")
            # current_param_ids = [id(p) for p in model.parameters()]
            # print("Model parameter IDs:", current_param_ids)

        if epoch >= 1000 and epoch % 5000 == 0:
            model.adjust_modules(sensitivities)
            optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])

            # 绘制损失曲线
    plt.plot(mse)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mse Loss')
    plt.grid(True)
    plt.show()

    # 绘制sensitivities_sum的变化图像
    plt.plot(range(0, num_epochs), sensitivities_sum)
    # 绘制sensitivities_sum = 1的水平直线
    x_vals = range(0, num_epochs)
    y_constant = [1] * num_epochs
    plt.plot(x_vals, y_constant, label='sensitivities_sum = 1')
    plt.xlabel('Epoch')
    plt.ylabel('Sum of Sensitivities')
    plt.title('Change of Sum of Sensitivities over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

# 主函数
if __name__ == "__main__":
    num_vars = 3  # 定义维数
    # 初始化一阶、二阶、三阶模型参数
    first_order_params = [(100, 2), (100, 2), (100, 2)]
    second_order_params = [(100, 2), (100, 2), (100, 2)]
    third_order_params = [(100, 1)]
    model = HDMR_Network(first_order_params, second_order_params, third_order_params)
    num_epochs = 30000
    num_samples = 100
    train_model(model, num_epochs, num_samples)

