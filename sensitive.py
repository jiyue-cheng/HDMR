import torch


# 生成示例数据（假设x1, x2, x3是独立的正态分布变量）
def generate_data(num_samples=1000):
    torch.manual_seed(42)  # 固定随机种子保证可复现性

    # x1 = torch.normal(0, 1, (num_samples,))  # 均值0，标准差1
    # x2 = torch.normal(0, 1, (num_samples,))  # 均值0，标准差1
    # x3 = torch.normal(0, 1, (num_samples,))  # 均值0，标准差1

    x1 = torch.randn(num_samples, 1)
    x2 = torch.randn(num_samples, 1)
    x3 = torch.randn(num_samples, 1)

    # x1 = torch.rand(num_samples, 1)
    # x2 = torch.rand(num_samples, 1)
    # x3 = torch.rand(num_samples, 1)
    print(f"x1均值: {x1.mean():.4f}, 标准差: {x1.std():.4f}")
    print(f"x2均值: {x2.mean():.4f}, 标准差: {x2.std():.4f}")
    print(f"x3均值: {x3.mean():.4f}, 标准差: {x3.std():.4f}")
    return x1, x2, x3


# 计算方差和灵敏度
def compute_sensitivity(x1, x2, x3):
    # 计算各项方差（设置unbiased=False对应总体方差）
    mu_x1 = torch.mean(3*x1)
    var_x1 = torch.mean((3*x1 - mu_x1) ** 2)

    mu_x2 = torch.mean(2*x2)
    var_x2 = torch.mean((2*x2 - mu_x2) ** 2)

    ortho = torch.mean(3 * x1 * 2 * x2) ** 2
    print(f"协方差: {ortho:.4f}")

    mu_x3 = torch.mean(5*x3)
    var_x3 = torch.mean((5*x3 - mu_x3) ** 2)
    mu_x1x2 = torch.mean(3 * x1 * x2)
    var_x1x2 = torch.mean((3 * x1 * x2 - mu_x1x2) ** 2)
    mu_x1x3 = torch.mean(3 * x1 * x3)
    var_x1x3 = torch.mean((3 * x1 * x3 - mu_x1x3) ** 2)
    mu_x2x3 = torch.mean(3 * x2 * x3)
    var_x2x3= torch.mean((3 * x2 * x3 - mu_x2x3) ** 2)
    mu_x1x2x3 = torch.mean(3 * x1 * x2 * x3)
    var_x1x2x3 = torch.mean((3 * x1 * x2 * x3 - mu_x1x2x3) ** 2)
    ortho = torch.mean(3 * x2 * x3 * 3 * x1 * x2 * x3) ** 2
    print(f"协方差: {ortho:.4f}")

    # 计算总方差（各独立项方差之和）[3](@ref)
    output = 3*x1 + 2*x2 + 5*x3 + 4 + 3*x1*x2 + 3*x1*x3 + 3*x2*x3 + 3*x1*x2*x3
    mu_output = torch.mean(output)
    var_output = torch.mean((output - mu_output) ** 2)

    # 计算灵敏度（方差贡献率）
    sensitivity = [var_x1 / var_output,
                   var_x2 / var_output,
                   var_x3 / var_output,
                   var_x1x2 / var_output,
                   var_x1x3 / var_output,
                   var_x2x3 / var_output,
                   var_x1x2x3 / var_output
                   ]

    return var_x1.item(), var_x2.item(), var_x3.item(), var_x1x2, var_x1x3, var_x2x3, var_x1x2x3, var_output.item(), sensitivity


# 执行计算
if __name__ == "__main__":
    x1, x2, x3 = generate_data()
    var_x1, var_x2, var_x3, var_x1x2, var_x1x3, var_x2x3,  var_x1x2x3, total_var, sens = compute_sensitivity(x1, x2, x3)

    # 格式化输出（保留4位小数）
    print(f"3x₁的方差: {var_x1:.4f}，灵敏度: {sens[0]:.4f}")
    print(f"2x₂的方差: {var_x2:.4f}，灵敏度: {sens[1]:.4f}")
    print(f"5x₃的方差: {var_x3:.4f}，灵敏度: {sens[2]:.4f}")
    print(f"3x₁x₂的方差: {var_x1x2:.4f}，灵敏度: {sens[3]:.4f}")
    print(f"3x₁x₃的方差: {var_x1x3:.4f}，灵敏度: {sens[4]:.4f}")
    print(f"3x₂x₃的方差: {var_x2x3:.4f}，灵敏度: {sens[5]:.4f}")
    print(f"3x₁x₂x₃的方差: {var_x1x2x3:.4f}，灵敏度: {sens[6]:.4f}")

    sen_sum = sens[0]+sens[1]+sens[2]+sens[3]+sens[4]++sens[5]++sens[6]
    print(f"灵敏度之和: {sen_sum:.4f}")
    print(f"总方差: {total_var:.4f}")