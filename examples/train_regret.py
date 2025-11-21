import torch
import matplotlib.pyplot as plt
import os
from models.regret_net import RegretNet
from utils.data_generation import generate_valuations
from utils.evaluation import compute_utility, compute_regret


def train_regret_net(num_buyers=3, num_items=2, num_samples=5000, epochs=200, lr=0.001, device='cpu'):
    """训练RegretNet模型"""

    # 生成多买家多物品的估值数据
    valuations = generate_valuations(num_samples, num_buyers, num_items, device=device)

    # 创建模型
    model = RegretNet(num_buyers, num_items).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 拉格朗日乘子
    lambda_regret = torch.ones(num_buyers).to(device) * 0.1
    lambda_IR = torch.ones(num_buyers).to(device) * 0.1
    rho = 1.0  # 增广拉格朗日参数

    # 训练循环
    losses = []
    regrets = []
    IR_violations = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        allocations, payments = model(valuations)

        # 计算卖家收入
        revenue = torch.sum(payments, dim=1)
        revenue_loss = -torch.mean(revenue)

        # 计算事后遗憾
        regret = compute_regret(model, valuations)
        regret_loss = torch.sum(lambda_regret * torch.mean(regret, dim=0)) + \
                      (rho / 2) * torch.sum(torch.mean(regret ** 2, dim=0))

        # 计算个体理性约束违反
        utilities = compute_utility(allocations, payments, valuations)
        IR_violation = torch.mean(torch.clamp(-utilities, min=0), dim=0)  # 负效用表示违反
        IR_loss = torch.sum(lambda_IR * IR_violation) + (rho / 2) * torch.sum(IR_violation ** 2)

        # 总损失
        total_loss = revenue_loss + regret_loss + IR_loss

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 更新拉格朗日乘子
        with torch.no_grad():
            lambda_regret += rho * torch.mean(regret, dim=0)
            lambda_IR += rho * IR_violation

        losses.append(total_loss.item())
        regrets.append(torch.mean(regret).item())
        IR_violations.append(torch.mean(IR_violation).item())

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss.item():.4f}, '
                  f'Revenue: {torch.mean(revenue).item():.4f}, '
                  f'Avg Regret: {torch.mean(regret).item():.4f}, '
                  f'Avg IR Violation: {torch.mean(IR_violation).item():.4f}')

    # 保存结果
    os.makedirs('results/models', exist_ok=True)
    torch.save(model.state_dict(), f'results/models/regret_net_{num_buyers}buyers_{num_items}items.pth')

    # 绘制训练结果
    os.makedirs('results/plots', exist_ok=True)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('RegretNet Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(regrets)
    plt.title('Average Regret')
    plt.xlabel('Epoch')
    plt.ylabel('Regret')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(IR_violations)
    plt.title('Average IR Violation')
    plt.xlabel('Epoch')
    plt.ylabel('IR Violation')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'results/plots/regret_net_training_{num_buyers}buyers_{num_items}items.png')
    plt.close()

    return model