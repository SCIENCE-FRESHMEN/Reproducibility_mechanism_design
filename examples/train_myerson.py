import torch
import matplotlib.pyplot as plt
import os
from models.myerson_net import MyersonNet
from utils.data_generation import generate_valuations


def train_myerson_net(num_buyers=3, num_samples=10000, epochs=100, lr=0.01, device='cpu'):
    """训练MyersonNet模型"""

    # 生成单物品多买家的估值数据
    # 只需要单物品，所以取所有物品中的第一个
    all_valuations = generate_valuations(num_samples, num_buyers, 1, device=device)
    valuations = all_valuations[:, :, 0]  # [batch_size, num_buyers]

    # 创建模型
    model = MyersonNet(num_buyers).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        allocations, payments = model(valuations)

        # 计算卖家收入
        revenue = torch.sum(payments, dim=1)
        loss = -torch.mean(revenue)  # 最大化收入

        # 反向传播
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'Avg Revenue: {torch.mean(revenue).item():.4f}')

    # 保存结果
    os.makedirs('results/models', exist_ok=True)
    torch.save(model.state_dict(), f'results/models/myerson_net_{num_buyers}buyers.pth')

    # 绘制训练损失
    os.makedirs('results/plots', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('MyersonNet Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'results/plots/myerson_net_loss_{num_buyers}buyers.png')
    plt.close()

    return model