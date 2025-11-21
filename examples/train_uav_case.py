import torch
import matplotlib.pyplot as plt
import os
from models.uav_auction import UAVAuctionModel


def train_uav_auction(num_uavs=5, num_samples=1000, epochs=100, lr=0.01, device='cpu'):
    """训练无人机拍卖模型"""

    # 生成模拟数据
    # 感知率: 1-11 packets/sec
    sensing_rates = (torch.rand(num_samples, num_uavs, device=device) * 10 + 1) * 100
    # 总能耗: 50-150 units
    total_energies = torch.rand(num_samples, num_uavs, device=device) * 100 + 50
    # 剩余能量: 10-60 units
    remaining_energies = torch.rand(num_samples, num_uavs, device=device) * 50 + 10

    # 创建模型
    model = UAVAuctionModel(num_uavs).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        allocations, payments, valuations, virtual_values = model(
            sensing_rates, total_energies, remaining_energies
        )

        # 损失函数：最大化收入
        revenue = torch.sum(payments, dim=1)
        loss = -torch.mean(revenue)

        # 反向传播
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'Avg Revenue: {torch.mean(revenue).item():.4f}')

    # 保存结果
    os.makedirs('results/models', exist_ok=True)
    torch.save(model.state_dict(), f'results/models/uav_auction_{num_uavs}uavs.pth')

    # 绘制训练损失
    os.makedirs('results/plots', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('UAV Auction Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'results/plots/uav_auction_loss_{num_uavs}uavs.png')
    plt.close()

    return model