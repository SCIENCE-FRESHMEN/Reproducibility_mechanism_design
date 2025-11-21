import torch
import matplotlib.pyplot as plt
import os
from models.rochet_net import RochetNet
from utils.data_generation import generate_valuations


def train_rochet_net(num_items=3, num_samples=10000, epochs=100, lr=0.01, device='cpu'):
    """训练RochetNet模型"""

    # 生成单买家多物品的估值数据
    valuations = torch.rand(num_samples, num_items).to(device)

    # 创建模型
    model = RochetNet(num_items).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        allocations, payments = model(valuations)

        # 损失函数：卖家收入的负值
        loss = -torch.mean(payments)

        # 反向传播
        if loss.requires_grad:  # 确保loss需要梯度
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'Avg Payment: {torch.mean(payments).item():.4f}')

    # 保存结果
    os.makedirs('results/models', exist_ok=True)
    torch.save(model.state_dict(), f'results/models/rochet_net_{num_items}items.pth')

    # 绘制训练损失
    os.makedirs('results/plots', exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('RochetNet Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'results/plots/rochet_net_loss_{num_items}items.png')
    plt.close()

    return model