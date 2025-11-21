import torch
import matplotlib.pyplot as plt
import os
from models.procurement_auction import ProcurementAuction
from utils.evaluation import compute_procurement_regret, compute_envy


def train_procurement_auction(num_suppliers=10, num_brackets=3, num_samples=1000, epochs=100, lr=0.001, device='cpu'):
    """训练采购拍卖模型"""

    # 生成模拟供应商出价数据
    # 出价随批量增加而减少（批量折扣）
    bids = torch.zeros(num_samples, num_suppliers, num_brackets, device=device)
    for k in range(num_brackets):
        base_price = torch.rand(num_samples, num_suppliers, device=device) * 50 + 50  # 50-100
        discount = torch.rand(num_samples, num_suppliers, device=device) * 0.3 * k  # 随批量增加而增加的折扣
        bids[:, :, k] = base_price * (1 - discount)

    # 创建模型
    model = ProcurementAuction(num_suppliers, num_brackets).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 拉格朗日乘子
    lambda_regret = 0.1
    lambda_envy = 0.1
    lambda_business = 0.1
    rho = 1.0

    # 训练循环
    losses = []
    costs = []
    regrets = []

    # 在训练循环中启用异常检测
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        allocations, payments = model(bids)

        # 总成本
        cost = torch.sum(payments * allocations, dim=1)
        cost_loss = torch.mean(cost)

        # 遗憾惩罚
        regret = compute_procurement_regret(model, bids)
        regret_loss = lambda_regret * torch.mean(regret) + (rho / 2) * torch.mean(regret ** 2)

        # 嫉妒惩罚
        envy = compute_envy(allocations, payments)
        envy_loss = lambda_envy * torch.mean(envy) + (rho / 2) * torch.mean(envy ** 2)

        # 商业约束惩罚（最小采购量）
        amin = 0.1  # 最小采购比例
        business_violation = torch.clamp(amin - torch.sum(allocations, dim=1), min=0)
        business_loss = lambda_business * torch.mean(business_violation) + (rho / 2) * torch.mean(
            business_violation ** 2)

        # 总损失
        total_loss = cost_loss + regret_loss + envy_loss + business_loss

        # 确保损失需要梯度
        if total_loss.requires_grad:
            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

        losses.append(total_loss.item())
        costs.append(torch.mean(cost).item())
        regrets.append(torch.mean(regret).item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss.item():.4f}, '
                  f'Avg Cost: {torch.mean(cost).item():.4f}, '
                  f'Avg Regret: {torch.mean(regret).item():.4f}, '
                  f'Avg Envy: {torch.mean(envy).item():.4f}')

    # 保存结果
    os.makedirs('results/models', exist_ok=True)
    torch.save(model.state_dict(),
               f'results/models/procurement_auction_{num_suppliers}suppliers_{num_brackets}brackets.pth')

    # 绘制训练损失
    os.makedirs('results/plots', exist_ok=True)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Procurement Auction Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(costs)
    plt.title('Average Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(regrets)
    plt.title('Average Regret')
    plt.xlabel('Epoch')
    plt.ylabel('Regret')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'results/plots/procurement_auction_training_{num_suppliers}suppliers_{num_brackets}brackets.png')
    plt.close()

    return model