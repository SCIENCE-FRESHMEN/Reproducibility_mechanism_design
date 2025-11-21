import torch
import torch.nn as nn


class AllocationNetwork(nn.Module):
    """RegretNet的分配网络"""

    def __init__(self, num_buyers, num_items, hidden_size=100):
        super(AllocationNetwork, self).__init__()
        self.num_buyers = num_buyers
        self.num_items = num_items

        # 全连接网络
        self.net = nn.Sequential(
            nn.Linear(num_buyers * num_items, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_buyers * num_items)
        )

    def forward(self, bids):
        """
        Args:
            bids: 买家出价 [batch_size, num_buyers, num_items]

        Returns:
            allocations: 分配概率 [batch_size, num_buyers, num_items]
        """
        batch_size = bids.shape[0]

        # 展平输入
        flat_bids = bids.view(batch_size, -1)

        # 前向传播
        flat_allocs = self.net(flat_bids)

        # 重塑为 [batch_size, num_buyers, num_items]
        allocs = flat_allocs.view(batch_size, self.num_buyers, self.num_items)

        # 应用softmax确保每个物品的分配概率和为1
        allocs = torch.softmax(allocs, dim=1)

        # 确保分配概率在[0,1]范围内，且每个物品的总分配不超过1
        allocs = torch.clamp(allocs, 0, 1)
        item_sums = torch.sum(allocs, dim=1, keepdim=True)
        allocs = allocs / torch.maximum(item_sums, torch.ones_like(item_sums, device=allocs.device))

        return allocs


class PaymentNetwork(nn.Module):
    """RegretNet的支付网络"""

    def __init__(self, num_buyers, num_items, hidden_size=100):
        super(PaymentNetwork, self).__init__()
        self.num_buyers = num_buyers
        self.num_items = num_items

        # 全连接网络
        self.net = nn.Sequential(
            nn.Linear(num_buyers * num_items, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_buyers)
        )

    def forward(self, bids):
        """
        Args:
            bids: 买家出价 [batch_size, num_buyers, num_items]

        Returns:
            payments: 支付金额 [batch_size, num_buyers]
        """
        batch_size = bids.shape[0]

        # 展平输入
        flat_bids = bids.view(batch_size, -1)

        # 前向传播
        payments = self.net(flat_bids)

        # 确保支付非负
        payments = torch.abs(payments)

        return payments


class RegretNet(nn.Module):
    """RegretNet: 多买家多物品的近似DSIC拍卖"""

    def __init__(self, num_buyers, num_items, hidden_size=100):
        super(RegretNet, self).__init__()
        self.allocation_net = AllocationNetwork(num_buyers, num_items, hidden_size)
        self.payment_net = PaymentNetwork(num_buyers, num_items, hidden_size)

    def forward(self, bids):
        """
        Args:
            bids: 买家出价 [batch_size, num_buyers, num_items]

        Returns:
            allocations: 分配概率 [batch_size, num_buyers, num_items]
            payments: 支付金额 [batch_size, num_buyers]
        """
        allocations = self.allocation_net(bids)
        payments = self.payment_net(bids)
        return allocations, payments