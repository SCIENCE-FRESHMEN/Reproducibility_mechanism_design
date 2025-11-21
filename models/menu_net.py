import torch
import torch.nn as nn


class MechanismNetwork(nn.Module):
    """MenuNet的机制网络"""

    def __init__(self, num_items, num_menu_items):
        super(MechanismNetwork, self).__init__()
        self.num_items = num_items
        self.num_menu_items = num_menu_items

        # 为每个菜单项和物品分配概率
        # 输入是常数1，输出是分配矩阵P和支付向量t
        self.alloc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, num_menu_items)
            ) for _ in range(num_items)
        ])

        # 支付网络
        self.payment_layer = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, num_menu_items)
        )

    def forward(self, x):
        """
        Args:
            x: 输入常数1 [batch_size, 1]

        Returns:
            P: 分配矩阵 [batch_size, num_items, num_menu_items]
            t: 支付向量 [batch_size, num_menu_items]
        """
        batch_size = x.shape[0]

        # 计算分配矩阵
        P = torch.zeros(batch_size, self.num_items, self.num_menu_items, device=x.device)
        for i in range(self.num_items):
            P[:, i] = torch.sigmoid(self.alloc_layers[i](x))

        # 确保IR：最后一个菜单项设置为0 (默认选项)
        P[:, :, -1] = 0.0

        # 计算支付向量
        t = torch.abs(self.payment_layer(x))  # 确保支付非负
        t[:, -1] = 0.0  # 默认选项支付为0

        return P, t


class BuyerNetwork(nn.Module):
    """MenuNet的买家网络，模拟买家选择行为"""

    def __init__(self, num_items, num_menu_items):
        super(BuyerNetwork, self).__init__()
        self.num_items = num_items
        self.num_menu_items = num_menu_items

        # 买家选择网络
        self.choice_net = nn.Sequential(
            nn.Linear(num_items + num_menu_items * num_items + num_menu_items, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_menu_items)
        )

    def forward(self, valuations, P, t):
        """
        Args:
            valuations: 买家估值 [batch_size, num_items]
            P: 分配矩阵 [batch_size, num_items, num_menu_items]
            t: 支付向量 [batch_size, num_menu_items]

        Returns:
            choices: 买家在菜单中的选择概率 [batch_size, num_menu_items]
        """
        batch_size = valuations.shape[0]

        # 计算每个菜单项的效用
        # 效用 = 价值 - 支付
        # 价值 = sum_i v_i * P_i,k
        value_per_menu = torch.zeros(batch_size, self.num_menu_items, device=valuations.device)
        for k in range(self.num_menu_items):
            # 计算菜单项k的价值
            value_per_menu[:, k] = torch.sum(valuations * P[:, :, k], dim=1)

        # 效用 = 价值 - 支付
        utility_per_menu = value_per_menu - t

        # 买家选择：效用最高的菜单项
        choices = torch.softmax(utility_per_menu * 10.0, dim=1)  # 10.0是温度参数，使选择更确定

        return choices, utility_per_menu


class MenuNet(nn.Module):
    """MenuNet: 基于菜单的拍卖设计"""

    def __init__(self, num_items, num_menu_items):
        super(MenuNet, self).__init__()
        self.mechanism_net = MechanismNetwork(num_items, num_menu_items)
        self.buyer_net = BuyerNetwork(num_items, num_menu_items)
        self.num_items = num_items
        self.num_menu_items = num_menu_items

    def forward(self, x, valuations):
        """
        Args:
            x: 输入常数1 [batch_size, 1]
            valuations: 买家估值 [batch_size, num_items]

        Returns:
            choices: 买家选择 [batch_size, num_menu_items]
            allocations: 实际分配 [batch_size, num_items]
            payments: 实际支付 [batch_size]
            P: 分配矩阵 [batch_size, num_items, num_menu_items]
            t: 支付向量 [batch_size, num_menu_items]
        """
        # 机制网络输出
        P, t = self.mechanism_net(x)

        # 买家网络输出
        choices, utilities = self.buyer_net(valuations, P, t)

        # 计算实际分配和支付
        batch_size = valuations.shape[0]
        allocations = torch.zeros(batch_size, self.num_items, device=valuations.device)
        for i in range(self.num_items):
            for k in range(self.num_menu_items):
                allocations[:, i] += choices[:, k] * P[:, i, k]

        payments = torch.sum(choices * t, dim=1)

        return choices, allocations, payments, P, t