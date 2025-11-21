import torch
import torch.nn as nn


class VirtualValueNetwork(nn.Module):
    """虚拟价值函数网络，用于MyersonNet"""

    def __init__(self, num_buyers, hidden_size=50):
        super(VirtualValueNetwork, self).__init__()
        self.num_buyers = num_buyers

        # 为每个买家创建一个虚拟价值函数网络
        # 使用论文中描述的min-max结构 (Eq. 24)
        self.K = 2  # 组数
        self.J = 3  # 每组的线性函数数

        # 创建参数: w_ki,j 和 beta_ki,j
        self.w_params = nn.Parameter(torch.randn(num_buyers, self.K, self.J))
        self.beta_params = nn.Parameter(torch.randn(num_buyers, self.K, self.J))

    def forward(self, bids):
        """
        Args:
            bids: 买家出价 [batch_size, num_buyers]

        Returns:
            virtual_values: 虚拟价值 [batch_size, num_buyers]
        """
        batch_size = bids.shape[0]
        virtual_values = torch.zeros(batch_size, self.num_buyers, device=bids.device)

        for i in range(self.num_buyers):
            # 获取买家i的出价
            bid_i = bids[:, i].unsqueeze(1)  # [batch_size, 1]

            # 计算线性函数: w_ki,j * bid_i + beta_ki,j
            # w_params[i]: [K, J], beta_params[i]: [K, J]
            w = torch.exp(self.w_params[i])  # 确保w为正 (论文建议)
            linear_funcs = torch.matmul(bid_i, w.view(1, self.K * self.J)) + self.beta_params[i].view(1,
                                                                                                      self.K * self.J)

            # 重塑为 [batch_size, K, J]
            linear_funcs = linear_funcs.view(batch_size, self.K, self.J)

            # 应用max over j
            max_over_j = torch.max(linear_funcs, dim=2)[0]  # [batch_size, K]

            # 应用min over k
            virtual_values[:, i] = torch.min(max_over_j, dim=1)[0]  # [batch_size]

        return virtual_values


class SecondPriceAuction(nn.Module):
    """第二价格拍卖机制"""

    def __init__(self):
        super(SecondPriceAuction, self).__init__()

    def forward(self, virtual_values):
        """
        Args:
            virtual_values: 虚拟价值 [batch_size, num_buyers]

        Returns:
            allocations: 分配概率 [batch_size, num_buyers]
            payments: 支付金额 [batch_size, num_buyers]
        """
        batch_size, num_buyers = virtual_values.shape

        # 分配：虚拟价值最高的买家获胜
        max_values, max_indices = torch.max(virtual_values, dim=1)

        # 创建分配矩阵
        allocations = torch.zeros(batch_size, num_buyers, device=virtual_values.device)
        allocations[torch.arange(batch_size), max_indices] = 1.0

        # 支付：第二高的虚拟价值
        sorted_values, _ = torch.sort(virtual_values, descending=True, dim=1)
        second_highest = sorted_values[:, 1] if num_buyers > 1 else torch.zeros(batch_size,
                                                                                device=virtual_values.device)

        # 支付矩阵
        payments = torch.zeros(batch_size, num_buyers, device=virtual_values.device)
        payments[torch.arange(batch_size), max_indices] = torch.clamp(second_highest, min=0)

        return allocations, payments


class MyersonNet(nn.Module):
    """MyersonNet: 单物品多买家的最优拍卖"""

    def __init__(self, num_buyers):
        super(MyersonNet, self).__init__()
        self.virtual_value_net = VirtualValueNetwork(num_buyers)
        self.second_price_auction = SecondPriceAuction()

    def forward(self, bids):
        """
        Args:
            bids: 买家出价 [batch_size, num_buyers]

        Returns:
            allocations: 分配概率 [batch_size, num_buyers]
            payments: 支付金额 [batch_size, num_buyers]
        """
        # 计算虚拟价值
        virtual_values = self.virtual_value_net(bids)

        # 应用第二价格拍卖
        allocations, payments = self.second_price_auction(virtual_values)

        return allocations, payments