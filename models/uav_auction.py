import torch
import torch.nn as nn


class UAVAuctionModel(nn.Module):
    """无人机辅助车辆网络能源管理拍卖模型"""

    def __init__(self, num_uavs):
        super(UAVAuctionModel, self).__init__()
        self.num_uavs = num_uavs

        # 用于单充电站场景的MyersonNet变体
        self.virtual_value_net = nn.Sequential(
            nn.Linear(2, 64),  # 输入：[估值, 感知率]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute_reward(self, sensing_rates, D, theta, eta):
        """
        计算无人机奖励 (Eq. 91)
        r_i = D^eta(1+theta) * lambda_i / sum_k lambda_k
        """
        total_sensing = torch.sum(sensing_rates, dim=1, keepdim=True)
        rewards = (D ** eta) * (1 + theta) * (sensing_rates / total_sensing)
        return rewards

    def compute_valuation(self, rewards, total_energy, remaining_energy, alpha):
        """
        计算无人机估值 (Eq. 92)
        v_i = (1 + r_i * E_i^Tot / E_i^R)^(1-alpha) / (1-alpha)
        """
        efficiency = rewards * (total_energy / remaining_energy)
        valuation = (1 + efficiency) ** (1 - alpha) / (1 - alpha)
        return valuation

    def forward(self, sensing_rates, total_energies, remaining_energies, D=5, theta=0.1, eta=0.5, alpha=0.5):
        """
        Args:
            sensing_rates: 无人机感知率 [batch_size, num_uavs]
            total_energies: 无人机总能耗 [batch_size, num_uavs]
            remaining_energies: 无人机剩余能量 [batch_size, num_uavs]
            D: 数字孪生数量
            theta: 衰减率
            eta: 同步因子
            alpha: 缩放因子

        Returns:
            allocations: 充电分配 [batch_size, num_uavs]
            payments: 支付金额 [batch_size, num_uavs]
        """
        batch_size = sensing_rates.shape[0]

        # 计算奖励
        rewards = self.compute_reward(sensing_rates, D, theta, eta)

        # 计算估值
        valuations = self.compute_valuation(rewards, total_energies, remaining_energies, alpha)

        # 计算虚拟价值
        # 输入: [valuation, sensing_rate]
        input_features = torch.stack([valuations, sensing_rates], dim=2)
        batch_size, num_uavs, _ = input_features.shape
        flat_input = input_features.view(batch_size * num_uavs, 2)
        flat_virtual_values = self.virtual_value_net(flat_input)
        virtual_values = flat_virtual_values.view(batch_size, num_uavs)

        # 应用第二价格拍卖
        max_values, max_indices = torch.max(virtual_values, dim=1)

        # 分配：虚拟价值最高的无人机获得充电
        allocations = torch.zeros(batch_size, self.num_uavs, device=sensing_rates.device)
        allocations[torch.arange(batch_size), max_indices] = 1.0

        # 支付：第二高的虚拟价值
        sorted_values, _ = torch.sort(virtual_values, descending=True, dim=1)
        second_highest = sorted_values[:, 1] if self.num_uavs > 1 else torch.zeros(batch_size,
                                                                                   device=sensing_rates.device)

        # 支付矩阵
        payments = torch.zeros(batch_size, self.num_uavs, device=sensing_rates.device)
        payments[torch.arange(batch_size), max_indices] = torch.clamp(second_highest, min=0)

        return allocations, payments, valuations, virtual_values