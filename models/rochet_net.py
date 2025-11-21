import torch
import torch.nn as nn


class RochetNet(nn.Module):
    """RochetNet: 单买家多物品的DSIC拍卖"""

    def __init__(self, num_items, num_hidden=10):
        """
        Args:
            num_items: 物品数量
            num_hidden: 隐藏层神经元数量
        """
        super(RochetNet, self).__init__()
        self.num_items = num_items
        self.num_hidden = num_hidden

        # 根据Theorem 3，我们需要w_k在[0,1]范围内，w_0,k为任意实数
        # 使用参数矩阵W: [num_hidden, num_items]和向量w0: [num_hidden]
        self.W = nn.Parameter(torch.rand(num_hidden, num_items))  # 初始化为[0,1]范围
        self.w0 = nn.Parameter(torch.randn(num_hidden))  # 初始化为标准正态分布

    def forward(self, valuations):
        """
        Args:
            valuations: 买家对每个物品的估值 [batch_size, num_items]

        Returns:
            allocations: 分配概率 [batch_size, num_items]
            payments: 支付金额 [batch_size]
        """
        batch_size = valuations.shape[0]
        device = valuations.device

        # 计算每个隐藏单元的输出: w_k^T * v + w_0,k
        # valuations: [batch_size, num_items]
        # W: [num_hidden, num_items]
        # hidden_outputs: [batch_size, num_hidden]
        hidden_outputs = torch.matmul(valuations, self.W.t()) + self.w0

        # 确保W在[0,1]范围内（根据Theorem 3）
        self.W.data = torch.clamp(self.W.data, 0, 1)

        # 应用max操作和0比较 (Theorem 3)
        # 找到每个样本的最大隐藏值及其索引
        max_hidden_values, max_indices = torch.max(hidden_outputs, dim=1)

        # 计算效用：取最大隐藏值和0的最大值
        utility = torch.maximum(max_hidden_values, torch.zeros(batch_size, device=device))

        # 计算分配概率 - 效用函数的梯度 (Theorem 3)
        allocations = torch.zeros(batch_size, self.num_items, device=device)

        # 为每个样本分配对应的权重向量
        for i in range(batch_size):
            if max_hidden_values[i] > 0:
                allocations[i] = self.W[max_indices[i]]

        # 确保分配概率在[0,1]范围内
        allocations = torch.clamp(allocations, 0, 1)

        # 计算支付: ∇u^T * v - u (Theorem 3)
        payments = torch.sum(allocations * valuations, dim=1) - utility

        # 确保支付非负
        payments = torch.clamp(payments, min=0)

        return allocations, payments