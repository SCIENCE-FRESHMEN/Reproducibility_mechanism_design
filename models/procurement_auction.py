import torch
import torch.nn as nn


class ProcurementAuction(nn.Module):
    """批量折扣采购拍卖模型"""

    def __init__(self, num_suppliers, num_brackets):
        super(ProcurementAuction, self).__init__()
        self.num_suppliers = num_suppliers
        self.num_brackets = num_brackets

        # RegretNet架构
        self.allocation_net = nn.Sequential(
            nn.Linear(num_suppliers * num_brackets, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_suppliers)
        )

        self.payment_net = nn.Sequential(
            nn.Linear(num_suppliers * num_brackets, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_suppliers)
        )

    def forward(self, bids):
        """
        Args:
            bids: 供应商出价，形状为[batch_size, num_suppliers, num_brackets]
                  bids[i, j, k]表示供应商j在区间k的出价

        Returns:
            allocations: 采购分配，形状为[batch_size, num_suppliers]
            payments: 支付金额，形状为[batch_size, num_suppliers]
        """
        batch_size = bids.shape[0]
        device = bids.device

        # 创建bids的副本，避免原地修改输入
        bids_clone = bids.clone().detach()

        # 确保批量折扣：每个区间的出价不大于前一个区间
        for k in range(1, self.num_brackets):
            # 使用torch.minimum创建新张量，而不是原地修改
            previous_bracket = bids_clone[:, :, k - 1]
            current_bracket = bids_clone[:, :, k]
            bids_clone[:, :, k] = torch.minimum(current_bracket, previous_bracket)

        # 展平输入
        flat_bids = bids_clone.view(batch_size, -1)

        # 计算分配
        allocations = self.allocation_net(flat_bids)
        allocations = torch.softmax(allocations, dim=1)  # 归一化为分配比例

        # 计算支付
        payments = self.payment_net(flat_bids)
        payments = torch.abs(payments)  # 确保支付非负

        return allocations, payments