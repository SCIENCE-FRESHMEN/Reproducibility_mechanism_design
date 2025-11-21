import torch


def compute_utility(allocations, payments, valuations):
    """计算买家的效用: u = v*p - t"""
    # allocations: [batch_size, num_buyers, num_items]
    # payments: [batch_size, num_buyers]
    # valuations: [batch_size, num_buyers, num_items]
    value = torch.sum(allocations * valuations, dim=2)  # 每个买家获得的总价值
    utility = value - payments
    return utility


def compute_regret(model, valuations, num_samples=10):
    """计算买家的事后遗憾"""
    batch_size, num_buyers, num_items = valuations.shape
    device = valuations.device

    # 存储每个买家的最大遗憾
    regrets = torch.zeros(batch_size, num_buyers, device=device)

    allocations, payments = model(valuations)

    # 为每个买家计算遗憾
    for i in range(num_buyers):
        # 获取真实出价下的效用
        true_utility = compute_utility(allocations, payments, valuations)[:, i]

        # 生成多个非真实出价
        max_utility = true_utility.clone()

        for _ in range(num_samples):
            # 生成与真实估值不同的出价
            fake_bids = valuations.clone().detach()
            # 为买家i生成随机出价
            fake_bids[:, i] = torch.rand(batch_size, num_items, device=device)

            # 计算非真实出价下的分配和支付
            fake_allocations, fake_payments = model(fake_bids)

            # 计算非真实出价下的效用
            fake_utility = compute_utility(fake_allocations, fake_payments, valuations)[:, i]

            # 更新最大效用
            max_utility = torch.max(max_utility, fake_utility)

        # 计算遗憾
        regrets[:, i] = max_utility - true_utility

    return regrets


def compute_procurement_regret(model, bids):
    """计算采购拍卖中供应商的遗憾"""
    batch_size, num_suppliers, num_brackets = bids.shape
    device = bids.device

    # 当前分配和支付
    allocations, payments = model(bids)

    # 供应商的效用：-payment（采购拍卖中供应商希望最大化收入）
    true_utilities = -payments

    # 计算最大可能效用
    max_utilities = true_utilities.clone().detach()  # 断开计算图

    for i in range(num_suppliers):
        for _ in range(5):  # 采样5个替代出价
            # 创建bids的副本
            fake_bids = bids.clone().detach()

            # 生成替代出价 - 避免原地修改
            multiplier = 0.8 + torch.rand(batch_size, num_brackets, device=device) * 0.4
            fake_bids[:, i] = bids[:, i] * multiplier

            # 确保批量折扣
            for k in range(1, num_brackets):
                # 创建新张量，而不是原地修改
                fake_bids[:, i, k] = torch.minimum(fake_bids[:, i, k], fake_bids[:, i, k - 1])

            # 获取替代分配和支付
            fake_allocs, fake_pays = model(fake_bids)

            # 计算替代效用
            fake_utilities = -fake_pays[:, i].detach()

            # 更新最大效用 - 创建新张量
            current_max = max_utilities[:, i].clone().detach()
            new_max = torch.maximum(current_max, fake_utilities.detach())
            max_utilities[:, i] = new_max

    # 计算遗憾 - 断开计算图
    regret = (max_utilities - true_utilities).detach()
    return torch.relu(regret)  # 确保遗憾非负


def compute_envy(allocations, payments):
    """计算嫉妒度（供应商之间的不公平性）"""
    batch_size, num_suppliers = allocations.shape
    device = allocations.device

    # 计算每个供应商的单位成本
    unit_costs = torch.zeros(batch_size, num_suppliers, device=device)

    for i in range(num_suppliers):
        denom = allocations[:, i] + 1e-8  # 避免除零
        cost_per_unit = payments[:, i] / denom
        unit_costs[:, i] = torch.where(
            allocations[:, i] > 0,
            cost_per_unit,
            torch.zeros(batch_size, device=device)
        )

    # 计算平均单位成本
    avg_unit_cost = torch.mean(unit_costs, dim=1, keepdim=True)

    # 嫉妒度：单位成本与平均的差异
    envy = torch.abs(unit_costs - avg_unit_cost)
    return torch.mean(envy, dim=1)