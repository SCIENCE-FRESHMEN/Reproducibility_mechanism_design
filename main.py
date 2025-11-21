from examples.train_rochet import train_rochet_net
from examples.train_regret import train_regret_net
from examples.train_myerson import train_myerson_net
from examples.train_uav_case import train_uav_auction
from examples.train_procurement_case import train_procurement_auction
import torch
import os

if __name__ == "__main__":
    # 创建结果目录
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 依次训练各个模型
    models = {}

    print("\n=== 训练 RochetNet (单买家3物品) ===")
    models['rochet'] = train_rochet_net(num_items=3, epochs=50, device=device)

    print("\n=== 训练 RegretNet (3买家2物品) ===")
    models['regret'] = train_regret_net(num_buyers=3, num_items=2, epochs=50, device=device)

    print("\n=== 训练 MyersonNet (3买家1物品) ===")
    models['myerson'] = train_myerson_net(num_buyers=3, epochs=50, device=device)

    print("\n=== 训练 UAV 能源管理案例 (5无人机) ===")
    models['uav'] = train_uav_auction(num_uavs=5, epochs=50, device=device)

    print("\n=== 训练 采购拍卖案例 (10供应商3批量区间) ===")
    models['procurement'] = train_procurement_auction(num_suppliers=10, num_brackets=3, epochs=50, device=device)

    print("\n所有模型训练完成！结果已保存到 results/ 目录。")