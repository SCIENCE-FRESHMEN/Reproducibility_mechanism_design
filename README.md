# Deep Learning Meets Mechanism Design: Key Results and Novel Applications

This repository implements deep learning approaches for mechanism design as described in the paper "Deep Learning Meets Mechanism Design: Key Results and Some Novel Applications". The code demonstrates how neural networks can learn auction mechanisms that satisfy desirable properties like incentive compatibility, individual rationality, revenue maximization, fairness constraints, and budget balance.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Case Studies](#case-studies)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+ (with CUDA support recommended for better performance)
- Other dependencies listed in requirements.txt

### Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-learning-mechanism-design.git
cd deep-learning-mechanism-design
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
dl_mechanism_design/
├── main.py                     # Main entry point to run all experiments
├── requirements.txt            # Project dependencies
├── results/                    # Output directory for models and plots
│   ├── models/                 # Saved model weights
│   └── plots/                  # Training visualizations
├── models/                     # Neural network implementations
│   ├── __init__.py
│   ├── rochet_net.py           # RochetNet implementation
│   ├── regret_net.py           # RegretNet implementation
│   ├── myerson_net.py          # MyersonNet implementation
│   ├── menu_net.py             # MenuNet implementation
│   ├── uav_auction.py          # UAV energy management model
│   └── procurement_auction.py  # Agricultural procurement model
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── data_generation.py      # Data generation functions
│   └── evaluation.py           # Evaluation metrics and utilities
└── examples/                   # Example training scripts
    ├── train_rochet.py         # Train RochetNet model
    ├── train_regret.py         # Train RegretNet model
    ├── train_myerson.py        # Train MyersonNet model
    ├── train_uav_case.py       # Train UAV energy management model
    └── train_procurement_case.py # Train procurement auction model
```

## Usage

### Quick Start
Run all experiments with default parameters:
```bash
python main.py
```

### Individual Model Training
You can train individual models:
```bash
# Train RochetNet (single buyer, multiple items)
python examples/train_rochet.py

# Train RegretNet (multiple buyers, multiple items)
python examples/train_regret.py

# Train MyersonNet (single item, multiple buyers)
python examples/train_myerson.py

# Train UAV energy management model
python examples/train_uav_case.py

# Train procurement auction model
python examples/train_procurement_case.py
```

### Custom Training Parameters
Each training script accepts command-line arguments to customize:
- Number of items/buyers
- Number of training samples
- Number of epochs
- Learning rate
- Device (CPU/GPU)

Example:
```bash
python examples/train_regret.py --num-buyers 5 --num-items 3 --epochs 100 --lr 0.001 --device cuda
```

## Models Implemented

### Revenue Maximizing Auctions
- **RochetNet**: Single buyer, multiple items auction that guarantees DSIC and IR constraints through network architecture
- **RegretNet**: Multi-buyer, multi-item auction that approximates DSIC through differentiable regret minimization
- **MyersonNet**: Single-item, multi-buyer optimal auction based on virtual valuations
- **RegretFormer**: Transformer-based extension of RegretNet with better generalization
- **Budgeted RegretNet**: Extension with budget constraints for buyers

### Welfare Maximizing Auctions
- **CNN-based Social Welfare Optimization**: Auctions that maximize social welfare with minimal economic burden
- **MLCA (Machine Learning Combinatorial Auction)**: Iterative combinatorial auction with preference learning

### Fairness-Based Auctions
- **ProportionNet**: Balances fairness and revenue in auction design
- **EEF1-NN**: Efficient and envy-free up to one good allocation through neural networks

### Budget-Balanced Auctions
- **Redistribution Mechanism**: VCG payments redistribution while maintaining DSIC

### Application-Specific Models
- **UAV Energy Management**: Energy allocation in UAV-assisted vehicular networks
- **Mobile Network Resource Allocation**: Subchannel and power allocation in wireless networks
- **Agricultural Input Procurement**: Volume discount procurement auction for farmer collectives

## Case Studies

The repository implements three novel applications from the paper:

1. **Efficient Energy Management in UAV-Assisted Vehicular Networks**
   - Models energy allocation for UAVs creating digital twins in metaverse environments
   - Balances energy consumption with data collection rewards

2. **Resource Allocation in Mobile Networks**
   - Implements auction-based resource allocation for Mobile Network Virtual Operators (MNVOs)
   - Optimizes subchannel and power allocation to maximize revenue

3. **Volume Discount Procurement Auction for Agricultural Inputs**
   - Models procurement auctions for Farmer Producer Organizations (FPOs)
   - Handles volume discounts and business constraints

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

Please ensure your code follows PEP8 style guidelines and includes appropriate documentation.

## Citation

If you use this code in your research, please cite the original paper:

```
@article{sankar2024deep,
  title={Deep Learning Meets Mechanism Design: Key Results and Some Novel Applications},
  author={Sankar, V. Udaya and Rao, Vishisht Srihari and Narahari, Y.},
  journal={arXiv preprint arXiv:2401.05683},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This implementation is for educational and research purposes only. The models may require hyperparameter tuning and architecture adjustments for specific real-world applications.
