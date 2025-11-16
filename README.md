# Monte Carlo Sensex Simulator (clean)

Simple, modern-minimal project that runs Geometric Brownian Motion simulations on historical index prices.

## Structure
- `monte_carlo.py` - clean simulation module and CLI
- `data/` - put your historical CSV (Date index, Close column)
- `outputs/` - generated charts and summary CSV

## Usage
```bash
python monte_carlo.py --csv data/sensex.csv --sims 5000 --out outputs
```

## Requirements
- Python 3.8+
- numpy, pandas, matplotlib
