# Monte Carlo Sensex Simulator (clean)
<p align="left">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" />
  <img src="https://img.shields.io/github/license/amanjot-a/monte-carlo-sensex" />
  <img src="https://img.shields.io/github/last-commit/amanjot-a/monte-carlo-sensex" />
  <img src="https://img.shields.io/github/repo-size/amanjot-a/monte-carlo-sensex" />
</p>

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
