"""monte_carlo.py

Clean, minimal Monte Carlo simulation for index prices using Geometric Brownian Motion.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_prices(csv_path):
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    if 'Close' in df.columns:
        prices = df['Close'].dropna()
    else:
        prices = df.select_dtypes('number').iloc[:,0].dropna()
    return prices

def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def simulate_gbm(S0, mu, sigma, days, n_sims, seed=None):
    rng = np.random.default_rng(seed)
    dt = 1/252
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    increments = rng.normal(loc=drift, scale=vol, size=(days, n_sims))
    log_paths = np.cumsum(increments, axis=0)
    paths = S0 * np.exp(log_paths)
    return paths

def run_simulation_from_csv(csv_path, horizons_days, n_sims=1000, output_dir='outputs', seed=0):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prices = load_prices(csv_path)
    returns = compute_log_returns(prices)
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    S0 = float(prices.iloc[-1])

    summary = {}
    for days in horizons_days:
        paths = simulate_gbm(S0, mu, sigma, days, n_sims, seed=seed)
        horizon_vals = paths[-1, :]
        pct_returns = (horizon_vals - S0) / S0
        summary[days] = {
            'median_return': float(np.median(pct_returns)),
            'mean_return': float(np.mean(pct_returns)),
            'std_return': float(np.std(pct_returns)),
            'pct_5': float(np.percentile(pct_returns, 5)),
            'pct_95': float(np.percentile(pct_returns, 95))
        }
        plt.figure()
        plt.hist(pct_returns, bins=50)
        plt.title(f'Distribution of returns after {days} days')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(out / f'dist_{days}d.png')
        plt.close()

    pd.DataFrame(summary).T.to_csv(out / 'mc_summary.csv')
    return summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Monte Carlo GBM simulations from price CSV')
    parser.add_argument('--csv', required=True, help='CSV with historical prices (date,indexed) and Close column or first numeric column')
    parser.add_argument('--sims', type=int, default=5000)
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()
    run_simulation_from_csv(args.csv, horizons_days=[21, 63, 126, 252], n_sims=args.sims, output_dir=args.out)
