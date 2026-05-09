import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import scipy.stats as stats

# --- Synthetic Data Generation ---
def generate_synthetic_data(n_assets=100, n_periods=252, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_periods)
    tickers = [f"Asset_{i+1}" for i in range(n_assets)]
    prices = np.cumprod(1 + np.random.normal(0, 0.01, (n_periods, n_assets)), axis=0) * 100
    df = pd.DataFrame(prices, index=dates, columns=tickers)
    return df

# --- Alpha Factor Engineering ---
def compute_factors(df):
    factors = {}
    returns = df.pct_change().fillna(0)
    # Momentum: past 20-day return
    factors['momentum'] = returns.rolling(20).mean()
    # Volatility: rolling 20-day std
    factors['volatility'] = returns.rolling(20).std()
    # Mean reversion: negative 5-day return
    factors['mean_reversion'] = -returns.rolling(5).mean()
    # Liquidity imbalance: rolling 20-day range / std
    factors['liquidity_imbalance'] = (df.rolling(20).max() - df.rolling(20).min()) / (returns.rolling(20).std() + 1e-6)
    # Synthetic sentiment: random walk with drift (per asset)
    factors['synthetic_sentiment'] = np.apply_along_axis(lambda x: np.cumsum(np.random.normal(0, 0.01, len(x))), 0, df.values)
    # Order flow pressure: random normal (per asset)
    factors['order_flow_pressure'] = np.random.normal(0, 1, df.shape)
    # Combine into DataFrame
    factor_df = pd.concat([pd.DataFrame(factors[f], index=df.index, columns=df.columns) for f in factors], axis=1, keys=factors.keys())
    return factor_df

# --- ML Regression & Feature Importance ---
def ml_regression_and_importance(factor_df, returns, model_type='rf'):
    # Flatten for ML
    X = factor_df.dropna().values.reshape(-1, len(factor_df.columns.levels[0]))
    y = returns.shift(-1).dropna().values.flatten()
    min_len = min(len(X), len(y))
    X, y = X[:min_len], y[:min_len]
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RidgeCV(alphas=np.logspace(-3, 3, 7))
    model.fit(X, y)
    y_pred = model.predict(X)
    importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.coef_
    return y, y_pred, importance, model

# --- Dimensionality Reduction ---
def run_dimensionality_reduction(factor_df):
    X = factor_df.dropna().values.reshape(-1, len(factor_df.columns.levels[0]))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_tsne = None  # t-SNE disabled for speed
    return X_pca, X_tsne

# --- Alpha Decay Simulation ---
def simulate_alpha_decay(factor, half_life=20):
    decay = np.exp(-np.arange(len(factor)) / half_life)
    return factor * decay[:, None]

def plot_dashboard(factor_df, y, y_pred, importance, X_pca, X_tsne, returns):
    sns.set_theme(style="darkgrid", palette="dark")
    plt.rcParams['axes.facecolor'] = '#181a1b'
    plt.rcParams['figure.facecolor'] = '#181a1b'
    plt.rcParams['text.color'] = '#e0e0e0'
    plt.rcParams['axes.labelcolor'] = '#e0e0e0'
    plt.rcParams['xtick.color'] = '#e0e0e0'
    plt.rcParams['ytick.color'] = '#e0e0e0'
    plt.rcParams['font.family'] = 'monospace'
    import os
    os.makedirs("dashboard_outputs", exist_ok=True)

    # Feature importance
    plt.figure(figsize=(7, 5))
    plt.bar(factor_df.columns.levels[0], importance)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("dashboard_outputs/feature_importance.png")
    plt.close()

    # Predicted vs Actual
    plt.figure(figsize=(7, 5))
    plt.scatter(y, y_pred, alpha=0.3, color='cyan')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title('Predicted vs Actual Returns')
    plt.tight_layout()
    plt.savefig("dashboard_outputs/predicted_vs_actual.png")
    plt.close()

    # Residuals
    plt.figure(figsize=(7, 5))
    residuals = y - y_pred
    sns.histplot(residuals, bins=50, color='orange')
    plt.title('Residual Error Distribution')
    plt.tight_layout()
    plt.savefig("dashboard_outputs/residual_error_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(7, 5))
    corr = factor_df.xs(factor_df.columns.levels[0][0], axis=1, level=0).corr()
    sns.heatmap(corr, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig("dashboard_outputs/correlation_heatmap.png")
    plt.close()

    # 3D Factor Landscape
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    X = X_pca[:, 0]
    Y = X_pca[:, 1]
    Z = y_pred[:len(X)]
    ax.plot_trisurf(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.set_title('3D Factor Landscape (PCA)')
    plt.tight_layout()
    plt.savefig("dashboard_outputs/3d_factor_landscape.png")
    plt.show()

    # Signal strength contour
    plt.figure(figsize=(7, 5))
    sns.kdeplot(x=X_pca[:, 0], y=X_pca[:, 1], fill=True, cmap='mako')
    plt.title('Signal Strength Contour (PCA)')
    plt.tight_layout()
    plt.savefig("dashboard_outputs/signal_strength_contour.png")
    plt.close()

    # PnL Simulation
    plt.figure(figsize=(7, 5))
    pnl = np.cumsum(y_pred)
    plt.plot(pnl, color='lime')
    plt.title('PnL Simulation')
    plt.tight_layout()
    plt.savefig("dashboard_outputs/pnl_simulation.png")
    plt.close()

    # Monte Carlo Simulation
    plt.figure(figsize=(7, 5))
    for _ in range(10):
        mc = np.cumsum(np.random.choice(y_pred, size=len(y_pred), replace=True))
        plt.plot(mc, alpha=0.3)
    plt.title('Monte Carlo Factor Simulations')
    plt.tight_layout()
    plt.savefig("dashboard_outputs/monte_carlo_simulation.png")
    plt.close()

    # Statistical summary
    plt.figure(figsize=(7, 5))
    stats_text = f"MSE: {mean_squared_error(y, y_pred):.4f}\nR2: {r2_score(y, y_pred):.4f}\nMean: {np.mean(y_pred):.4f}\nStd: {np.std(y_pred):.4f}"
    plt.text(0.1, 0.5, stats_text, fontsize=14, color='#e0e0e0', fontfamily='monospace')
    plt.axis('off')
    plt.title('Statistical Summary')
    plt.tight_layout()
    plt.savefig("dashboard_outputs/statistical_summary.png")
    plt.close()
    print("All dashboard plots saved to the 'dashboard_outputs' folder.")

# --- Main Entrypoint ---
def main():
    df = generate_synthetic_data(n_assets=10, n_periods=50)
    factor_df = compute_factors(df)
    returns = df.pct_change().fillna(0)
    y, y_pred, importance, model = ml_regression_and_importance(factor_df, returns)
    X_pca, X_tsne = run_dimensionality_reduction(factor_df)
    plot_dashboard(factor_df, y, y_pred, importance, X_pca, X_tsne, returns)

if __name__ == "__main__":
    main()
