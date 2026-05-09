import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import norm, gaussian_kde
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    'axes.facecolor': '#181a1b',
    'figure.facecolor': '#181a1b',
    'axes.labelcolor': '#c7c7c7',
    'xtick.color': '#c7c7c7',
    'ytick.color': '#c7c7c7',
    'grid.color': '#222',
    'axes.edgecolor': '#444',
    'font.family': 'monospace',
    'font.size': 10
})

def generate_order_book(n_steps=2000, n_levels=10, shock_prob=0.02, seed=42):
    """
    Generate synthetic order book data with liquidity shocks and volatility spikes.
    """
    np.random.seed(seed)
    base_price = 100.0
    prices = [base_price]
    spread = 0.02
    order_book = []
    liquidity = np.ones((n_steps, n_levels*2)) * 1000  # bid/ask depth
    buy_pressure = np.zeros(n_steps)
    sell_pressure = np.zeros(n_steps)
    volatility = np.zeros(n_steps)
    stress_event = np.zeros(n_steps)

    for t in range(1, n_steps):
        # Simulate normal price movement
        ret = np.random.normal(0, 0.01)
        # Liquidity shock event
        if np.random.rand() < shock_prob:
            shock = np.random.normal(0, 0.2)
            ret += shock
            spread *= np.random.uniform(1.5, 3.0)
            liquidity[t, :] = liquidity[t-1, :] * np.random.uniform(0.2, 0.5)
            stress_event[t] = 1
        else:
            spread = max(0.01, spread * np.random.uniform(0.98, 1.02))
            liquidity[t, :] = liquidity[t-1, :] * np.random.uniform(0.98, 1.02)
        prices.append(prices[-1] * np.exp(ret))
        # Simulate buy/sell pressure
        buy_pressure[t] = np.random.normal(0, 1) + (stress_event[t] * np.random.uniform(5, 15))
        sell_pressure[t] = np.random.normal(0, 1) - (stress_event[t] * np.random.uniform(5, 15))
        volatility[t] = abs(ret) + stress_event[t] * np.random.uniform(0.05, 0.2)

    prices = np.array(prices)
    mid = prices
    bid = mid - spread/2
    ask = mid + spread/2
    data = pd.DataFrame({
        'mid': mid,
        'bid': bid,
        'ask': ask,
        'spread': ask - bid,
        'volatility': volatility,
        'buy_pressure': buy_pressure,
        'sell_pressure': sell_pressure,
        'stress_event': stress_event
    })
    # Add order book depth columns
    for i in range(n_levels):
        data[f'bid_depth_{i+1}'] = liquidity[:, i]
        data[f'ask_depth_{i+1}'] = liquidity[:, n_levels + i]
    return data

# --- Feature Engineering ---
def create_features(df):
    """
    Create features for ML models.
    """
    features = df.copy()
    features['imbalance'] = (features['buy_pressure'] - features['sell_pressure'])
    features['depth_imbalance'] = features[[f'bid_depth_{i+1}' for i in range(5)]].sum(axis=1) - \
                                  features[[f'ask_depth_{i+1}' for i in range(5)]].sum(axis=1)
    features['future_return'] = np.log(features['mid'].shift(-5) / features['mid'])
    features['future_volatility'] = features['volatility'].rolling(5).mean().shift(-5)
    features = features.dropna()
    return features

# --- ML Modeling ---
def run_ml_models(features):
    """
    Fit Ridge, Lasso, and XGBoost-style (Ridge with high alpha) regressors.
    """
    X = features[[
        'spread', 'volatility', 'imbalance', 'depth_imbalance',
        'buy_pressure', 'sell_pressure', 'stress_event'
    ] + [f'bid_depth_{i+1}' for i in range(5)] + [f'ask_depth_{i+1}' for i in range(5)]]
    y = features['future_return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'XGBoost-style': Ridge(alpha=10.0)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'X_test': X_test
        }
    return results, X_train, X_test, y_train, y_test

# --- Monte Carlo Simulation ---
def monte_carlo_simulation(model, X_test, n_sims=1000):
    """
    Run Monte Carlo scenario simulations for stress testing.
    """
    preds = []
    for _ in range(n_sims):
        idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_samp = X_test.iloc[idx]
        preds.append(model.predict(X_samp))
    preds = np.array(preds)
    return preds

# --- Visualization Functions ---
def plot_feature_importance(model, X, ax):
    imp = np.abs(model.coef_)
    idx = np.argsort(imp)
    ax.barh(np.array(X.columns)[idx], imp[idx], color='#00bfae')
    ax.set_title('Feature Importance', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')


def plot_pred_vs_actual(y_test, y_pred, ax):
    ax.scatter(y_test, y_pred, alpha=0.3, color='#ffb300', s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='#888')
    ax.set_title('Predicted vs Actual', fontsize=10)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.grid(True, alpha=0.2)


def plot_residuals(y_test, y_pred, ax):
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=40, kde=True, color='#2979ff', ax=ax)
    ax.set_title('Residual Distribution', fontsize=10)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.2)


def plot_corr_heatmap(features, ax):
    corr = features.corr()
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax, cbar=True)
    ax.set_title('Correlation Matrix', fontsize=10)


def plot_3d_liquidity_stress(features, ax):
    x = features['spread']
    y = features['depth_imbalance']
    z = features['volatility']
    # Only plot if at least 3 unique points
    if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
        surf = ax.plot_trisurf(x, y, z, cmap='viridis', alpha=0.8)
        ax.set_title('3D Liquidity Stress Surface', fontsize=10)
        ax.set_xlabel('Spread')
        ax.set_ylabel('Depth Imbalance')
        ax.set_zlabel('Volatility')
    else:
        # For 3D axes, use annotate for fallback text
        ax.set_title('3D Liquidity Stress Surface', fontsize=10)
        ax.set_xlabel('Spread')
        ax.set_ylabel('Depth Imbalance')
        ax.set_zlabel('Volatility')
        ax.text2D(0.5, 0.5, 'Not enough unique points for 3D plot', color='red',
                 fontsize=10, ha='center', va='center', transform=ax.figure.transFigure)


def plot_3d_volatility_shock(features, ax):
    x = features['imbalance']
    y = features['stress_event']
    z = features['future_volatility']
    # Only plot if at least 3 unique points
    if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
        surf = ax.plot_trisurf(x, y, z, cmap='plasma', alpha=0.8)
        ax.set_title('3D Volatility Shock', fontsize=10)
        ax.set_xlabel('Imbalance')
        ax.set_ylabel('Stress Event')
        ax.set_zlabel('Future Volatility')
    else:
        ax.set_title('3D Volatility Shock', fontsize=10)
        ax.set_xlabel('Imbalance')
        ax.set_ylabel('Stress Event')
        ax.set_zlabel('Future Volatility')
        ax.text2D(0.5, 0.5, 'Not enough unique points for 3D plot', color='red',
                 fontsize=10, ha='center', va='center', transform=ax.figure.transFigure)


def plot_stress_event_cloud(features, ax):
    stress = features[features['stress_event'] > 0]
    ax.scatter(stress['imbalance'], stress['future_return'], c=stress['spread'], cmap='cool', alpha=0.7, s=10)
    ax.set_title('Stress Event Scatter Cloud', fontsize=10)
    ax.set_xlabel('Imbalance')
    ax.set_ylabel('Future Return')
    ax.grid(True, alpha=0.2)


def plot_execution_pressure_heatmap(features, ax):
    heatmap, xedges, yedges = np.histogram2d(features['buy_pressure'], features['sell_pressure'], bins=40)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='magma', aspect='auto', alpha=0.8)
    ax.set_title('Execution Pressure Heatmap', fontsize=10)
    ax.set_xlabel('Buy Pressure')
    ax.set_ylabel('Sell Pressure')

# --- Main Dashboard ---
def main():
    # Generate synthetic data
    df = generate_order_book()
    features = create_features(df)
    results, X_train, X_test, y_train, y_test = run_ml_models(features)
    model = results['Ridge']['model']
    y_pred = results['Ridge']['y_pred']

    # Monte Carlo simulation
    mc_preds = monte_carlo_simulation(model, X_test)
    mc_mean = mc_preds.mean(axis=0)
    mc_std = mc_preds.std(axis=0)

    # --- Only Monte Carlo PnL Distribution Plot ---
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#181a1b')
    pnl = mc_preds.sum(axis=1)
    sns.histplot(pnl, bins=40, kde=True, color='#00bfae', ax=ax)
    ax.set_title('Simulated PnL Distribution (Monte Carlo)', fontsize=14)
    ax.set_xlabel('PnL')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
