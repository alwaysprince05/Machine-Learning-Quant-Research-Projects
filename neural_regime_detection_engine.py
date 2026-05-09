import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# --- Synthetic Market Data Generation ---
def generate_synthetic_market_data(n_steps=2000, seed=42):
    np.random.seed(seed)
    regimes = ['trend', 'mean_revert', 'panic', 'crash', 'high_vol']
    regime_lengths = np.random.choice([200, 300, 400], size=10)
    regime_types = np.random.choice(regimes, size=len(regime_lengths), replace=True)
    data = []
    regime_labels = []
    t = 0
    for reg, length in zip(regime_types, regime_lengths):
        mu, sigma = 0, 1
        if reg == 'trend':
            mu, sigma = 0.05, 0.8
        elif reg == 'mean_revert':
            mu, sigma = 0, 0.5
        elif reg == 'panic':
            mu, sigma = -0.1, 2.0
        elif reg == 'crash':
            mu, sigma = -0.2, 3.0
        elif reg == 'high_vol':
            mu, sigma = 0, 2.5
        segment = np.cumsum(np.random.normal(mu, sigma, length))
        if len(data) > 0:
            segment += data[-1]
        data.extend(segment)
        regime_labels.extend([reg]*length)
        t += length
    data = np.array(data[:n_steps])
    regime_labels = np.array(regime_labels[:n_steps])
    returns = np.diff(data, prepend=data[0])
    volatility = pd.Series(returns).rolling(20).std().fillna(0).values
    df = pd.DataFrame({'price': data, 'returns': returns, 'volatility': volatility, 'regime_true': regime_labels})
    return df

# --- Feature Engineering ---
def create_features(df):
    df['ma_10'] = df['price'].rolling(10).mean()
    df['ma_50'] = df['price'].rolling(50).mean()
    df['momentum'] = df['price'] - df['price'].shift(10)
    df['vol_rolling'] = df['returns'].rolling(20).std()
    df = df.fillna(0)
    return df

# --- Regime Detection Models ---
def fit_gmm(X, n_components=5):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    return gmm

def fit_kmeans(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

def fit_pca(X, n_components=3):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

# --- Regime Classification & Confidence ---
def classify_regimes_gmm(gmm, X):
    labels = gmm.predict(X)
    probs = gmm.predict_proba(X)
    confidence = probs.max(axis=1)
    return labels, confidence, probs

def classify_regimes_kmeans(kmeans, X):
    labels = kmeans.predict(X)
    # KMeans does not provide probabilities, so use distance to centroid
    distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)
    confidence = 1 - (distances / (distances.max() + 1e-6))
    return labels, confidence

# --- Visualization Utilities ---
def set_dark_theme():
    plt.style.use('dark_background')
    sns.set_theme(style='darkgrid', rc={'axes.facecolor':'#181818', 'figure.facecolor':'#181818'})
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = '#181818'
    plt.rcParams['axes.facecolor'] = '#181818'

# --- Dashboard Plotting ---
def plot_dashboard(df, X_pca, gmm, gmm_labels, gmm_conf, gmm_probs, kmeans, kmeans_labels, kmeans_conf):
    set_dark_theme()
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 3)

    # 1. Regime transition heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    transition_matrix = np.zeros((5, 5))
    for i in range(1, len(gmm_labels)):
        transition_matrix[gmm_labels[i-1], gmm_labels[i]] += 1
    sns.heatmap(transition_matrix, annot=True, fmt='.0f', cmap='mako', ax=ax1)
    ax1.set_title('Regime Transition Heatmap')
    ax1.set_xlabel('To Regime')
    ax1.set_ylabel('From Regime')

    # 2. 3D volatility-regime surface
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    surf = ax2.plot_trisurf(np.arange(len(df)), df['volatility'], gmm_labels, cmap=cm.viridis, linewidth=0.2)
    ax2.set_title('3D Volatility-Regime Surface')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility')
    ax2.set_zlabel('Regime')

    # 3. Confidence landscape
    ax3 = fig.add_subplot(gs[0, 2])
    sns.lineplot(x=np.arange(len(gmm_conf)), y=gmm_conf, ax=ax3, color='cyan')
    ax3.set_title('GMM Regime Confidence')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Confidence')

    # 4. PCA projection scatter
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='Spectral', alpha=0.7)
    ax4.set_title('PCA Projection (GMM Regimes)')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax4, label='Regime')

    # 5. Hidden-state transition matrix
    ax5 = fig.add_subplot(gs[1, 1])
    sns.heatmap(gmm.means_, annot=True, fmt='.2f', cmap='rocket', ax=ax5)
    ax5.set_title('GMM Regime Means (Hidden States)')
    ax5.set_xlabel('Feature')
    ax5.set_ylabel('Regime')

    # 6. Regime probability distributions
    ax6 = fig.add_subplot(gs[1, 2])
    for i in range(gmm_probs.shape[1]):
        sns.kdeplot(gmm_probs[:, i], label=f'Regime {i}', ax=ax6)
    ax6.set_title('Regime Probability Distributions')
    ax6.set_xlabel('Probability')
    ax6.set_ylabel('Density')
    ax6.legend()

    # 7. Time vs volatility vs regime 3D plot
    ax7 = fig.add_subplot(gs[2, 0], projection='3d')
    ax7.scatter(np.arange(len(df)), df['volatility'], gmm_labels, c=gmm_labels, cmap='Spectral', alpha=0.7)
    ax7.set_title('Time vs Volatility vs Regime')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('Volatility')
    ax7.set_zlabel('Regime')

    # 8. Monte Carlo regime simulations
    ax8 = fig.add_subplot(gs[2, 1])
    for _ in range(10):
        sim = np.cumsum(np.random.normal(0, 1, len(df)))
        ax8.plot(sim, alpha=0.3, color='orange')
    ax8.set_title('Monte Carlo Regime Simulations')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Simulated Price')

    # 9. Statistical diagnostics panel
    ax9 = fig.add_subplot(gs[2, 2])
    sns.histplot(df['volatility'], bins=30, color='magenta', ax=ax9, kde=True)
    ax9.set_title('Volatility Distribution')
    ax9.set_xlabel('Volatility')
    ax9.set_ylabel('Frequency')

    plt.suptitle('Neural Regime Detection Engine — Quantitative Finance Dashboard', fontsize=20, color='white', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# --- Main Pipeline ---
def main():
    # 1. Generate synthetic market data
    df = generate_synthetic_market_data(n_steps=2000)
    df = create_features(df)
    features = ['returns', 'volatility', 'ma_10', 'ma_50', 'momentum', 'vol_rolling']
    X = df[features].values

    # 2. Fit models
    gmm = fit_gmm(X, n_components=5)
    kmeans = fit_kmeans(X, n_clusters=5)
    pca, X_pca = fit_pca(X, n_components=3)

    # 3. Regime classification
    gmm_labels, gmm_conf, gmm_probs = classify_regimes_gmm(gmm, X)
    kmeans_labels, kmeans_conf = classify_regimes_kmeans(kmeans, X)

    # 4. Dashboard visualization
    plot_dashboard(df, X_pca, gmm, gmm_labels, gmm_conf, gmm_probs, kmeans, kmeans_labels, kmeans_conf)

if __name__ == '__main__':

    # Uncomment the following line to run the full dashboard:
    # main()

    # --- Display only the 3D volatility-regime surface plot ---
    def plot_volatility_regime_surface(df, gmm_labels):
        set_dark_theme()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(np.arange(len(df)), df['volatility'], gmm_labels, cmap=cm.viridis, linewidth=0.2)
        ax.set_title('3D Volatility-Regime Surface')
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility')
        ax.set_zlabel('Regime')
        plt.tight_layout()
        plt.savefig('surface_plot.png', dpi=300)
        print('Saved 3D volatility-regime surface plot as surface_plot.png')
        plt.show()

    # Minimal pipeline for just this plot
    df = generate_synthetic_market_data(n_steps=2000)
    df = create_features(df)
    features = ['returns', 'volatility', 'ma_10', 'ma_50', 'momentum', 'vol_rolling']
    X = df[features].values
    gmm = fit_gmm(X, n_components=5)
    gmm_labels, _, _ = classify_regimes_gmm(gmm, X)
    plot_volatility_regime_surface(df, gmm_labels)
