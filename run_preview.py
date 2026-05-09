"""
Preview runner — saves all dashboard plots to dashboard_outputs/
without requiring a GUI display window.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

os.makedirs("dashboard_outputs/regime_engine", exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def set_dark_theme():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor':   '#0d1117',
        'figure.facecolor': '#0d1117',
        'axes.labelcolor':  'white',
        'xtick.color':      'white',
        'ytick.color':      'white',
        'axes.edgecolor':   '#444',
        'axes.titlecolor':  'white',
        'grid.color':       '#222',
    })

def generate_data(n_steps=2000, seed=42):
    np.random.seed(seed)
    regimes = ['trend', 'mean_revert', 'panic', 'crash', 'high_vol']
    regime_lengths = np.random.choice([200, 300, 400], size=10)
    regime_types   = np.random.choice(regimes, size=len(regime_lengths), replace=True)
    data, labels = [], []
    for reg, length in zip(regime_types, regime_lengths):
        mu, sigma = {'trend':(0.05,0.8),'mean_revert':(0,0.5),
                     'panic':(-0.1,2.0),'crash':(-0.2,3.0),'high_vol':(0,2.5)}[reg]
        seg = np.cumsum(np.random.normal(mu, sigma, length))
        if data: seg += data[-1]
        data.extend(seg); labels.extend([reg]*length)
    data   = np.array(data[:n_steps])
    labels = np.array(labels[:n_steps])
    returns    = np.diff(data, prepend=data[0])
    volatility = pd.Series(returns).rolling(20).std().fillna(0).values
    df = pd.DataFrame({'price':data,'returns':returns,'volatility':volatility,'regime_true':labels})
    df['ma_10']      = df['price'].rolling(10).mean()
    df['ma_50']      = df['price'].rolling(50).mean()
    df['momentum']   = df['price'] - df['price'].shift(10)
    df['vol_rolling']= df['returns'].rolling(20).std()
    return df.fillna(0)

# ── generate data & fit models ────────────────────────────────────────────────
print("Generating synthetic market data …")
df = generate_data()
features = ['returns','volatility','ma_10','ma_50','momentum','vol_rolling']
X = df[features].values

print("Fitting GMM …")
gmm    = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(X)
gmm_labels = gmm.predict(X)
gmm_probs  = gmm.predict_proba(X)
gmm_conf   = gmm_probs.max(axis=1)

print("Fitting KMeans …")
kmeans         = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
kmeans_labels  = kmeans.predict(X)

print("Fitting PCA …")
pca, X_pca = PCA(n_components=3, random_state=42), None
X_pca = pca.fit_transform(X)

# ── FULL DASHBOARD ────────────────────────────────────────────────────────────
print("Rendering full dashboard …")
set_dark_theme()
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('#0d1117')
gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

ACCENT  = '#00e5ff'
ACCENT2 = '#ff4081'
PALETTE = 'Spectral'

# 1 — Regime Transition Heatmap
ax1 = fig.add_subplot(gs[0, 0])
tm  = np.zeros((5, 5))
for i in range(1, len(gmm_labels)):
    tm[gmm_labels[i-1], gmm_labels[i]] += 1
sns.heatmap(tm, annot=True, fmt='.0f', cmap='mako', ax=ax1,
            linewidths=0.5, linecolor='#222')
ax1.set_title('Regime Transition Heatmap', color='white', pad=10)
ax1.set_xlabel('To Regime');  ax1.set_ylabel('From Regime')

# 2 — 3D Volatility-Regime Surface
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
ax2.plot_trisurf(np.arange(len(df)), df['volatility'], gmm_labels,
                 cmap=cm.viridis, linewidth=0.1, alpha=0.85)
ax2.set_title('3D Volatility-Regime Surface', color='white', pad=10)
ax2.set_xlabel('Time'); ax2.set_ylabel('Volatility'); ax2.set_zlabel('Regime')
ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='white')

# 3 — GMM Confidence
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(gmm_conf, color=ACCENT, linewidth=0.6, alpha=0.9)
ax3.fill_between(range(len(gmm_conf)), gmm_conf, alpha=0.15, color=ACCENT)
ax3.set_title('GMM Regime Confidence', color='white', pad=10)
ax3.set_xlabel('Time'); ax3.set_ylabel('Confidence')

# 4 — PCA Scatter
ax4 = fig.add_subplot(gs[1, 0])
sc  = ax4.scatter(X_pca[:,0], X_pca[:,1], c=gmm_labels, cmap=PALETTE, alpha=0.6, s=4)
plt.colorbar(sc, ax=ax4, label='Regime')
ax4.set_title('PCA Projection (GMM Regimes)', color='white', pad=10)
ax4.set_xlabel('PC1'); ax4.set_ylabel('PC2')

# 5 — GMM Hidden State Means
ax5 = fig.add_subplot(gs[1, 1])
sns.heatmap(gmm.means_, annot=True, fmt='.2f', cmap='rocket', ax=ax5,
            linewidths=0.5, linecolor='#222')
ax5.set_title('GMM Regime Means (Hidden States)', color='white', pad=10)
ax5.set_xlabel('Feature'); ax5.set_ylabel('Regime')

# 6 — Regime Probability Distributions
ax6 = fig.add_subplot(gs[1, 2])
colors_kde = [ACCENT,'#ff4081','#69f0ae','#ffd740','#e040fb']
for i in range(gmm_probs.shape[1]):
    sns.kdeplot(gmm_probs[:,i], label=f'Regime {i}', ax=ax6, color=colors_kde[i])
ax6.set_title('Regime Probability Distributions', color='white', pad=10)
ax6.set_xlabel('Probability'); ax6.set_ylabel('Density'); ax6.legend(fontsize=8)

# 7 — 3D Time×Volatility×Regime Scatter
ax7 = fig.add_subplot(gs[2, 0], projection='3d')
ax7.scatter(np.arange(len(df)), df['volatility'], gmm_labels,
            c=gmm_labels, cmap=PALETTE, alpha=0.5, s=2)
ax7.set_title('Time × Volatility × Regime', color='white', pad=10)
ax7.set_xlabel('Time'); ax7.set_ylabel('Volatility'); ax7.set_zlabel('Regime')
ax7.set_facecolor('#0d1117'); ax7.tick_params(colors='white')

# 8 — Monte Carlo Simulations
ax8 = fig.add_subplot(gs[2, 1])
np.random.seed(0)
for _ in range(15):
    sim = np.cumsum(np.random.normal(0, 1, len(df)))
    ax8.plot(sim, alpha=0.3, color='orange', linewidth=0.7)
ax8.set_title('Monte Carlo Regime Simulations', color='white', pad=10)
ax8.set_xlabel('Time'); ax8.set_ylabel('Simulated Price')

# 9 — Volatility Distribution
ax9 = fig.add_subplot(gs[2, 2])
sns.histplot(df['volatility'], bins=40, color=ACCENT2, ax=ax9, kde=True,
             edgecolor='none', alpha=0.7)
ax9.set_title('Volatility Distribution', color='white', pad=10)
ax9.set_xlabel('Volatility'); ax9.set_ylabel('Frequency')

fig.suptitle('⚡ Neural Regime Detection Engine — Prince Maurya Quant Research',
             fontsize=18, color='white', y=0.995, fontweight='bold')

out = "dashboard_outputs/regime_engine/full_dashboard.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"✅  Saved: {out}")
plt.close()

# ── INDIVIDUAL CHARTS ─────────────────────────────────────────────────────────
def save(name):
    path = f"dashboard_outputs/regime_engine/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"✅  Saved: {path}")
    plt.close()

# Price + Regimes
set_dark_theme()
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['price'], color=ACCENT, linewidth=0.8)
for i, reg in enumerate(np.unique(gmm_labels)):
    mask = gmm_labels == reg
    ax.fill_between(np.arange(len(df)), df['price'].min(), df['price'].max(),
                    where=mask, alpha=0.12, label=f'Regime {reg}')
ax.set_title('Synthetic Price Series with Detected Regimes', color='white')
ax.set_xlabel('Time'); ax.set_ylabel('Price')
ax.legend(fontsize=8, loc='upper left')
save('price_with_regimes')

# Volatility surface (standalone)
set_dark_theme()
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(np.arange(len(df)), df['volatility'], gmm_labels.astype(float),
                cmap=cm.viridis, linewidth=0.1, alpha=0.9)
ax.set_title('3D Volatility-Regime Surface', color='white', pad=15)
ax.set_xlabel('Time'); ax.set_ylabel('Volatility'); ax.set_zlabel('Regime')
ax.set_facecolor('#0d1117'); ax.tick_params(colors='white')
save('surface_plot_hd')

print("\n🎉  All outputs saved to dashboard_outputs/regime_engine/")
