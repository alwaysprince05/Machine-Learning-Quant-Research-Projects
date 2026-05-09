"""
Master runner — executes all 3 Quant Research projects
and saves every chart to preview_outputs/
Author: Prince Maurya
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda: None   # suppress any plt.show() calls

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import os, sys

OUT = "preview_outputs"
os.makedirs(f"{OUT}/1_liquidity_shock",    exist_ok=True)
os.makedirs(f"{OUT}/2_regime_detection",   exist_ok=True)
os.makedirs(f"{OUT}/3_alpha_explorer",     exist_ok=True)

DARK = {
    'axes.facecolor':   '#0d1117',
    'figure.facecolor': '#0d1117',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#c9d1d9',
    'ytick.color':      '#c9d1d9',
    'axes.edgecolor':   '#30363d',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
}

def dark():
    plt.style.use('dark_background')
    plt.rcParams.update(DARK)

def save(path):
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"  ✅  {path}")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT 1 — Liquidity Shock Simulator
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🔵  PROJECT 1: Liquidity Shock Simulator")

np.random.seed(42)
n = 2000
base = 100.0; prices=[base]; spread=0.02
liq  = np.ones((n,10))*1000
buy_p=np.zeros(n); sell_p=np.zeros(n)
vol=np.zeros(n); stress=np.zeros(n)

for t in range(1, n):
    ret = np.random.normal(0, 0.01)
    if np.random.rand() < 0.02:
        ret += np.random.normal(0, 0.2)
        spread *= np.random.uniform(1.5, 3.0)
        liq[t] = liq[t-1] * np.random.uniform(0.2, 0.5)
        stress[t] = 1
    else:
        spread = max(0.01, spread * np.random.uniform(0.98, 1.02))
        liq[t] = liq[t-1] * np.random.uniform(0.98, 1.02)
    prices.append(prices[-1]*np.exp(ret))
    buy_p[t]  = np.random.normal(0,1) + stress[t]*np.random.uniform(5,15)
    sell_p[t] = np.random.normal(0,1) - stress[t]*np.random.uniform(5,15)
    vol[t]    = abs(ret) + stress[t]*np.random.uniform(0.05,0.2)

prices = np.array(prices)
df1 = pd.DataFrame({
    'mid':prices,'bid':prices-spread/2,'ask':prices+spread/2,
    'spread':spread,'volatility':vol,'buy_pressure':buy_p,
    'sell_pressure':sell_p,'stress_event':stress
})
df1['imbalance']       = df1['buy_pressure'] - df1['sell_pressure']
df1['future_return']   = np.log(df1['mid'].shift(-5)/df1['mid'])
df1['future_vol']      = df1['volatility'].rolling(5).mean().shift(-5)
df1 = df1.dropna()

X1 = df1[['spread','volatility','imbalance','buy_pressure','sell_pressure','stress_event']]
y1 = df1['future_return']
Xtr,Xte,ytr,yte = train_test_split(X1,y1,test_size=0.25,random_state=42)
mdl1 = Ridge(alpha=1.0).fit(Xtr,ytr)
ypr1 = mdl1.predict(Xte)
mc_preds = np.array([mdl1.predict(Xte.iloc[np.random.choice(len(Xte),len(Xte),replace=True)])
                     for _ in range(500)])

# 1a. Price + stress events
dark(); fig,ax=plt.subplots(figsize=(14,5))
ax.plot(df1['mid'].values, color='#58a6ff', lw=0.8, label='Price')
stress_idx = df1[df1['stress_event']>0].index
ax.scatter(range(len(df1)), df1['mid'].values,
           c=df1['stress_event'].values, cmap='Reds', s=2, alpha=0.6)
ax.set_title('Synthetic Price Series with Liquidity Shock Events', color='white', pad=10)
ax.set_xlabel('Time'); ax.set_ylabel('Price')
save(f"{OUT}/1_liquidity_shock/price_with_shocks.png")

# 1b. Monte Carlo PnL
dark(); fig,ax=plt.subplots(figsize=(10,5))
pnl = mc_preds.sum(axis=1)
ax.hist(pnl, bins=50, color='#00e5ff', alpha=0.8, edgecolor='none')
ax.axvline(np.percentile(pnl,5), color='#ff4081', lw=2, linestyle='--', label='5th pct (VaR)')
ax.set_title('Monte Carlo PnL Distribution', color='white', pad=10)
ax.set_xlabel('PnL'); ax.set_ylabel('Frequency'); ax.legend()
save(f"{OUT}/1_liquidity_shock/monte_carlo_pnl.png")

# 1c. Feature importance
dark(); fig,ax=plt.subplots(figsize=(9,5))
imp = np.abs(mdl1.coef_); idx=np.argsort(imp)
ax.barh(np.array(X1.columns)[idx], imp[idx], color='#00bfae')
ax.set_title('Feature Importance (Ridge)', color='white', pad=10)
save(f"{OUT}/1_liquidity_shock/feature_importance.png")

# 1d. Predicted vs Actual
dark(); fig,ax=plt.subplots(figsize=(7,6))
ax.scatter(yte, ypr1, alpha=0.3, color='#ffd740', s=8)
lims=[yte.min(),yte.max()]
ax.plot(lims,lims,'--',color='#888',lw=1)
ax.set_title('Predicted vs Actual Returns', color='white', pad=10)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
save(f"{OUT}/1_liquidity_shock/predicted_vs_actual.png")

# 1e. 3D Liquidity Stress Surface (scatter — more robust than trisurf)
dark(); fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection='3d')
s=df1.sample(800,random_state=0).copy()
sc3=ax.scatter(s['buy_pressure'].values, s['sell_pressure'].values, s['volatility'].values,
               c=s['volatility'].values, cmap='viridis', alpha=0.7, s=6)
fig.colorbar(sc3, ax=ax, shrink=0.5, label='Volatility')
ax.set_title('3D Liquidity Stress Surface', color='white', pad=12)
ax.set_xlabel('Buy Pressure'); ax.set_ylabel('Sell Pressure'); ax.set_zlabel('Volatility')
ax.set_facecolor('#0d1117'); ax.tick_params(colors='white')
save(f"{OUT}/1_liquidity_shock/3d_liquidity_surface.png")

# 1f. Execution pressure heatmap
dark(); fig,ax=plt.subplots(figsize=(8,6))
h,xe,ye=np.histogram2d(df1['buy_pressure'],df1['sell_pressure'],bins=40)
ax.imshow(h.T,extent=[xe[0],xe[-1],ye[0],ye[-1]],origin='lower',cmap='magma',aspect='auto')
ax.set_title('Execution Pressure Heatmap', color='white', pad=10)
ax.set_xlabel('Buy Pressure'); ax.set_ylabel('Sell Pressure')
save(f"{OUT}/1_liquidity_shock/execution_pressure.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT 2 — Neural Regime Detection Engine
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🟢  PROJECT 2: Neural Regime Detection Engine")

np.random.seed(42)
regimes=['trend','mean_revert','panic','crash','high_vol']
rl=np.random.choice([200,300,400],size=10)
rt=np.random.choice(regimes,size=len(rl),replace=True)
data2,labs2=[],[]
for reg,length in zip(rt,rl):
    mu,sig={'trend':(0.05,0.8),'mean_revert':(0,0.5),'panic':(-0.1,2.0),'crash':(-0.2,3.0),'high_vol':(0,2.5)}[reg]
    seg=np.cumsum(np.random.normal(mu,sig,length))
    if data2: seg+=data2[-1]
    data2.extend(seg); labs2.extend([reg]*length)
data2=np.array(data2[:2000]); labs2=np.array(labs2[:2000])
rets2=np.diff(data2,prepend=data2[0])
vol2=pd.Series(rets2).rolling(20).std().fillna(0).values
df2=pd.DataFrame({'price':data2,'returns':rets2,'volatility':vol2})
df2['ma10']=df2['price'].rolling(10).mean()
df2['ma50']=df2['price'].rolling(50).mean()
df2['mom']=df2['price']-df2['price'].shift(10)
df2['vol_r']=df2['returns'].rolling(20).std()
df2=df2.fillna(0)

X2=df2[['returns','volatility','ma10','ma50','mom','vol_r']].values
gmm=GaussianMixture(n_components=5,covariance_type='full',random_state=42).fit(X2)
gl=gmm.predict(X2); gp=gmm.predict_proba(X2); gc=gp.max(axis=1)
km=KMeans(n_clusters=5,random_state=42,n_init=10).fit(X2)
pca2=PCA(n_components=3,random_state=42)
Xpca=pca2.fit_transform(X2)

# 2a. Full 9-panel dashboard
dark(); fig=plt.figure(figsize=(22,16)); fig.patch.set_facecolor('#0d1117')
gs=fig.add_gridspec(3,3,hspace=0.45,wspace=0.38)
COLS=['#00e5ff','#ff4081','#69f0ae','#ffd740','#e040fb']

ax=fig.add_subplot(gs[0,0])
tm=np.zeros((5,5))
for i in range(1,len(gl)): tm[gl[i-1],gl[i]]+=1
sns.heatmap(tm,annot=True,fmt='.0f',cmap='mako',ax=ax,linewidths=0.5,linecolor='#222')
ax.set_title('Regime Transition Heatmap',color='white',pad=8)

ax=fig.add_subplot(gs[0,1],projection='3d')
ax.plot_trisurf(np.arange(len(df2)),df2['volatility'],gl.astype(float),cmap=cm.viridis,linewidth=0.1,alpha=0.85)
ax.set_title('3D Volatility-Regime Surface',color='white',pad=8)
ax.set_xlabel('Time'); ax.set_ylabel('Volatility'); ax.set_zlabel('Regime')
ax.set_facecolor('#0d1117'); ax.tick_params(colors='white')

ax=fig.add_subplot(gs[0,2])
ax.plot(gc,color='#00e5ff',lw=0.6); ax.fill_between(range(len(gc)),gc,alpha=0.12,color='#00e5ff')
ax.set_title('GMM Regime Confidence',color='white',pad=8)

ax=fig.add_subplot(gs[1,0])
sc=ax.scatter(Xpca[:,0],Xpca[:,1],c=gl,cmap='Spectral',alpha=0.6,s=4)
plt.colorbar(sc,ax=ax,label='Regime')
ax.set_title('PCA Projection (GMM Regimes)',color='white',pad=8)

ax=fig.add_subplot(gs[1,1])
sns.heatmap(gmm.means_,annot=True,fmt='.2f',cmap='rocket',ax=ax,linewidths=0.5,linecolor='#222')
ax.set_title('GMM Hidden State Means',color='white',pad=8)

ax=fig.add_subplot(gs[1,2])
for i,c in enumerate(COLS): sns.kdeplot(gp[:,i],label=f'Regime {i}',ax=ax,color=c)
ax.set_title('Regime Probability Distributions',color='white',pad=8); ax.legend(fontsize=8)

ax=fig.add_subplot(gs[2,0],projection='3d')
ax.scatter(np.arange(len(df2)),df2['volatility'],gl,c=gl,cmap='Spectral',alpha=0.5,s=2)
ax.set_title('Time × Volatility × Regime',color='white',pad=8)
ax.set_facecolor('#0d1117'); ax.tick_params(colors='white')

ax=fig.add_subplot(gs[2,1])
np.random.seed(0)
for _ in range(15): ax.plot(np.cumsum(np.random.normal(0,1,len(df2))),alpha=0.3,color='orange',lw=0.7)
ax.set_title('Monte Carlo Regime Simulations',color='white',pad=8)

ax=fig.add_subplot(gs[2,2])
sns.histplot(df2['volatility'],bins=40,color='#ff4081',ax=ax,kde=True,edgecolor='none',alpha=0.7)
ax.set_title('Volatility Distribution',color='white',pad=8)

fig.suptitle('⚡ Neural Regime Detection Engine — Prince Maurya Quant Research',
             fontsize=18,color='white',y=0.998,fontweight='bold')
save(f"{OUT}/2_regime_detection/full_dashboard.png")

# 2b. Price with regimes
dark(); fig,ax=plt.subplots(figsize=(14,5))
ax.plot(df2['price'],color='#58a6ff',lw=0.8)
for i in range(5):
    m=gl==i; ax.fill_between(np.arange(len(df2)),df2['price'].min(),df2['price'].max(),
                              where=m,alpha=0.1,label=f'Regime {i}')
ax.set_title('Synthetic Price with GMM Detected Regimes',color='white',pad=10)
ax.legend(fontsize=8,loc='upper left')
save(f"{OUT}/2_regime_detection/price_with_regimes.png")

# 2c. 3D surface standalone
dark(); fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection='3d')
ax.plot_trisurf(np.arange(len(df2)),df2['volatility'],gl.astype(float),cmap=cm.viridis,alpha=0.9)
ax.set_title('3D Volatility-Regime Surface (HD)',color='white',pad=12)
ax.set_xlabel('Time'); ax.set_ylabel('Volatility'); ax.set_zlabel('Regime')
ax.set_facecolor('#0d1117'); ax.tick_params(colors='white')
save(f"{OUT}/2_regime_detection/3d_surface_hd.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT 3 — Multi-Factor Alpha Landscape Explorer
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🟡  PROJECT 3: Multi-Factor Alpha Landscape Explorer")

np.random.seed(42)
n_assets=20; n_periods=100
dates=pd.date_range(end=pd.Timestamp.today(),periods=n_periods)
prices3=pd.DataFrame(
    np.cumprod(1+np.random.normal(0,0.01,(n_periods,n_assets)),axis=0)*100,
    index=dates,columns=[f"A{i+1}" for i in range(n_assets)])
rets3=prices3.pct_change().fillna(0)

factors3={}
factors3['momentum']           = rets3.rolling(20).mean()
factors3['volatility']         = rets3.rolling(20).std()
factors3['mean_reversion']     = -rets3.rolling(5).mean()
factors3['liquidity_imbalance']= (prices3.rolling(20).max()-prices3.rolling(20).min())/(rets3.rolling(20).std()+1e-6)
fdf=pd.concat(factors3,axis=1).dropna()

X3=fdf.values; y3=rets3.shift(-1).dropna().values.flatten()
ml=min(len(X3),len(y3)); X3=X3[:ml]; y3=y3[:ml]
rf=RandomForestRegressor(n_estimators=50,random_state=42).fit(X3,y3)
yp3=rf.predict(X3); imp3=rf.feature_importances_
pca3=PCA(n_components=2); Xp3=pca3.fit_transform(X3)

# 3a. Feature importance
dark(); fig,ax=plt.subplots(figsize=(10,5))
factor_names=[f"{f}-{a}" for f,a in fdf.columns]
idx=np.argsort(imp3)
ax.barh(np.array(factor_names)[idx],imp3[idx],color='#00e5ff')
ax.set_title('Alpha Factor Importance (Random Forest)',color='white',pad=10)
plt.tight_layout()
save(f"{OUT}/3_alpha_explorer/feature_importance.png")

# 3b. 3D Factor Landscape
dark(); fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection='3d')
Xp3j=Xp3.copy(); Xp3j[:,0]+=np.random.normal(0,Xp3j[:,0].std()*0.001,len(Xp3j))
Xp3j[:,1]+=np.random.normal(0,Xp3j[:,1].std()*0.001,len(Xp3j))
ax.plot_trisurf(Xp3j[:,0],Xp3j[:,1],yp3[:len(Xp3j)],cmap='viridis',alpha=0.8)
ax.set_title('3D Alpha Factor Landscape (PCA)',color='white',pad=12)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('Predicted Alpha')
ax.set_facecolor('#0d1117'); ax.tick_params(colors='white')
save(f"{OUT}/3_alpha_explorer/3d_factor_landscape.png")

# 3c. Predicted vs Actual
dark(); fig,ax=plt.subplots(figsize=(7,6))
ax.scatter(y3,yp3,alpha=0.3,color='cyan',s=8)
ax.plot([y3.min(),y3.max()],[y3.min(),y3.max()],'r--',lw=1)
ax.set_title('Predicted vs Actual Returns',color='white',pad=10)
ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
save(f"{OUT}/3_alpha_explorer/predicted_vs_actual.png")

# 3d. PnL simulation
dark(); fig,ax=plt.subplots(figsize=(12,5))
ax.plot(np.cumsum(yp3),color='#69f0ae',lw=1.5)
ax.fill_between(range(len(yp3)),np.cumsum(yp3),alpha=0.15,color='#69f0ae')
ax.set_title('Cumulative PnL Simulation',color='white',pad=10)
ax.set_xlabel('Time'); ax.set_ylabel('Cumulative Return')
save(f"{OUT}/3_alpha_explorer/pnl_simulation.png")

# 3e. Correlation heatmap
dark(); fig,ax=plt.subplots(figsize=(9,7))
sns.heatmap(pd.DataFrame(X3).corr(),cmap='coolwarm',center=0,ax=ax,linewidths=0.3)
ax.set_title('Factor Correlation Heatmap',color='white',pad=10)
save(f"{OUT}/3_alpha_explorer/correlation_heatmap.png")

# 3f. Monte Carlo alpha simulations
dark(); fig,ax=plt.subplots(figsize=(12,5))
np.random.seed(1)
for _ in range(20):
    mc=np.cumsum(np.random.choice(yp3,size=len(yp3),replace=True))
    ax.plot(mc,alpha=0.25,lw=0.8)
ax.set_title('Monte Carlo Alpha Simulations',color='white',pad=10)
ax.set_xlabel('Time'); ax.set_ylabel('Simulated Cumulative Return')
save(f"{OUT}/3_alpha_explorer/monte_carlo_alpha.png")

# 3g. Signal strength contour
dark(); fig,ax=plt.subplots(figsize=(8,6))
sns.kdeplot(x=Xp3[:,0],y=Xp3[:,1],fill=True,cmap='mako',ax=ax)
ax.set_title('Signal Strength Contour (PCA)',color='white',pad=10)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
save(f"{OUT}/3_alpha_explorer/signal_strength_contour.png")

print(f"\n🎉  Done! All outputs saved to '{OUT}/'")
print(f"    Project 1 → {OUT}/1_liquidity_shock/")
print(f"    Project 2 → {OUT}/2_regime_detection/")
print(f"    Project 3 → {OUT}/3_alpha_explorer/")
