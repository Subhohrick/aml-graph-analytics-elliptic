"""
Money Laundering Detection on the Elliptic Bitcoin Dataset using Graph Analytics

Run from the project folder:  python aml_graph_project.py
Expects data/ (3 Elliptic CSVs) next to this file; charts and tables go to outputs/.
The "# %%" markers let you also run it cell by cell in VS Code.
"""

# %% [markdown]
# # Money Laundering Detection on the Elliptic Bitcoin Dataset using Graph Analytics
#
# **Goal:** Test whether graph-based features (degree, PageRank, clustering, communities) improve classic ML models
# (Logistic Regression, Random Forest, XGBoost) at detecting illicit Bitcoin transactions, and translate the results into
# business terms (alert budget, cost of errors, model decay).
#
# **Sections**
# 1. Setup
# 2. Load data
# 3. Exploratory data analysis
# 4. Graph construction and graph features
# 5. Temporal train/test split
# 6. Supervised models: raw vs raw + graph features
# 7. Unsupervised methods: Isolation Forest and K-Means
# 8. Explainability (SHAP)
# 9. Business analysis: alert budget, cost-based threshold, model decay
# 10. Save outputs

# %% [markdown]
# ## 1. Setup

# %%
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')        # save charts to files without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
sns.set_style('whitegrid')

# Paths are relative to the project folder (where this script sits)
try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:          # running cell-by-cell in an interactive window
    BASE = os.getcwd()
DATA = os.path.join(BASE, 'data') + os.sep     # folder with the 3 Elliptic CSVs
OUT = os.path.join(BASE, 'outputs') + os.sep   # charts and results are saved here
os.makedirs(OUT, exist_ok=True)

TRAIN_END = 34           # train on time steps 1-34, test on 35-49
SEED = 42

# %% [markdown]
# ## 2. Load data

# %%
feat = pd.read_csv(DATA + 'elliptic_txs_features.csv', header=None)
feat.columns = ['txId', 'time_step'] + [f'f{i}' for i in range(1, 166)]
cls = pd.read_csv(DATA + 'elliptic_txs_classes.csv')
edges = pd.read_csv(DATA + 'elliptic_txs_edgelist.csv')

df = feat.merge(cls, on='txId', how='left')
df['class'] = df['class'].astype(str)
df['label'] = df['class'].map({'1': 1, '2': 0})   # 1 = illicit, 0 = licit, NaN = unknown

print('Features:', feat.shape)
print('Edges   :', edges.shape)
print(df['class'].value_counts())

# %% [markdown]
# ## 3. Exploratory data analysis

# %%
# 3.1 Class distribution
counts = df['class'].map({'1': 'Illicit', '2': 'Licit', 'unknown': 'Unknown'}).value_counts()
ax = counts.plot(kind='bar', color=['grey', 'tab:green', 'tab:red'], title='Class distribution')
for i, v in enumerate(counts):
    ax.text(i, v, f'{v:,}', ha='center', va='bottom')
plt.ylabel('Transactions'); plt.tight_layout(); plt.savefig(OUT + 'class_distribution.png'); plt.close()

labelled = df[df.label.notna()]
print(f"Illicit share among labelled: {labelled.label.mean():.2%}")
print(f"Illicit share among all     : {(df.label == 1).mean():.2%}")

# %%
# 3.2 Licit vs illicit transactions per time step
t = labelled.groupby(['time_step', 'label']).size().unstack(fill_value=0)
t.columns = ['Licit', 'Illicit']
t.plot(figsize=(11, 4), color=['tab:green', 'tab:red'], title='Labelled transactions per time step')
plt.axvline(TRAIN_END + 0.5, ls='--', c='k', label='Train / test split')
plt.legend(); plt.tight_layout(); plt.savefig(OUT + 'tx_per_timestep.png'); plt.close()

(t['Illicit'] / t.sum(axis=1) * 100).plot(figsize=(11, 4), color='tab:red', title='% illicit per time step')
plt.axvline(TRAIN_END + 0.5, ls='--', c='k'); plt.ylabel('%')
plt.tight_layout(); plt.savefig(OUT + 'pct_illicit_per_timestep.png'); plt.close()

# %% [markdown]
# ## 4. Graph construction and graph features

# %%
G = nx.from_pandas_edgelist(edges, 'txId1', 'txId2', create_using=nx.DiGraph)
G.add_nodes_from(df.txId)            # include isolated transactions
Gu = G.to_undirected()

print('Nodes:', G.number_of_nodes())
print('Edges:', G.number_of_edges())
print('Connected components:', nx.number_connected_components(Gu))
print(f'Average degree: {2 * Gu.number_of_edges() / Gu.number_of_nodes():.2f}')

# %%
# Graph features are computed once and cached, since PageRank and Louvain take a few minutes
gf_file = OUT + 'graph_features.csv'

if os.path.exists(gf_file):
    gf = pd.read_csv(gf_file)
    print('Loaded cached graph features')
else:
    gf = pd.DataFrame({'txId': list(G.nodes())})
    gf['in_deg'] = gf.txId.map(dict(G.in_degree()))
    gf['out_deg'] = gf.txId.map(dict(G.out_degree()))
    gf['total_deg'] = gf.in_deg + gf.out_deg
    print('Computing PageRank...');   gf['pagerank'] = gf.txId.map(nx.pagerank(G, alpha=0.85))
    print('Computing clustering...'); gf['clustering'] = gf.txId.map(nx.clustering(Gu))
    gf['avg_nbr_deg'] = gf.txId.map(nx.average_neighbor_degree(Gu))

    print('Computing Louvain communities...')
    comms = nx.community.louvain_communities(Gu, seed=SEED)
    node_comm = {n: i for i, c in enumerate(comms) for n in c}
    comm_size = {i: len(c) for i, c in enumerate(comms)}
    gf['community_size'] = gf.txId.map(node_comm).map(comm_size)

    comps = list(nx.connected_components(Gu))
    node_comp_size = {n: len(c) for c in comps for n in c}
    gf['component_size'] = gf.txId.map(node_comp_size)

    gf.to_csv(gf_file, index=False)
    print('Saved graph features to', gf_file)

df = df.merge(gf, on='txId', how='left')
GRAPH = ['in_deg', 'out_deg', 'total_deg', 'pagerank', 'clustering',
         'avg_nbr_deg', 'community_size', 'component_size']
df[GRAPH] = df[GRAPH].fillna(0)
print(df[GRAPH].describe().T)

# %%
# Do graph features differ between licit and illicit transactions?
comp = df[df.label.notna()].groupby('label')[GRAPH].median().T
comp.columns = ['Licit (median)', 'Illicit (median)']
print(comp)

# %%
# Visualise the neighbourhood of one illicit transaction
example = df[(df.label == 1) & (df.total_deg >= 3)].txId.iloc[0]
sub_nodes = nx.ego_graph(Gu, example, radius=2).nodes()
sub = G.subgraph(sub_nodes)
lab_map = df.set_index('txId')['class'].to_dict()
colors = [{'1': 'tab:red', '2': 'tab:green'}.get(lab_map.get(n), 'lightgrey') for n in sub.nodes()]

plt.figure(figsize=(8, 6))
nx.draw_networkx(sub, pos=nx.spring_layout(sub, seed=SEED), node_color=colors,
                 node_size=[300 if n == example else 80 for n in sub.nodes()],
                 with_labels=False, arrows=True, arrowsize=8, edge_color='grey')
plt.title(f'2-hop neighbourhood of illicit transaction {example}\n(red = illicit, green = licit, grey = unknown)')
plt.axis('off'); plt.tight_layout(); plt.savefig(OUT + 'illicit_subgraph.png'); plt.close()

# %% [markdown]
# ## 5. Temporal train/test split
#
# We train on time steps 1–34 and test on 35–49, so the model always predicts *future* transactions.
# `time_step` itself is **not** used as a feature, because test time steps never appear in training.

# %%
LOCAL = [f'f{i}' for i in range(1, 94)]      # 93 local transaction features
ALL = [f'f{i}' for i in range(1, 166)]       # local + 72 aggregated neighbour features

FEATURE_SETS = {
    'Local': LOCAL,
    'Local + Graph': LOCAL + GRAPH,
    'All': ALL,
    'All + Graph': ALL + GRAPH,
}

lab = df[df.label.notna()].copy()
lab['label'] = lab.label.astype(int)
train = lab[lab.time_step <= TRAIN_END]
test = lab[lab.time_step > TRAIN_END]
y_train, y_test = train.label, test.label

print('Train:', train.shape, f'illicit = {y_train.mean():.2%}')
print('Test :', test.shape, f'illicit = {y_test.mean():.2%}')

# %% [markdown]
# ## 6. Supervised models: raw vs raw + graph features

# %%
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

def make_models():
    return {
        'Logistic Regression': make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
        'XGBoost': XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric='aucpr', n_jobs=-1, random_state=SEED),
    }

def evaluate(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {'Precision': precision_score(y_true, pred, zero_division=0),
            'Recall': recall_score(y_true, pred, zero_division=0),
            'F1': f1_score(y_true, pred, zero_division=0),
            'PR-AUC': average_precision_score(y_true, prob)}

results, fitted, probs = [], {}, {}
for fs_name, cols in FEATURE_SETS.items():
    for m_name, model in make_models().items():
        model.fit(train[cols], y_train)
        p = model.predict_proba(test[cols])[:, 1]
        key = (m_name, fs_name)
        fitted[key], probs[key] = model, p
        results.append({'Model': m_name, 'Features': fs_name, **evaluate(y_test, p)})
        print(f'Done: {m_name:20s} | {fs_name}')

res = pd.DataFrame(results).round(3)
res.to_csv(OUT + 'model_results.csv', index=False)
print(res.sort_values('F1', ascending=False).to_string(index=False))

# %%
# F1 comparison chart
pivot = res.pivot(index='Model', columns='Features', values='F1')[list(FEATURE_SETS)]
pivot.plot(kind='bar', figsize=(10, 4), title='Illicit-class F1 by model and feature set')
plt.ylabel('F1'); plt.xticks(rotation=0); plt.legend(loc='lower right')
plt.tight_layout(); plt.savefig(OUT + 'f1_comparison.png'); plt.close()

# Effect of adding graph features
gain = pd.DataFrame({
    'Local -> Local + Graph': pivot['Local + Graph'] - pivot['Local'],
    'All -> All + Graph': pivot['All + Graph'] - pivot['All'],
}).round(3)
print('Change in F1 from adding graph features:'); print(gain)

# %%
# Best model overall
best = res.sort_values('F1', ascending=False).iloc[0]
BEST_KEY = (best.Model, best.Features)
best_prob = probs[BEST_KEY]
print('Best model:', BEST_KEY)

cm = confusion_matrix(y_test, (best_prob >= 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues',
            xticklabels=['Pred licit', 'Pred illicit'], yticklabels=['Licit', 'Illicit'])
plt.title(f'Confusion matrix: {BEST_KEY[0]} ({BEST_KEY[1]})')
plt.tight_layout(); plt.savefig(OUT + 'confusion_matrix.png'); plt.close()

# %% [markdown]
# ## 7. Unsupervised methods
#
# ### 7.1 Isolation Forest
# Trained on **all** transactions (including unknown), without labels. We then check how many of the anomalies it flags are actually illicit.

# %%
iso_cols = ALL + GRAPH
contamination = round(float((df.label == 1).mean()), 3)   # expected illicit share, about 2%
iso = IsolationForest(n_estimators=300, contamination=contamination, n_jobs=-1, random_state=SEED)
iso.fit(df[iso_cols])

df['anomaly'] = (iso.predict(df[iso_cols]) == -1).astype(int)
df['anomaly_score'] = -iso.score_samples(df[iso_cols])     # higher = more anomalous

lab_iso = df[df.label.notna()]
print(f'Flagged anomalies: {df.anomaly.sum():,}')
print(f'Precision (flagged that are illicit): {precision_score(lab_iso.label, lab_iso.anomaly):.3f}')
print(f'Recall (illicit that were flagged)  : {recall_score(lab_iso.label, lab_iso.anomaly):.3f}')
print(f'PR-AUC of anomaly score              : {average_precision_score(lab_iso.label, lab_iso.anomaly_score):.3f}')
print(f'Random baseline PR-AUC               : {lab_iso.label.mean():.3f}')

# %% [markdown]
# ### 7.2 K-Means clustering on graph features

# %%
X_graph = np.log1p(df[GRAPH].clip(lower=0))   # log transform because degree-type features are highly skewed
X_graph = StandardScaler().fit_transform(X_graph)

inertias = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_graph).inertia_ for k in range(2, 11)]
plt.plot(range(2, 11), inertias, marker='o'); plt.xlabel('k'); plt.ylabel('Inertia')
plt.title('Elbow method'); plt.tight_layout(); plt.savefig(OUT + 'kmeans_elbow.png'); plt.close()

# %%
K = 5   # change this based on the elbow chart
df['cluster'] = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X_graph)

profile = df.groupby('cluster').agg(
    transactions=('txId', 'count'),
    labelled=('label', 'count'),
    illicit_rate=('label', 'mean'),
    **{f'avg_{c}': (c, 'mean') for c in ['in_deg', 'out_deg', 'pagerank', 'clustering', 'component_size']}
).round(4).sort_values('illicit_rate', ascending=False)
profile.to_csv(OUT + 'cluster_profile.csv')
print(profile)

# %% [markdown]
# ## 8. Explainability (SHAP) on XGBoost with All + Graph features

# %%
shap_cols = ALL + GRAPH
xgb_model = fitted[('XGBoost', 'All + Graph')]
X_shap = test[shap_cols].sample(min(2000, len(test)), random_state=SEED)

explainer = shap.TreeExplainer(xgb_model)
sv = explainer(X_shap)

shap.plots.bar(sv, max_display=20, show=False)
plt.tight_layout(); plt.savefig(OUT + 'shap_bar.png'); plt.close()
shap.plots.beeswarm(sv, max_display=20, show=False)
plt.tight_layout(); plt.savefig(OUT + 'shap_beeswarm.png'); plt.close()

imp = pd.Series(np.abs(sv.values).mean(0), index=shap_cols).sort_values(ascending=False)
rank = {f: i + 1 for i, f in enumerate(imp.index)}
print('Rank of graph features among', len(shap_cols), 'features:')
print({g: rank[g] for g in GRAPH})

# %%
# Explain the single highest-risk transaction in the sample
top_idx = int(np.argmax(xgb_model.predict_proba(X_shap)[:, 1]))
shap.plots.waterfall(sv[top_idx], max_display=12, show=False)
plt.tight_layout(); plt.savefig(OUT + 'shap_waterfall_top.png'); plt.close()

# %% [markdown]
# ## 9. Business analysis
#
# ### 9.1 Alert budget (precision@k)
# A compliance team can only review a fixed number of alerts per time step. How many real illicit transactions are in the model's top-k?

# %%
K_ALERTS = 100
ev = test[['txId', 'time_step', 'label']].copy()
ev['prob'] = best_prob

rows = []
for ts, g in ev.groupby('time_step'):
    top = g.nlargest(K_ALERTS, 'prob')
    rows.append({'time_step': ts, 'alerts': len(top), 'true_illicit_in_alerts': int(top.label.sum()),
                 'precision@k': top.label.mean(), 'illicit_total': int(g.label.sum())})
pk = pd.DataFrame(rows)
pk['share_of_illicit_caught'] = pk.true_illicit_in_alerts / pk.illicit_total.replace(0, np.nan)
print(f'Average precision@{K_ALERTS}: {pk["precision@k"].mean():.3f}')
print(pk.round(3).to_string(index=False))

# %% [markdown]
# ### 9.2 Cost-based threshold
# **Assumed costs (state these clearly in the report):** a missed illicit transaction costs ₹5,00,000 (fines, reputational risk);
# a false alert costs ₹2,000 (analyst time). Change these to test sensitivity.

# %%
COST_FN = 500_000
COST_FP = 2_000

thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
cost_rows = []
for t in thresholds:
    pred = (best_prob >= t).astype(int)
    fn = int(((pred == 0) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    cost_rows.append({'threshold': t, 'FN': fn, 'FP': fp, 'total_cost': fn * COST_FN + fp * COST_FP})
cost = pd.DataFrame(cost_rows)
best_t = cost.loc[cost.total_cost.idxmin()]

plt.figure(figsize=(9, 4))
plt.plot(cost.threshold, cost.total_cost / 1e7, marker='o')
plt.axvline(best_t.threshold, ls='--', c='r', label=f'Lowest cost at {best_t.threshold}')
plt.axvline(0.5, ls=':', c='k', label='Default 0.5')
plt.xlabel('Decision threshold'); plt.ylabel('Total cost (₹ crore)')
plt.title('Total cost of errors vs threshold'); plt.legend()
plt.tight_layout(); plt.savefig(OUT + 'cost_curve.png'); plt.close()

default_cost = cost.loc[cost.threshold == 0.5, 'total_cost'].iloc[0]
print(f'Cost at default 0.5 : ₹{default_cost:,.0f}')
print(f'Cost at best {best_t.threshold}: ₹{best_t.total_cost:,.0f}')
print(f'Saving              : ₹{default_cost - best_t.total_cost:,.0f}')

# %% [markdown]
# ### 9.3 Model decay over time

# %%
decay = []
for ts, g in ev.groupby('time_step'):
    pred = (g.prob >= 0.5).astype(int)
    decay.append({'time_step': ts, 'F1': f1_score(g.label, pred, zero_division=0),
                  'illicit_count': int(g.label.sum())})
decay = pd.DataFrame(decay)

fig, ax1 = plt.subplots(figsize=(11, 4))
ax1.plot(decay.time_step, decay.F1, marker='o', c='tab:blue', label='F1')
ax1.set_ylabel('F1', color='tab:blue'); ax1.set_xlabel('Time step')
ax2 = ax1.twinx()
ax2.bar(decay.time_step, decay.illicit_count, alpha=0.25, color='tab:red', label='Illicit count')
ax2.set_ylabel('Illicit transactions', color='tab:red')
ax1.axvline(43, ls='--', c='k')
plt.title(f'Model performance per test time step: {BEST_KEY[0]} ({BEST_KEY[1]})')
plt.tight_layout(); plt.savefig(OUT + 'model_decay.png'); plt.close()
print(decay.round(3).to_string(index=False))

# %% [markdown]
# ## 10. Save outputs

# %%
ev.to_csv(OUT + 'test_predictions.csv', index=False)
pk.to_csv(OUT + 'precision_at_k.csv', index=False)
cost.to_csv(OUT + 'cost_by_threshold.csv', index=False)
decay.to_csv(OUT + 'model_decay.csv', index=False)
print('All outputs saved in', os.path.abspath(OUT))