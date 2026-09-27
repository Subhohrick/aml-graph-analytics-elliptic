# Money Laundering Detection on Bitcoin Transactions using Graph Analytics

This project detects illicit (criminal) Bitcoin transactions using machine learning and graph analytics on the **Elliptic Bitcoin dataset**. It tests whether graph-based features, derived from how transactions are connected, improve classic ML models, and translates the results into business terms for anti-money laundering (AML) teams: alert budgets, cost of errors, and model decay over time.

## Dataset

The [Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) is a real Bitcoin transaction graph released by Elliptic, a blockchain analytics company.

- **203,769 transactions** (nodes) and **234,355 money flows** (edges)
- **166 anonymised features** per transaction: 93 local features and 72 features aggregated from neighbouring transactions
- **49 time steps**, roughly two weeks apart
- **Labels:** 4,545 illicit (scams, ransomware, dark markets, etc.), 42,019 licit (exchanges, wallets, miners, etc.), and 157,205 unknown

The dataset is not included in this repository due to its size. Download it from Kaggle and place the three CSV files directly in a `data/` folder:

```
data/
├── elliptic_txs_features.csv
├── elliptic_txs_classes.csv
└── elliptic_txs_edgelist.csv
```

## Methodology

1. **Exploratory analysis:** class imbalance and illicit activity over time
2. **Graph construction:** transactions as nodes, money flows as directed edges (NetworkX)
3. **Graph features:** in/out/total degree, PageRank, clustering coefficient, average neighbour degree, Louvain community size, connected component size
4. **Temporal split:** train on time steps 1–34, test on 35–49, to mimic predicting future transactions
5. **Supervised models:** Logistic Regression, Random Forest, and XGBoost, each trained on four feature sets (Local, Local + Graph, All, All + Graph)
6. **Unsupervised methods:** Isolation Forest (anomaly detection) and K-Means (risk segmentation)
7. **Explainability:** SHAP feature importance
8. **Business analysis:** precision@100 alert budget, cost-based threshold selection, and model decay over time

## How to Run

```
pip install -r requirements.txt
python aml_graph_project.py
```

All charts and result tables are saved to the `outputs/` folder. The first run takes about 10–20 minutes; graph features are cached for later runs.

## Results

| Model | Features | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| Random Forest | All | 0.991 | 0.693 | **0.815** | 0.797 |
| Random Forest | All + Graph | 0.988 | 0.688 | 0.811 | 0.798 |
| XGBoost | All + Graph | 0.842 | 0.736 | 0.786 | **0.803** |
| XGBoost | Local | 0.734 | 0.730 | 0.732 | 0.787 |
| Logistic Regression | All + Graph | 0.204 | 0.815 | 0.327 | 0.274 |

Metrics are for the illicit class on the test period (time steps 35–49). The
