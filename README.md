# Graph-Based ETA Prediction — Delhivery Logistics Network

> **Summer Projects 2026** · Machine Learning & Consulting  
> Optimizing Delivery ETAs with Graph-Based Network Intelligence

---

## Live Dashboard

🚀 **[View Live Dashboard →](https://YOUR-APP-URL.streamlit.app)**  
*(Replace with your Streamlit Community Cloud URL after deployment)*

---

## Project Overview

This project builds a **graph-enhanced ETA prediction system** for Delhivery's logistics network. It models the entire hub-and-spoke delivery network as a directed weighted graph, computes structural features (betweenness centrality, PageRank, clustering), and uses them to outperform a baseline XGBoost model.

### Key Results
| Metric | Baseline XGBoost | Graph-Enhanced XGBoost |
|--------|-----------------|----------------------|
| MAE | 7.01 min | 6.69 min |
| Improvement | — | **4.60% reduction in MAE** |

---

## Deliverables

| # | Deliverable | Location |
|---|------------|----------|
| 1 | Graph construction & data pipeline | `src/graph_builder.py`, `notebooks/02_graph_construction.ipynb` |
| 2 | Bottleneck & corridor audit | `src/graph.py`, `notebooks/03_network_visualization.ipynb` |
| 3 | Graph-enhanced ETA model | `src/graph_model.py`, `src/train_and_save_models.py` |
| 4 | FTL vs Carting decision framework | `src/ftl vs carting.py` |
| 5 | Network Operations Strategy Memo | `docs/` |
| 6 | Live Streamlit Dashboard *(optional)* | `dashboard.py` — [Live Link](https://YOUR-APP-URL.streamlit.app) |

---

## Project Structure

```
graph-eta/
├── dashboard.py              # Live Streamlit dashboard
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # delivery_data.csv (not in repo — too large)
│   └── processed/
│       ├── graph_features.csv
│       └── corridor_stats.csv
├── outputs/                  # Serialized models & artifacts
│   ├── baseline_xgb.pkl
│   ├── graph_xgb.pkl
│   ├── corridor_graph.pkl
│   └── ...
├── src/
│   ├── graph.py              # Network visualization
│   ├── graph_model.py        # Model training
│   ├── train_and_save_models.py
│   └── ...
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_graph_construction.ipynb
    ├── 03_network_visualization.ipynb
    └── 04_graph_features.ipynb
```

---

## Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO/graph-eta

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add delivery_data.csv to data/raw/  (download from provided link)

# 4. Train models (first time only — takes ~2 mins)
python src/train_and_save_models.py

# 5. Launch dashboard
streamlit run dashboard.py
```

---

## Dashboard Features

- **🌐 Network Explorer** — Interactive Plotly graph of 1,483 hubs and 2,203 corridors with real-time filters
- **📡 Live Monitor** — Simulated shipment stream with Graph-Enhanced ETA predictions and delay risk scores
- **🎯 ETA Calculator** — Predict ETA for any source → destination hub pair
- **📊 Model Insights** — Feature importance comparison and delay distribution analytics
