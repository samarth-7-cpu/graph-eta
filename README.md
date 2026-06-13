# 🚚 Graph-Based ETA Prediction
### Delhivery Logistics Network

**Summer Projects 2026** · Consulting & Analytics Club, IIT Guwahati

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-FF6600?style=for-the-badge)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Analytics-4B8BBE?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

*Optimizing Delivery ETAs with Graph-Based Network Intelligence*

[![Live Dashboard](https://img.shields.io/badge/🚀_Live_Dashboard-Click_Here-success?style=for-the-badge)](https://graph-eta-kxvt2l6pz6plyaousc7apr.streamlit.app/)

</div>

---

## 📌 Project Overview

This project builds a **graph-enhanced ETA prediction system** for Delhivery's logistics network.

The hub-and-spoke logistics network is modeled as a **directed weighted graph**, where:

- 🏭 Logistics facilities are represented as **nodes**
- 🛣️ Shipment corridors are represented as **edges**
- ⏱️ Corridor delays are represented as **edge weights**

Graph-theoretic metrics — Degree Centrality, Betweenness Centrality, PageRank, and Clustering Coefficient — are incorporated into a machine learning pipeline to improve ETA prediction accuracy, identify bottlenecks, and generate actionable operational recommendations.

> The final output is not just a predictive model — it is a complete analytics and consulting solution for logistics network optimization.

---

## 🏆 Key Results

| Metric | Baseline XGBoost | Graph-Enhanced XGBoost |
|--------|:----------------:|:----------------------:|
| MAE | 7.26 min | **6.90 min** |
| Improvement | — | ✅ **4.96% reduction in MAE** |
| Predictions within 15% of actual | 98.82% | 98.77% |

### 🔍 Network Findings

> 📍 Identified **2,634 chronic delay corridors** (actual time > 20% over OSRM)  
> 🔴 **Top 3 bottleneck hubs** account for **40.8%** of all delayed shipments (55,812 of 136,902)  
> 🚛 **FTL shipments** consistently outperform Carting across all distance buckets on delay ratio

---

## 📦 Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Graph Construction & Data Pipeline | `src/graph_builder.py`, `notebooks/02_graph_construction.ipynb` |
| 2 | Bottleneck & Corridor Audit | `notebooks/03_network_visualization.ipynb` |
| 3 | Graph-Enhanced ETA Prediction Model | `src/graph_model.py`, `src/train_and_save_models.py` |
| 4 | FTL vs Carting Decision Framework | `src/ftl_vs_carting.py` |
| 5 | Network Operations Strategy Memo | `docs/strategy_memo.pdf` |
| 6 | Interactive Dashboard *(Optional)* | `dashboard.py` · [Live Link ↗](https://graph-eta-kxvt2l6pz6plyaousc7apr.streamlit.app/) |

---

## 🗂️ Project Structure

```text
graph-eta/
│
├── dashboard.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── delivery_data.csv
│   └── processed/
│       ├── corridor_stats.csv
│       └── graph_features.csv
│
├── outputs/
│   ├── baseline_xgb.pkl
│   ├── graph_xgb.pkl
│   ├── corridor_graph.pkl
│   └── ...
│
├── src/
│   ├── graph_builder.py
│   ├── graph_model.py
│   ├── train_and_save_models.py
│   └── ftl_vs_carting.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_network_visualization.ipynb
│   └── 04_graph_features.ipynb
│
└── docs/
    └── strategy_memo.pdf
```

---

## 🚀 Running Locally

### Prerequisites

- Python 3.9+
- `delivery_data.csv` — place inside `data/raw/delivery_data.csv`

### Installation

```bash
git clone https://github.com/samarth-7-cpu/graph-eta.git
cd graph-eta
pip install -r requirements.txt
```

### Pipeline

**Step 1 — Build Graph Features**
```bash
python src/graph_builder.py
# Generates: corridor_stats.csv, graph_features.csv
```

**Step 2 — Train Models**
```bash
python src/train_and_save_models.py
# Generates: baseline_xgb.pkl, graph_xgb.pkl
```

> ⚠️ Steps 1 and 2 only need to be run **once**. After that, the serialized `.pkl` files in `outputs/` are used directly by the dashboard.

**Step 3 — Launch Dashboard**
```bash
streamlit run dashboard.py
```

---

## 📊 Dashboard Features

| Panel | Description |
|-------|-------------|
| 🌐 **Network Explorer** | Interactive logistics network · Bottleneck hub analysis · Corridor delay exploration |
| 🎯 **ETA Prediction** | Delivery time estimation for any source → destination hub pair · Delay risk scoring |
| 📈 **Model Insights** | Baseline vs Graph model comparison · Feature importance analysis |
| 📡 **Operational Monitoring** | Hub-level analytics · Network performance tracking |

---

## 🔴 Top Bottleneck Hubs

| Hub | Betweenness Centrality | SLA Breach Rate | Recommended Action |
|-----|:----------------------:|:---------------:|-------------------|
| IND000000ACB | 0.189 | 97.0% | Capacity upgrade |
| IND562132AAA | 0.111 | 97.1% | Congestion monitoring |
| IND501359AAE | 0.080 | 97.5% | Alternate routing |
| IND712311AAA | 0.063 | 99.7% | Resource allocation |
| IND421302AAG | 0.055 | 99.4% | Backup route planning |

> 💡 Upgrading the **top 3 hubs** is projected to reduce network-wide late deliveries by up to **40.8%**.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Analysis | Python · Pandas · NumPy |
| Graph Analytics | NetworkX |
| Machine Learning | XGBoost · Scikit-Learn |
| Visualization | Matplotlib · Seaborn · Plotly |
| Deployment | Streamlit |
| Version Control | Git · GitHub |

---

## 🔮 Future Improvements

- Node2Vec Graph Embeddings
- Dynamic Route Optimization
- Real-Time ETA Updating
- Traffic and Weather Integration
- Reinforcement Learning for Routing Decisions

---

## 👥 Team

| Name | Role |
|------|------|
| Samarth Sharma |  Data & Graph Engineer · Graph Construction · Corridor Aggregation · Feature Engineering |
| Hitesh Chandra | ML Lead · Baseline & Graph-Enhanced XGBoost Models · FTL vs Carting Decision Framework |
| Kartik Khemani | Visualisation & Strategy · Network Visualizations · Bottleneck Analysis · Strategy Memo |

---

## 📄 License

This project is licensed under the MIT License.

---
