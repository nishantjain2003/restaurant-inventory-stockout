# 🍛 Restaurant Inventory & Stockout Forecasting System

A demand forecasting and stockout prediction system for a simulated South Indian restaurant chain, built end-to-end as a portfolio project — synthetic data generation, feature engineering, two LightGBM models, and an interactive Streamlit dashboard.

## Overview

No public dataset existed combining restaurant sales, recipes, and inventory data for South Indian cuisine, so the entire dataset was custom-built with realistic business logic: weekly seasonality, festival demand spikes (Pongal, Onam, Diwali, Ugadi), category-specific inventory policies, and supplier lead-time variability.

## Results

| Model | Metric | Result |
|---|---|---|
| Demand Forecasting (LightGBM Regression) | R² | 0.99 |
| Demand Forecasting (LightGBM Regression) | MAPE | ~13% |
| Stockout Classifier (LightGBM Classification) | Recall | 0.97 (132/136 real stockouts caught) |
| Stockout Classifier (LightGBM Classification) | Precision | 0.18 (deliberately favors recall) |

The stockout classifier is tuned toward high recall over precision: in this business, missing a real stockout (a dish pulled off the menu) is far more costly than a false alarm (a manager double-checks stock that turns out fine).

## Data Pipeline

1. **dish_ingredient_bom.csv** — recipe table: 18 dishes → ingredients + quantity per plate (90 rows, 40 ingredients)
2. **dish_sales_daily.csv** — synthetic daily sales, 3 outlets, 2 years, with seasonality and festival spikes (39,474 rows)
3. **ingredient_consumption_daily.csv** — sales × recipe = daily ingredient usage (87,720 rows)
4. **inventory_daily.csv** — simulated stockroom with reorder policies, lead times, and stockout events (87,720 rows, ~2% stockout rate)
5. **model_ready_features.csv** — calendar + lag/rolling features, leak-checked (86,040 rows)

## Project Structure

\\\
restaurant_project/
├── app/dashboard.py           Streamlit dashboard (single-item + chain-wide views)
├── data/raw/                  Source data (BOM, sales)
├── data/processed/            Derived data (consumption, inventory, features)
├── models/                    Trained LightGBM models
├── notebooks/                 Full pipeline as a runnable Jupyter notebook
├── src/                       Pipeline scripts (numbered, run in order)
└── requirements.txt
\\\

## Setup

\\\powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/01_generate_sales.py
python src/02_generate_ingredient_consumption.py
python src/03_generate_inventory.py
python src/05_feature_engineering.py
python src/06_train_demand_model.py
python src/07_train_stockout_model.py

streamlit run app/dashboard.py
\\\

## Dashboard

- **Single Item Lookup** — enter an outlet, ingredient, and current stock to get a 7–14 day depletion forecast, stockout alert, and daily risk table
- **Chain-Wide Overview** — a daily alert panel showing which items are nearing stockout across all outlets, risk breakdowns by outlet/category, and actual-vs-predicted accuracy charts

## Key Design Decisions

- **Time-based train/test split** — no random shuffling, since this is time series data
- **No lookahead leakage** — all lag/rolling features verified to use only past data
- **Short forecast horizon (7–14 days)** — recursive forecasting compounds error over longer horizons, so the tool is designed to be re-run daily with fresh real stock data rather than trusted far into the future
- **Recall-optimized stockout classifier** — reflects the real asymmetric cost of missing a stockout vs. a false alarm

## Limitations

- All data is synthetic — a demonstration of methodology, not validated on real operational data
- Requires a known current stock level as input (does not solve automated stock counting)
- Chain-wide risk estimates are conservative by design (biased toward over-flagging risk)

## Built With

Python · pandas · LightGBM · scikit-learn · Streamlit · Plotly
