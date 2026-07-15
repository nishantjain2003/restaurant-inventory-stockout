import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
import sys


APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))

st.set_page_config(page_title="Restaurant Inventory Forecast", layout="wide")

@st.cache_resource
def load_artifacts():
    demand_model = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'demand_forecast_lgbm.pkl'))
    stockout_model = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'stockout_classifier_lgbm.pkl'))
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'model_ready_features.csv'), parse_dates=['date'])
    for col in ['outlet_id','ingredient_id','ingredient_category']:
        df[col] = df[col].astype('category')
    festival_lookup_full = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'raw', 'dish_sales_daily.csv'), parse_dates=['date'])[['date','festival']].drop_duplicates()
    demand_test = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'processed', 'demand_forecast_test_predictions.csv'), parse_dates=['date'])
    return demand_model, stockout_model, df, festival_lookup_full, demand_test

demand_model, stockout_model, df, festival_lookup_full, demand_test = load_artifacts()

demand_feature_cols = [
    'outlet_id', 'ingredient_id', 'ingredient_category',
    'weekday', 'is_weekend', 'month', 'day_of_month', 'is_festival', 'days_to_next_festival',
    'consumption_lag_1', 'consumption_lag_7', 'consumption_lag_14',
    'consumption_roll_mean_7', 'consumption_roll_std_7', 'consumption_roll_mean_14',
    'recent_stockout_flag'
]
stockout_feature_cols = [
    'outlet_id', 'ingredient_id', 'ingredient_category',
    'weekday', 'is_weekend', 'month', 'day_of_month', 'is_festival', 'days_to_next_festival',
    'consumption_lag_1', 'consumption_lag_7', 'consumption_lag_14',
    'consumption_roll_mean_7', 'consumption_roll_std_7', 'consumption_roll_mean_14',
    'closing_stock_lag_1', 'stock_to_demand_ratio', 'recent_stockout_flag'
]

category_policy = {
    'Vegetable':  dict(lead_time=1, safety_days=1, order_cover_days=2),
    'Dairy':      dict(lead_time=1, safety_days=1, order_cover_days=2),
    'Herb':       dict(lead_time=1, safety_days=1, order_cover_days=2),
    'Meat':       dict(lead_time=1, safety_days=1, order_cover_days=2),
    'Grain':      dict(lead_time=3, safety_days=1, order_cover_days=4),
    'Pulse':      dict(lead_time=3, safety_days=1, order_cover_days=4),
    'Spice':      dict(lead_time=5, safety_days=2, order_cover_days=6),
    'Sweetener':  dict(lead_time=3, safety_days=1, order_cover_days=4),
    'Nuts':       dict(lead_time=3, safety_days=2, order_cover_days=5),
    'Oil':        dict(lead_time=3, safety_days=1, order_cover_days=4),
    'Fat':        dict(lead_time=3, safety_days=1, order_cover_days=4),
    'Beverage':   dict(lead_time=3, safety_days=1, order_cover_days=4),
}

def days_to_next_festival(d, festival_dates):
    future = [fd for fd in festival_dates if fd >= d]
    return 999 if not future else (future[0] - d).days

def forecast_and_deplete(outlet_id, ingredient_id, current_stock, horizon_days, assume_reordering=False):
    hist = df[(df.outlet_id == outlet_id) & (df.ingredient_id == ingredient_id)].sort_values('date')
    if hist.empty:
        raise ValueError(f"No history found for outlet={outlet_id}, ingredient={ingredient_id}")

    last_date = hist['date'].max()
    ingredient_category = hist['ingredient_category'].iloc[-1]
    consumption_history = hist['consumption'].tolist()
    festival_dates = sorted(festival_lookup_full[festival_lookup_full.festival.notna()]['date'].unique())
    policy = category_policy[ingredient_category]
    pending_orders = {}

    results = []
    stock = current_stock
    stockout_date = None

    for step in range(1, horizon_days + 1):
        future_date = last_date + pd.Timedelta(days=step)

        if assume_reordering and future_date in pending_orders:
            stock += pending_orders.pop(future_date)

        lag_1 = consumption_history[-1]
        lag_7 = consumption_history[-7] if len(consumption_history) >= 7 else np.mean(consumption_history)
        lag_14 = consumption_history[-14] if len(consumption_history) >= 14 else np.mean(consumption_history)
        roll_mean_7 = np.mean(consumption_history[-7:])
        roll_std_7 = np.std(consumption_history[-7:]) if len(consumption_history) >= 2 else 0
        roll_mean_14 = np.mean(consumption_history[-14:])

        base_row = {
            'outlet_id': outlet_id, 'ingredient_id': ingredient_id, 'ingredient_category': ingredient_category,
            'weekday': future_date.weekday(), 'is_weekend': int(future_date.weekday() >= 5),
            'month': future_date.month, 'day_of_month': future_date.day,
            'is_festival': int(future_date in festival_dates),
            'days_to_next_festival': days_to_next_festival(future_date, festival_dates),
            'consumption_lag_1': lag_1, 'consumption_lag_7': lag_7, 'consumption_lag_14': lag_14,
            'consumption_roll_mean_7': roll_mean_7, 'consumption_roll_std_7': roll_std_7,
            'consumption_roll_mean_14': roll_mean_14, 'recent_stockout_flag': 0
        }
        row = pd.DataFrame([base_row])
        for col in ['outlet_id','ingredient_id','ingredient_category']:
            row[col] = row[col].astype('category')
            row[col] = row[col].cat.set_categories(df[col].cat.categories)

        predicted_consumption = max(0, demand_model.predict(row[demand_feature_cols])[0])
        consumption_history.append(predicted_consumption)

        stock_row = row.copy()
        stock_row['closing_stock_lag_1'] = stock
        stock_row['stock_to_demand_ratio'] = stock / (roll_mean_7 + 1e-6)
        stockout_proba = stockout_model.predict_proba(stock_row[stockout_feature_cols])[0][1]

        stock = stock - predicted_consumption
        stockout_today = stock <= 0
        if stockout_today and stockout_date is None:
            stockout_date = future_date

        results.append({
            'date': future_date, 'predicted_consumption': round(predicted_consumption, 1),
            'projected_stock': round(max(0, stock), 1),
            'stockout_probability': round(stockout_proba, 3), 'stockout': stockout_today
        })

        if stockout_today:
            stock = 0

        if assume_reordering:
            reorder_point = roll_mean_7 * (policy['lead_time'] + policy['safety_days'])
            order_qty = roll_mean_7 * policy['order_cover_days']
            if stock < reorder_point and not pending_orders:
                arrival = future_date + pd.Timedelta(days=policy['lead_time'])
                pending_orders[arrival] = pending_orders.get(arrival, 0) + order_qty

    return pd.DataFrame(results), stockout_date


@st.cache_data(show_spinner="Scanning all outlets and ingredients...")
def compute_full_risk_scan(horizon_days=7):
    combos = df[['outlet_id','outlet_name','ingredient_id','ingredient_name','ingredient_category']].drop_duplicates()
    results = []
    for _, combo in combos.iterrows():
        hist = df[(df.outlet_id == combo.outlet_id) & (df.ingredient_id == combo.ingredient_id)].sort_values('date')
        last_known_stock = hist['closing_stock'].iloc[-1]
        fdf, stockout_date = forecast_and_deplete(combo.outlet_id, combo.ingredient_id, last_known_stock,
                                                    horizon_days, assume_reordering=True)
        max_proba = fdf['stockout_probability'].max()
        results.append({
            'outlet_name': combo.outlet_name, 'ingredient_name': combo.ingredient_name,
            'ingredient_category': combo.ingredient_category, 'last_known_stock': last_known_stock,
            'stockout_within_horizon': stockout_date is not None,
            'predicted_stockout_date': stockout_date,
            'days_until_stockout': (stockout_date - fdf['date'].iloc[0] + pd.Timedelta(days=1)).days if stockout_date is not None else None,
            'max_stockout_probability': round(max_proba, 3)
        })
    return pd.DataFrame(results)


outlets = df[['outlet_id','outlet_name']].drop_duplicates().sort_values('outlet_id')
ingredients = df[['ingredient_id','ingredient_name','unit']].drop_duplicates().sort_values('ingredient_name')

st.title("🍛 Restaurant Inventory & Stockout Forecast")
st.caption("Demo system for a South Indian restaurant chain — predicts ingredient depletion and stockout risk using historical demand patterns.")

tab1, tab2 = st.tabs(["📋 Single Item Lookup", "📊 Chain-Wide Overview"])

with tab1:
    with st.sidebar:
        st.header("Enter Stock Check")
        outlet_choice = st.selectbox("Outlet", outlets['outlet_name'].tolist(), key='single_outlet')
        outlet_id = outlets[outlets.outlet_name == outlet_choice]['outlet_id'].iloc[0]

        ingredient_choice = st.selectbox("Ingredient", ingredients['ingredient_name'].tolist(), key='single_ingredient')
        ing_row = ingredients[ingredients.ingredient_name == ingredient_choice].iloc[0]
        ingredient_id = ing_row['ingredient_id']
        unit = ing_row['unit']

        current_stock = st.number_input(f"Current Stock ({unit})", min_value=0.0, value=10000.0, step=100.0)
        horizon_days = st.slider("Forecast horizon (days)", min_value=3, max_value=14, value=7)
        assume_reorder = st.checkbox("Assume normal reordering continues", value=True,
                                       help="If checked, simulates your usual reorder policy during the forecast window (more realistic). If unchecked, shows worst-case depletion with no restocking.")

        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    if predict_btn:
        forecast_df, stockout_date = forecast_and_deplete(outlet_id, ingredient_id, current_stock, horizon_days, assume_reorder)

        if stockout_date is not None:
            st.error(f"⚠️ **STOCKOUT ALERT**: {ingredient_choice} at {outlet_choice} is projected to run out on **{stockout_date.date()}** "
                      f"(within the {horizon_days}-day forecast window). Reorder now.")
        else:
            st.success(f"✅ No stockout predicted for {ingredient_choice} at {outlet_choice} within the next {horizon_days} days.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['projected_stock'],
                                   mode='lines+markers', name='Projected Stock', line=dict(color='steelblue', width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Stock = 0 (Stockout)")
        if stockout_date is not None:
            fig.add_vline(x=stockout_date, line_dash="dot", line_color="crimson",
                           annotation_text="Predicted Stockout", annotation_position="top")
        fig.update_layout(title=f"{horizon_days}-Day Inventory Forecast — {ingredient_choice} @ {outlet_choice}",
                            xaxis_title="Date", yaxis_title=f"Projected Stock ({unit})", height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily Forecast Detail")
        display_df = forecast_df.copy()
        display_df['stockout_probability'] = (display_df['stockout_probability'] * 100).round(1).astype(str) + '%'
        display_df['stockout'] = display_df['stockout'].map({True: '🔴 Yes', False: '🟢 No'})
        display_df.columns = ['Date', f'Predicted Consumption ({unit})', f'Projected Stock ({unit})', 'Stockout Risk', 'Stockout']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"Selected: **{ingredient_choice}** at **{outlet_choice}**, current stock = **{current_stock:,.0f} {unit}**. Click **Predict** to run the forecast.")


with tab2:
    st.subheader("Chain-Wide Risk Overview (7-day horizon)")
    st.caption("Scans every ingredient at every outlet using their last known stock level. Assumes normal reordering continues.")

    overview_df = compute_full_risk_scan(horizon_days=7)
    at_risk = overview_df[overview_df['stockout_within_horizon']]

    # ---- Today's Alert Panel (urgency-sorted) ----
    st.markdown("### 🔔 Today's Alerts — Items Nearing Stockout")
    urgent = at_risk.sort_values('days_until_stockout').copy()

    if urgent.empty:
        st.success("No items are projected to stock out within the next 7 days. All clear.")
    else:
        for _, item in urgent.head(10).iterrows():
            days = item['days_until_stockout']
            if days <= 1:
                icon, level = "🔴", "URGENT"
            elif days <= 3:
                icon, level = "🟠", "SOON"
            else:
                icon, level = "🟡", "WATCH"
            st.markdown(
                f"{icon} **[{level}]** **{item['ingredient_name']}** at **{item['outlet_name']}** — "
                f"projected to run out in **{days} day{'s' if days != 1 else ''}** "
                f"({item['predicted_stockout_date'].strftime('%b %d')}), category: {item['ingredient_category']}"
            )
        if len(urgent) > 10:
            st.caption(f"...and {len(urgent) - 10} more at-risk items (see full table below).")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Items Tracked", len(overview_df))
    col2.metric("At-Risk Items (7d)", len(at_risk), delta=f"{len(at_risk)/len(overview_df)*100:.0f}% of chain", delta_color="inverse")
    col3.metric("Outlets", overview_df['outlet_name'].nunique())

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**At-Risk Items by Outlet**")
        risk_by_outlet = overview_df.groupby('outlet_name')['stockout_within_horizon'].sum().reset_index()
        risk_by_outlet.columns = ['Outlet', 'At-Risk Count']
        fig_outlet = px.bar(risk_by_outlet, x='Outlet', y='At-Risk Count', color='Outlet',
                              text='At-Risk Count', height=350)
        fig_outlet.update_layout(showlegend=False)
        st.plotly_chart(fig_outlet, use_container_width=True)

    with c2:
        st.markdown("**At-Risk Items by Ingredient Category**")
        risk_by_cat = overview_df.groupby('ingredient_category')['stockout_within_horizon'].sum().reset_index()
        risk_by_cat.columns = ['Category', 'At-Risk Count']
        risk_by_cat = risk_by_cat.sort_values('At-Risk Count', ascending=True)
        fig_cat = px.bar(risk_by_cat, x='At-Risk Count', y='Category', orientation='h',
                           color='At-Risk Count', color_continuous_scale='Oranges', height=350)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("---")
    st.markdown("**🚨 Top 15 Highest-Risk Items (Chain-Wide)**")
    top_risk = overview_df.sort_values('max_stockout_probability', ascending=False).head(15).copy()
    top_risk['max_stockout_probability'] = (top_risk['max_stockout_probability'] * 100).round(1).astype(str) + '%'
    top_risk['stockout_within_horizon'] = top_risk['stockout_within_horizon'].map({True: '🔴', False: '🟢'})
    top_risk_display = top_risk[['outlet_name','ingredient_name','ingredient_category','max_stockout_probability','stockout_within_horizon']]
    top_risk_display.columns = ['Outlet', 'Ingredient', 'Category', 'Peak Risk %', 'At Risk']
    st.dataframe(top_risk_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**📈 Demand Forecast Accuracy — Actual vs Predicted (Test Period)**")
    acc_col1, acc_col2 = st.columns(2)
    with acc_col1:
        acc_outlet = st.selectbox("Outlet", outlets['outlet_name'].tolist(), key='acc_outlet')
    with acc_col2:
        acc_ingredient = st.selectbox("Ingredient", ingredients['ingredient_name'].tolist(), key='acc_ingredient')

    acc_outlet_id = outlets[outlets.outlet_name == acc_outlet]['outlet_id'].iloc[0]
    acc_ingredient_id = ingredients[ingredients.ingredient_name == acc_ingredient]['ingredient_id'].iloc[0]

    acc_data = demand_test[(demand_test.outlet_id == acc_outlet_id) & (demand_test.ingredient_id == acc_ingredient_id)].sort_values('date')

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=acc_data['date'], y=acc_data['consumption'], mode='lines', name='Actual', line=dict(color='steelblue')))
    fig_acc.add_trace(go.Scatter(x=acc_data['date'], y=acc_data['predicted_consumption'], mode='lines', name='Predicted', line=dict(color='orange', dash='dash')))
    fig_acc.update_layout(title=f"Actual vs Predicted Consumption — {acc_ingredient} @ {acc_outlet}",
                            xaxis_title="Date", yaxis_title="Consumption", height=400)
    st.plotly_chart(fig_acc, use_container_width=True)

    mape = (abs(acc_data['consumption'] - acc_data['predicted_consumption']) / acc_data['consumption']).mean() * 100
    st.caption(f"MAPE for this item in the test period: {mape:.1f}%")