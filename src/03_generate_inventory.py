import pandas as pd
import numpy as np

np.random.seed(7)

cons = pd.read_csv('data/processed/ingredient_consumption_daily.csv', parse_dates=['date'])
bom = pd.read_csv('data/raw/dish_ingredient_bom.csv')

ingredients = bom[['ingredient_id','ingredient_name','ingredient_category','unit']].drop_duplicates().reset_index(drop=True)

# category -> (lead_time_days, safety_days, order_multiple_days)
# tighter buffers than a first pass - real restaurants don't hold huge safety stock
# (perishables especially can't - storage + spoilage constraints)
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

# lead time variability: supplier occasionally delivers late (real-world friction)
def jittered_lead_time(base_lead_time):
    if np.random.random() < 0.12:  # 12% chance of a delayed delivery
        return base_lead_time + np.random.randint(1, 3)
    return base_lead_time

results = []

for outlet_id in cons.outlet_id.unique():
    outlet_cons = cons[cons.outlet_id == outlet_id]
    outlet_name = outlet_cons.outlet_name.iloc[0]

    for _, ing in ingredients.iterrows():
        ing_id = ing['ingredient_id']
        policy = category_policy[ing['ingredient_category']]
        lead_time = policy['lead_time']
        safety_days = policy['safety_days']
        order_cover_days = policy['order_cover_days']

        series = outlet_cons[outlet_cons.ingredient_id == ing_id].sort_values('date').reset_index(drop=True)
        if series.empty:
            continue

        # rolling 14-day historical average, shifted by 1 day (no lookahead)
        # first 14 days use expanding mean as fallback
        rolling_avg = series['consumption'].shift(1).rolling(window=14, min_periods=1).mean()
        overall_avg = series['consumption'].mean()
        rolling_avg = rolling_avg.fillna(overall_avg)

        stock = overall_avg * (lead_time + safety_days + order_cover_days)  # starting stock, reasonably stocked
        pending_orders = {}  # date -> qty arriving

        dates = series['date'].tolist()
        consumptions = series['consumption'].tolist()
        rolling_avgs = rolling_avg.tolist()

        for i, date in enumerate(dates):
            # receive any pending order arriving today
            if date in pending_orders:
                stock += pending_orders.pop(date)

            avg_daily_consumption = max(rolling_avgs[i], 1e-6)
            reorder_point = avg_daily_consumption * (lead_time + safety_days)
            order_qty = avg_daily_consumption * order_cover_days

            demand = consumptions[i]
            fulfilled = min(stock, demand)
            unmet_demand = demand - fulfilled
            stock = max(0, stock - demand)
            stockout_flag = 1 if unmet_demand > 0 else 0

            # place order if stock below reorder point (and no order already pending)
            order_placed = 0
            if stock < reorder_point and not pending_orders:
                actual_lead_time = jittered_lead_time(lead_time)
                arrival_date = date + pd.Timedelta(days=actual_lead_time)
                pending_orders[arrival_date] = pending_orders.get(arrival_date, 0) + order_qty
                order_placed = 1

            results.append([
                date, outlet_id, outlet_name, ing_id, ing['ingredient_name'], ing['ingredient_category'], ing['unit'],
                round(demand, 1), round(stock, 1), stockout_flag, round(unmet_demand, 1),
                order_placed, round(reorder_point, 1)
            ])

inventory = pd.DataFrame(results, columns=[
    'date','outlet_id','outlet_name','ingredient_id','ingredient_name','ingredient_category','unit',
    'consumption','closing_stock','stockout_flag','unmet_demand','order_placed','reorder_point'
])

inventory.to_csv('data/processed/inventory_daily.csv', index=False)

print(inventory.shape)
print('\nOverall stockout rate:', round(inventory.stockout_flag.mean()*100, 2), '%')
print('\nStockout rate by ingredient category:')
print(inventory.groupby('ingredient_category')['stockout_flag'].mean().sort_values(ascending=False)*100)
print('\nSample rows:')
print(inventory.head(10))

