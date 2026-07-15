import pandas as pd
import numpy as np

np.random.seed(42)

bom = pd.read_csv('data/raw/dish_ingredient_bom.csv')
dishes = bom[['dish_id','dish_name','dish_category']].drop_duplicates().reset_index(drop=True)

outlets = pd.DataFrame({
    'outlet_id': ['O01','O02','O03'],
    'outlet_name': ['Koramangala - Office Hub', 'Jayanagar - Residential', 'Whitefield - Mixed'],
    'profile': ['office','residential','mixed']
})

start_date = '2023-01-01'
end_date = '2024-12-31'
dates = pd.date_range(start_date, end_date, freq='D')

# Festival dates (approx, South Indian relevant festivals) with dish boost map
festivals = {
    '2023-01-15': ('Pongal', {'D07':4.0,'D09':2.0,'D12':1.5}),
    '2023-04-14': ('Ugadi', {'D18':3.0,'D08':2.0}),
    '2023-08-29': ('Onam', {'D09':2.5,'D13':2.0,'D18':2.5}),
    '2023-11-12': ('Diwali', {'D18':3.0,'D08':2.5,'D14':1.8}),
    '2024-01-15': ('Pongal', {'D07':4.0,'D09':2.0,'D12':1.5}),
    '2024-04-09': ('Ugadi', {'D18':3.0,'D08':2.0}),
    '2024-09-15': ('Onam', {'D09':2.5,'D13':2.0,'D18':2.5}),
    '2024-11-01': ('Diwali', {'D18':3.0,'D08':2.5,'D14':1.8}),
}
festival_dates = {pd.Timestamp(k): v for k, v in festivals.items()}

# base daily demand per dish (plates/day), varies by category
base_demand = {
    'Breakfast': 90, 'Lunch': 70, 'Beverage': 120, 'Dessert': 40
}

# outlet multiplier by profile
outlet_mult = {'office': 1.2, 'residential': 0.9, 'mixed': 1.0}

# weekday pattern per category: Mon=0 ... Sun=6
weekday_pattern = {
    'Breakfast': [1.0,1.0,1.0,1.0,1.0,1.15,1.2],
    'Lunch':     [1.1,1.1,1.1,1.1,1.15,0.85,0.8],
    'Beverage':  [0.9,0.9,0.9,0.9,1.0,1.2,1.3],
    'Dessert':   [0.7,0.7,0.7,0.7,0.9,1.4,1.5],
}
# office outlets skew weekday lunch even more; residential skews weekend
def profile_adjust(profile, category, wd):
    mult = 1.0
    if profile == 'office' and category == 'Lunch' and wd < 5:
        mult *= 1.25
    if profile == 'office' and wd >= 5:
        mult *= 0.7
    if profile == 'residential' and wd >= 5:
        mult *= 1.3
    return mult

rows = []
for _, outlet in outlets.iterrows():
    for _, dish in dishes.iterrows():
        cat = dish['dish_category']
        b = base_demand[cat]
        # slight per-dish popularity variation (fixed per dish)
        dish_pop = np.random.uniform(0.6, 1.4)
        for date in dates:
            wd = date.weekday()
            trend = 1 + 0.00015 * (date - pd.Timestamp(start_date)).days  # slow growth over 2 years
            seasonal = weekday_pattern[cat][wd]
            prof_adj = profile_adjust(outlet['profile'], cat, wd)
            fest_mult = 1.0
            fest_name = None
            if date in festival_dates:
                fname, boosts = festival_dates[date]
                fest_mult = boosts.get(dish['dish_id'], 1.3)  # small general bump even if not specific
                fest_name = fname
            expected = b * dish_pop * seasonal * prof_adj * fest_mult * trend * outlet_mult[outlet['profile']]
            noise = np.random.normal(1.0, 0.12)
            units_sold = max(0, int(round(expected * noise)))
            rows.append([date, outlet['outlet_id'], outlet['outlet_name'], dish['dish_id'], dish['dish_name'],
                         dish['dish_category'], wd, units_sold, fest_name])

sales = pd.DataFrame(rows, columns=['date','outlet_id','outlet_name','dish_id','dish_name',
                                     'dish_category','weekday','units_sold','festival'])
sales.to_csv('data/raw/dish_sales_daily.csv', index=False)
print(sales.shape)
print(sales.head())
print(sales.groupby('outlet_name')['units_sold'].sum())