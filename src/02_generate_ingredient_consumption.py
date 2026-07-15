import pandas as pd

sales = pd.read_csv('data/raw/dish_sales_daily.csv', parse_dates=['date'])
bom = pd.read_csv('data/raw/dish_ingredient_bom.csv')

# merge sales with BOM (each dish sale explodes into its ingredient rows)
merged = sales.merge(bom[['dish_id','ingredient_id','ingredient_name','ingredient_category',
                           'quantity_per_plate','unit']], on='dish_id', how='left')

# consumption = units_sold * quantity_per_plate
merged['consumption'] = merged['units_sold'] * merged['quantity_per_plate']

# aggregate to ingredient level per outlet per day
ingredient_daily = merged.groupby(
    ['date','outlet_id','outlet_name','ingredient_id','ingredient_name','ingredient_category','unit'],
    as_index=False
)['consumption'].sum()

ingredient_daily = ingredient_daily.sort_values(['outlet_id','ingredient_id','date']).reset_index(drop=True)
ingredient_daily.to_csv('data/processed/ingredient_consumption_daily.csv', index=False)

print(ingredient_daily.shape)
print(ingredient_daily.head(10))

# quick check: rice (I01) consumption for one outlet one day
check = ingredient_daily[(ingredient_daily.ingredient_id=='I01') & (ingredient_daily.outlet_id=='O01') & (ingredient_daily.date=='2023-01-02')]
print('\nSample check - Idli Rice (I01) consumption on 2023-01-02, Outlet O01:')
print(check)