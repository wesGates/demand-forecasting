from src.data import load_m5, trim_leading_zeros

df = load_m5("data", store_id="CA_1", cat_id="FOODS")
df = trim_leading_zeros(df)