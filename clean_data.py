import pandas as pd

# Load original CSV from Kaggle
df = pd.read_csv("data/gold_price_india_current.csv")
#check the columns
print("Columns in CSV:", df.columns.tolist())

# Convert column names if needed (actual column names may vary)
# For example, if it's "Date" and "Price (INR/10g)"
df.rename(columns={"Date": "ds", "Price": "y"}, inplace=True)

print("Columns in CSV:", df.columns.tolist())
# Convert to datetime
df['ds'] = pd.to_datetime(df['ds'], dayfirst=True , errors='coerce')
df.dropna(subset=['ds'], inplace=True)
df.dropna(subset=['y'], inplace=True)

#remove any dates greater than current date
from datetime import datetime

today = datetime.today()
df = df[df['ds'] <= today]

# Convert to ₹ per gram (if it's per 10g)
#df['y'] = df['y'] / 10(only do this if not already done in the dataset)

# Sort in ascending order if needed
df = df.sort_values('ds')
print("last date in dataset:", df['ds'].max())
# Save cleaned file
df.to_csv("data/gold_price_india_cleaned.csv", index=False)
