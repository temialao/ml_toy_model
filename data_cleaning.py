import pandas as pd
from utils import load_data_from_csv, round_dataframe, resample_to_business_days, save_data_to_csv

# Load the data
df = load_data_from_csv('aluminium_exercise_data.csv')

# Convert the date column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Set the date as the index
df.set_index('date', inplace=True)

# Resample to business days
df = resample_to_business_days(df)

# Round the data
df = round_dataframe(df)

# Save the cleaned data
save_data_to_csv(df, 'cleaned_data.csv')
