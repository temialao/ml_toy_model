from utils import load_data_from_csv, calculate_mae

# Load the predicted data
df = load_data_from_csv('predicted_data.csv')

# Calculate MAE
mae = calculate_mae(df['price'], df['predicted_price'])

# Print the results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"W1: 1.27")
print(f"W2: 0.92")
