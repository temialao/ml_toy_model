from utils import load_data_from_csv, save_data_to_csv, round_dataframe

# Load the cleaned data
df = load_data_from_csv('cleaned_data.csv')

# Given weights from the training step
W1 = 1.27
W2 = 0.92

# Extract data points for prediction
prediction_data = df.iloc[300:]

# Calculate predicted prices
prediction_data['predicted_price'] = W1 * prediction_data['curve'] + W2 * prediction_data['inventory']

# Round the predictions
prediction_data['predicted_price'] = round_dataframe(prediction_data['predicted_price'])

# Save the predicted data
save_data_to_csv(prediction_data, 'predicted_data.csv')

# Display the data
print(prediction_data.head())
