from sklearn.linear_model import LinearRegression
from utils import load_data_from_csv

# Load the cleaned data
df = load_data_from_csv('cleaned_data.csv')

# Extract training data
training_data = df.iloc[:300]

# Define predictors and target variable
X = training_data[['curve', 'inventory']]
y = training_data['price']

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Extract the weights
W1, W2 = model.coef_

# Print the results
print(f"W1: {W1:.2f}")
print(f"W2: {W2:.2f}")
