import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample house data
data = {
    'size_sqft': [800, 1000, 1200, 1500, 1800],
    'price': [4000000, 5000000, 6000000, 7500000, 9000000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input (X) and Output (y)
X = df[['size_sqft']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict price for new house
predicted_price = model.predict([[1600]])

print("Predicted House Price:", int(predicted_price[0]))
