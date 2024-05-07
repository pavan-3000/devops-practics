import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib 
# Generate some random data for demonstration purposes
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 3 + 4 * X + np.random.randn(100, 1)

print(X)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)



joblib.dump(model,"model.pkl")

