import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("coffee_price_thailand.csv")

X = data.drop("price", axis=1)
y = data["price"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "model.pkl")

print("✅ Model trained & saved!")