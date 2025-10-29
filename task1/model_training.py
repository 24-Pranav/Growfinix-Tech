import pandas as pd
import numpy as np
import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("📂 Loading dataset...")
df = pd.read_csv("data.csv")
print("✅ Dataset loaded. Shape:", df.shape)


print("⚙️ Preprocessing dataset...")

current_year = datetime.datetime.now().year
df["house_age"] = current_year - df["yr_built"]
df["years_since_renovation"] = np.where(
    df["yr_renovated"] == 0,
    0,
    current_year - df["yr_renovated"]
)

drop_cols = ["date", "yr_built", "yr_renovated", "street", "country"]
df = df.drop(columns=drop_cols)

label_enc = LabelEncoder()
for col in ["city", "statezip"]:
    df[col] = label_enc.fit_transform(df[col])

df = df[df["price"] < 5000000]


X = df.drop("price", axis=1)
y = df["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("🤖 Training models...")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "R2": r2}
    print(f"{name}: RMSE={rmse:.2f}, R²={r2:.2f}")

    filename = f"{name.replace(' ', '_')}.pkl"
    joblib.dump(model, filename, protocol=4)
    print(f"✅ Saved: {filename}")

results_df = pd.DataFrame(results).T
results_df.to_csv("model_results.csv")
print("\n📊 Model Results saved to model_results.csv")
print(results_df)
