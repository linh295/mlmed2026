import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# LOAD DATASET
train_df = pd.read_csv("training_set_pixel_size_and_HC.csv")

X = train_df[["pixel size(mm)"]].values
y = train_df["head circumference (mm)"].values

print("Dataset shape:", train_df.shape)
print(train_df.head())

# TRAIN / VALIDATION SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# BASELINE MODEL
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_val)
baseline_mae = mean_absolute_error(y_val, y_pred)

print(f"Baseline Linear Regression MAE: {baseline_mae:.3f}")

# POLYNOMIAL REGRESSION
for degree in [2, 3, 4, 5]:
    poly_model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    poly_model.fit(X_train, y_train)
    y_pred = poly_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    print(f"Polynomial degree {degree} - MAE: {mae:.3f}")

# GRADIENT BOOSTING REGRESSION
best_mae = np.inf
best_params = None

for lr in [0.05, 0.1, 0.2]:
    for depth in [2, 3, 4]:
        gbr = GradientBoostingRegressor(
            learning_rate=lr,
            max_depth=depth,
            random_state=42
        )

        gbr.fit(X_train, y_train)
        y_pred = gbr.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)

        print(f"GBR lr={lr}, depth={depth} → MAE: {mae:.3f}")

        if mae < best_mae:
            best_mae = mae
            best_params = (lr, depth)

print("\nBest Gradient Boosting configuration:")
print(f"Learning rate = {best_params[0]}, Max depth = {best_params[1]}")
print(f"Best Validation MAE = {best_mae:.3f}")

# TRAIN FINAL MODEL
final_model = GradientBoostingRegressor(
    learning_rate=best_params[0],
    max_depth=best_params[1],
    random_state=42
)

final_model.fit(X, y)

print("\nFinal model trained on full training set.")
