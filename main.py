import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/cars_fuel_efficiency.csv")
df = df.dropna(subset=["power_hp"])
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isna().sum())

print("\n3:")
from sklearn.linear_model import LinearRegression

corr = df.corr(numeric_only=True)["fuel_efficiency_km_per_l"]
best = corr.drop("fuel_efficiency_km_per_l").abs().idxmax()

X = df[[best]].values
y = df["fuel_efficiency_km_per_l"].values

model = LinearRegression().fit(X, y)
print(f"{best}: y = {model.intercept_:.3f} + {model.coef_[0]:.3f} * x")

plt.scatter(X, y, s=10)
plt.plot(X, model.predict(X), color="red")
plt.xlabel(best); plt.ylabel("km/l"); plt.show()
print(model.predict([[150]]))

print("\n4:")
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2175
)

for name, Xt, yt in [("train", X_train, y_train), ("test", X_test, y_test)]:
    m = LinearRegression().fit(X_train, y_train)
    pred = m.predict(Xt)
    mae  = (abs(yt - pred)).mean()
    mse  = ((yt - pred)**2).mean()
    r2   = m.score(Xt, yt)
    print(f"{name}: MAE={mae:.3f} MSE={mse:.3f} RMSE={mse**0.5:.3f} R2={r2:.3f}")

cv = cross_val_score(LinearRegression(), X, y, cv=5)
print(f"5-fold R2: mean={cv.mean():.3f} ± {cv.std():.3f}")

print("\n5:")
numeric = ["engine_litres", "power_hp", "mass_kg", "cylinders", "accel_0_100_s"]
for feats in [[best], [best, "power_hp"], numeric]:
    Xi = df[feats].values
    s = cross_val_score(LinearRegression(), Xi, y, cv=5)
    print(f"{feats}: mean R2 = {s.mean():.3f}")

print("\n6:")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

for d in range(1, 6):
    pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=d, include_bias=False),
        LinearRegression(),
    )
    tr = cross_val_score(pipe, X, y, cv=5)
    print(f"degree {d}: mean R2 = {tr.mean():.3f}")