import numpy as np
import pandas as pd
import statsmodels.api as sm

# Set parameters
n = 1000
pe = 0.5  # Proportion of male
maf = 0.5  # Probability of A allele
p2 = maf * maf # AA
p0 = (1 - maf) * (1 - maf) # aa
p1 = 1 - p0 - p2 # Aa

be = 0.0
bz = 0.15 # continuous E;
bx1 = 0.1 # additive coding of g1;
bex1 = -0.2 # crossing interaction for male*x1 with additive coding for g1;
bx4 = 0.15 # dominant coding of g4;
bx5 = 0.15 # recessive coding of g5;
bzx4 = 0.15 # interaction of z with g4 (dominant);
bzx5 = 0.15 # interaction of z with g5 (recessive);
sigma = 1.0 # residual variance;

np.random.seed(0)

data = []

for _ in range(n):
    e = np.random.rand() < pe
    z = np.random.normal(0, 1)
    g = np.random.choice([0, 1, 2], size=10, p=[p0, p1, p2])
    x = g - 1  # Shift to additive coding

    x4 = 1 if g[3] in [1, 2] else 0 #  recode x4 based on g4 to be dominant AA,Aa vs others;
    x5 = 1 if g[4] == 2 else 0 #  recode x5 based on g5 to be recessive AA vs others;

    ex1 = e * x[0]
    # ex4 = e * x[3]
    # ex5 = e * x[4]
    zx4 = z * x4
    zx5 = z * x5

    lp = be * e + bz * z + bx1 * x[0] + bx4 * x4 + bx5 * x5 + bex1 * ex1 + bzx4 * zx4 + bzx5 * zx5
    y = lp + sigma * np.random.normal(0, 1)

    data.append([y, e, z] + g.tolist())

df = pd.DataFrame(data, columns=["y", "e", "z"] + [f"g{i + 1}" for i in range(10)])
df["e"] = df["e"].astype(int)
df.to_csv("simulationChallenge3.csv", index=False)

# Regression Analysis
X_main_effects = sm.add_constant(df[["e", "z"] + [f"g{i + 1}" for i in range(10)]])
y = df["y"]
model_main = sm.OLS(y, X_main_effects).fit()
print("LR Model:")
print(model_main.summary())

X_true_model = sm.add_constant(df[["e", "z", "g1"]])
X_true_model["x4"] = df["g4"].apply(lambda g: 1 if g in [1, 2] else 0)
X_true_model["x5"] = df["g5"].apply(lambda g: 1 if g == 2 else 0)
X_true_model["ex1"] = X_true_model["e"] * X_true_model["g1"]
X_true_model["zx4"] = X_true_model["z"] * X_true_model["x4"]
X_true_model["zx5"] = X_true_model["z"] * X_true_model["x5"]

model_true = sm.OLS(y, X_true_model).fit()
print("True Model:")
print(model_true.summary())
