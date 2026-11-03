import pandas as pd
from pathlib import Path
import shutil
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from itertools import product

# Load the IC's (1029 batches)

df = pd.read_csv('datasets/initial_conditions.csv')

print("Total Batches:",len(df))

unique_batches = df.drop_duplicates(subset=["Glucose [mM]","Glutamine [mM]", "VCD [cells/mL]"])

random_30 = unique_batches.sample(n=30, random_state=42)

random_30.to_csv('datasets/initial_conditions_random_30.csv', index=False)

source_folder = Path("datasets")

selected_folder = Path("datasets/selected_batches")
selected_folder.mkdir(exist_ok=True)

for batch_id in random_30["Batch ID"]:
    filename = f"batch_{int(batch_id):04d}.csv"
    shutil.copy2(source_folder / filename, selected_folder / filename)


print("Selected batch files:", len(list(selected_folder.glob("batch_*.csv"))))

# Store the best result from each batch
results = []

# Loop through all 30 selected batches
for _, row in random_30.iterrows():

    batch_id = int(row["Batch ID"])
    filename = f"batch_{batch_id:04d}.csv"
    batch = pd.read_csv(selected_folder / filename)

    # Initial conditions from initial_conditions.csv
    glucose = row["Glucose [mM]"]
    glutamine = row["Glutamine [mM]"]
    vcd = row["VCD [cells/mL]"]

    # Operating cost ($/L)
    cost = 10 * glucose + 20 * glutamine + 0.0005 * vcd

    # Calculate economics at every harvest time
    batch["Value [$/L]"] = 1000 * batch["mAb [mg/L]"]
    batch["Profit [$/L]"] = batch["Value [$/L]"] - cost

    batch["Space-Time Profit [$/L/day]"] = (
        batch["Profit [$/L]"] /
        (batch["Time [h]"] / 24 + 1)
    )

    # Find the best observed harvest time
    best_index = batch["Space-Time Profit [$/L/day]"].idxmax()
    best = batch.loc[best_index]

    # Save best result
    results.append({"Batch ID": batch_id,"Glucose [mM]": glucose,"Glutamine [mM]": glutamine,"Initial VCD [cells/mL]": vcd,"Harvest Time [h]": best["Time [h]"],"mAb [mg/L]": best["mAb [mg/L]"],"Cost [$/L]": cost,"Value [$/L]": best["Value [$/L]"],"Profit [$/L]": best["Profit [$/L]"],"Space-Time Profit [$/L/day]": best["Space-Time Profit [$/L/day]"]})

# Convert the results into a table
results_df = pd.DataFrame(results)

print("\nBest observed harvest conditions for all 30 batches:")
print(results_df.to_string(index=False))


# Save the best observed economic results for all 30 batches
results_df.to_csv("tables/observed_batch_economics.csv", index=False)


# Combine all 30 batches into one dataset
combined_batches = []

for _, row in random_30.iterrows():

    batch_id = int(row["Batch ID"])
    filename = f"batch_{batch_id:04d}.csv"

    batch = pd.read_csv(selected_folder / filename)

    # Add the initial operating conditions
    batch["Batch ID"] = batch_id
    batch["Initial Glucose [mM]"] = row["Glucose [mM]"]
    batch["Initial Glutamine [mM]"] = row["Glutamine [mM]"]
    batch["Initial VCD [cells/mL]"] = row["VCD [cells/mL]"]

    combined_batches.append(batch)

# Combine the individual DataFrames
combined_df = pd.concat(combined_batches, ignore_index=True)

# Save the combined dataset
combined_df.to_csv("datasets/combined_30_batches.csv", index=False)

print("\nCombined dataset shape:", combined_df.shape)
print(combined_df.head())


# Polynomial Reg

# Import the data used for ml
df = pd.read_csv(os.path.join("datasets", "combined_30_batches.csv"))

features = ["Initial Glucose [mM]", "Initial Glutamine [mM]", "Initial VCD [cells/mL]", "Time [h]"]

target = "mAb [mg/L]"

x = df[features].to_numpy()
y = df[target].to_numpy()

groups = df["Batch ID"].to_numpy()

print("Features shape:", x.shape)
print("Target shape:", y.shape)
print("Number of batches:",len(np.unique(groups)))

batch_ids = np.unique(groups)

batch_train, batch_test = train_test_split(batch_ids, test_size=0.2, random_state=0)

train_mask = np.isin(groups, batch_train)
test_mask = np.isin(groups, batch_test)

x_train = x[train_mask]
x_test = x[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

groups_train = groups[train_mask]

print("Training batches:", len(batch_train))
print("Testing batches:", len(batch_test))
print("Training observations:", len(x_train))
print("Testing observations:", len(x_test))


# actual poly reg
model_poly = make_pipeline(StandardScaler(),PolynomialFeatures(include_bias=False),LinearRegression())

# define polynomial degrees to test
param_grid_poly = dict(polynomialfeatures__degree=[1, 2, 3, 4])

# Hyperparameter optimization using grouped 4-fold CV
group_cv = GroupKFold(n_splits=4)

model_gscv_poly = GridSearchCV(model_poly,param_grid=param_grid_poly,cv=group_cv,scoring="neg_root_mean_squared_error",return_train_score=True)

# Fitting the model using training data
model_gscv_poly.fit(x_train, y_train, groups=groups_train)

# Extracted and sorted cv results
df_results_poly = pd.DataFrame(model_gscv_poly.cv_results_)
df_results_poly = df_results_poly.sort_values(by=["rank_test_score"], ascending=True)

# Column names
columns_view_poly = ["rank_test_score","param_polynomialfeatures__degree","mean_test_score","std_test_score","mean_train_score"]

print()
print("Polynomial Regression")
print(df_results_poly.loc[:, columns_view_poly].to_string())

print()
print(f"Best hyperparameters: {model_gscv_poly.best_params_}")
print(f"Best CV RMSE: {-model_gscv_poly.best_score_:.5f} mg/L")


# Evaluate the best polynomial model on the test data
y_test_pred = model_gscv_poly.predict(x_test)

m_test_rmse = root_mean_squared_error(y_test, y_test_pred)
m_test_r2 = r2_score(y_test, y_test_pred)

print()
print("Polynomial Regression - Test Performance")
print(f"Test RMSE: {m_test_rmse:.5f} mg/L")
print(f"Test R^2: {m_test_r2:.5f}")


# KNN Reg
model_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())

# Define the grid to search
param_grid = dict(kneighborsregressor__n_neighbors=[i for i in range(1, 81)],kneighborsregressor__weights=["uniform", "distance"])

# Create the model using GridSearchCV
model_gscv = GridSearchCV(model_knn, param_grid=param_grid, cv=group_cv, scoring="neg_root_mean_squared_error", return_train_score=True)

model_gscv.fit(x_train, y_train, groups=groups_train)

# Extracted and sorted cv results
df_results = pd.DataFrame(model_gscv.cv_results_)
df_results = df_results.sort_values(by=["rank_test_score"], ascending=True)

# Column names
columns_view = ["rank_test_score", "param_kneighborsregressor__n_neighbors", "param_kneighborsregressor__weights", "mean_test_score", "std_test_score", "mean_train_score"]

print()
print("KNN Regression")
print(df_results.loc[:, columns_view].to_string())

print()
print(f"Best hyperparameters: {model_gscv.best_params_}")
print(f"Best CV RMSE: {-model_gscv.best_score_:.5f} mg/L")

# Evaluate KNN on the test data
y_test_pred_knn = model_gscv.predict(x_test)

m_test_rmse_knn = root_mean_squared_error(y_test, y_test_pred_knn)
m_test_r2_knn = r2_score(y_test, y_test_pred_knn)

print()
print("KNN Regression - Test Performance")
print(f"Test RMSE: {m_test_rmse_knn:.5f} mg/L")
print(f"Test R²: {m_test_r2_knn:.5f}")


# Compare polynomial regression and KNN
df_comparison = pd.DataFrame({"Model": ["Polynomial Regression", "KNN Regression"],"CV RMSE [mg/L]": [-model_gscv_poly.best_score_,-model_gscv.best_score_],"Test RMSE [mg/L]": [m_test_rmse,m_test_rmse_knn],"Test R2": [m_test_r2,m_test_r2_knn]})

df_comparison = df_comparison.sort_values(by="CV RMSE [mg/L]",ascending=True)

print("\nModel Comparison")
print(df_comparison.to_string(index=False))

df_comparison.to_csv("tables/model_comparison.csv",index=False)

# Define the operating conditions to evaluate
glucose_levels = np.arange(5, 36, 5)
glutamine_levels = np.arange(1, 8, 1)
vcd_levels = np.arange(200000, 1400001, 200000)
harvest_times = np.arange(0, 673, 8)

combinations = list(product(glucose_levels,glutamine_levels,vcd_levels,harvest_times))

conditions = pd.DataFrame(combinations, columns=features)

# Predict mAb concentration for every operating combination
conditions["Predicted mAb [mg/L]"] = model_gscv_poly.predict(conditions[features].to_numpy())

print(conditions.head())


# Calculate culture cost ($/L)
conditions["Cost [$/L]"] = (10 * conditions["Initial Glucose [mM]"]+ 20 * conditions["Initial Glutamine [mM]"]+ 0.0005 * conditions["Initial VCD [cells/mL]"])

# Calculate culture value($/L)
conditions["Value [$/L]"] = (1000 * conditions["Predicted mAb [mg/L]"])

# Calculate profit ($/L)
conditions["Profit [$/L]"] = (conditions["Value [$/L]"] - conditions["Cost [$/L]"])

# Calculate space-time profit ($/L/day)
conditions["Space-Time Profit [$/L/day]"] = (conditions["Profit [$/L]"]/ (conditions["Time [h]"] / 24 + 1))

# Calculate space-time yield (mg/L/day)
conditions["Space-Time Yield [mg/L/day]"] = (conditions["Predicted mAb [mg/L]"] / (conditions["Time [h]"] / 24 + 1))

print(conditions.head())


# Find the operating conditions with maximum space-time profit
best_index = conditions["Space-Time Profit [$/L/day]"].idxmax()

best_conditions = conditions.loc[best_index]

print("\nOptimal Operating Conditions")
print(best_conditions.to_string())


# RNG
rng = np.random.default_rng(0)

# # of simulations per foreast
n_simulations = 1000


# Predict mAb concentrations using the training data
y_train_pred = model_gscv_poly.predict(x_train)

# Calculate residual errors
train_noise = y_train_pred - y_train

print("\nTraining Residuals")
print("Mean residual:", np.mean(train_noise))
print("Residual standard deviation:", np.std(train_noise))


# Predicted mAb concentration at optimal conditions
optimal_mab = best_conditions["Predicted mAb [mg/L]"]

# Sample residuals to simulate uncertainty
simulated_mab = []

for i in range(n_simulations):
    white_noise = rng.choice(train_noise)
    mab = optimal_mab + white_noise

    # Prevent negative concentrations
    mab = max(0, mab)

    simulated_mab.append(mab)

simulated_mab = np.array(simulated_mab)


# Calculate the 95% prediction interval
prediction_interval = 0.95
alpha = 1 - prediction_interval

mab_lower = np.quantile(simulated_mab, q=alpha / 2)
mab_upper = np.quantile(simulated_mab, q=1 - alpha / 2)

print("\nPredicted mAb Concentration")
print(f"Predicted mAb: {optimal_mab:.5f} mg/L")
print(f"Approximate 95% PI: [{mab_lower:.5f}, {mab_upper:.5f}] mg/L")


# Calculate simulated culture values and profits
simulated_value = 1000 * simulated_mab
simulated_profit = simulated_value - best_conditions["Cost [$/L]"]

# Calculate approximate 95% prediction interval for profit
profit_lower = np.quantile(simulated_profit, q=alpha / 2)
profit_upper = np.quantile(simulated_profit, q=1 - alpha / 2)

print("\nPredicted Culture Profit")
print(f"Predicted profit: {best_conditions['Profit [$/L]']:.2f} $/L")
print(f"Approximate 95% PI: [{profit_lower:.2f}, {profit_upper:.2f}] $/L")


# Calculate total cycle time in days
cycle_time = best_conditions["Time [h]"] / 24 + 1

# Calculate simulated space-time profit and yield
simulated_st_profit = simulated_profit / cycle_time
simulated_st_yield = simulated_mab / cycle_time

# Calculate approximate 95% prediction intervals
st_profit_lower = np.quantile(simulated_st_profit, alpha / 2)
st_profit_upper = np.quantile(simulated_st_profit, 1 - alpha / 2)

st_yield_lower = np.quantile(simulated_st_yield, alpha / 2)
st_yield_upper = np.quantile(simulated_st_yield, 1 - alpha / 2)

print("\nSpace-Time Profit")
print(f"Predicted: {best_conditions['Space-Time Profit [$/L/day]']:.2f} $/L/day")
print(f"Approximate 95% PI: [{st_profit_lower:.2f}, {st_profit_upper:.2f}] $/L/day")

print("\nSpace-Time Yield")
print(f"Predicted: {best_conditions['Space-Time Yield [mg/L/day]']:.5f} mg/L/day")
print(f"Approximate 95% PI: [{st_yield_lower:.5f}, {st_yield_upper:.5f}] mg/L/day")

# Fig 1 poly reg validation

fig_width = 14.39 / 2.54
fig_height = 12.09 / 2.54

# Sort results by polynomial degree
df_plot = df_results_poly.sort_values(by="param_polynomialfeatures__degree")

degrees = df_plot["param_polynomialfeatures__degree"].to_numpy()
rmse_train = -df_plot["mean_train_score"].to_numpy()
rmse_valid = -df_plot["mean_test_score"].to_numpy()

# Create figure
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(fig_width, fig_height), dpi=200,layout="constrained")

# Plot training and validation RMSE
ax.plot(degrees, rmse_train,color="blue", marker="o",linestyle="-", label="Training")

ax.plot(degrees, rmse_valid,color="red", marker="o",linestyle="--", label="Validation")

# Format axes
ax.set_xlabel("Polynomial Degree", fontsize=14)
ax.set_ylabel("RMSE [mg/L]", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.set_xticks(degrees)
ax.set_ylim(bottom=0)
ax.legend(fontsize=12)

# Save figure
fig.savefig(os.path.join("figures", "figure_01_polynomial_validation.png"))
plt.close(fig)


# Fig 2 actual vs predicted mAb conc

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(fig_width, fig_height),dpi=200,layout="constrained")

# Plot actual and predicted concentrations
ax.scatter(y_test, y_test_pred,color="green",marker="o",s=25,alpha=0.7,label="Polynomial Regression")

# Plot ideal prediction line
y_min = min(0, y_test.min(), y_test_pred.min())
y_max = max(y_test.max(), y_test_pred.max())

ax.plot([y_min, y_max],[y_min, y_max],color="black",linestyle="--",linewidth=2,label="Ideal Prediction")

# Format axes
ax.set_xlabel("Actual mAb Concentration [mg/L]", fontsize=14)
ax.set_ylabel("Predicted mAb Concentration [mg/L]", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)

ax.set_xlim(y_min, y_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal", adjustable="box")
ax.legend(fontsize=12)

# Save figure
fig.savefig(os.path.join("figures", "figure_02_polynomial_predictions.png"))
plt.close(fig)



# Fig 3 space-time profit vs. harvest time

# Select the optimal initial operating conditions
profit_curve = conditions[
    (conditions["Initial Glucose [mM]"] == best_conditions["Initial Glucose [mM]"]) &
    (conditions["Initial Glutamine [mM]"] == best_conditions["Initial Glutamine [mM]"]) &
    (conditions["Initial VCD [cells/mL]"] == best_conditions["Initial VCD [cells/mL]"])
]

# Plot predicted space-time profit
fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200, layout="constrained")

ax.plot(profit_curve["Time [h]"],profit_curve["Space-Time Profit [$/L/day]"],color="blue", linewidth=2, label="Predicted Profit")

# Mark the optimal harvest time
ax.scatter(best_conditions["Time [h]"],best_conditions["Space-Time Profit [$/L/day]"],color="red", s=60, zorder=3, label="Optimal Harvest")

ax.set_xlabel("Harvest Time [h]")
ax.set_ylabel("Space-Time Profit [$/L/day]")
ax.legend()

fig.savefig("figures/figure_03_profit_vs_time.png")
plt.close(fig)

# Fig 4: predicted mAb concentration vs. harvest time

fig, ax = plt.subplots(figsize=(6.5, 4), dpi=200, layout="constrained")

ax.plot(profit_curve["Time [h]"],profit_curve["Predicted mAb [mg/L]"],color="green", linewidth=2, label="Predicted mAb")

# Mark the optimal harvest time
ax.axvline(best_conditions["Time [h]"],color="red", linestyle="--",label="Optimal Harvest Time")

ax.set_xlabel("Harvest Time [h]")
ax.set_ylabel("Predicted mAb Concentration [mg/L]")
ax.legend()

fig.savefig("figures/figure_04_mab_vs_time.png")
plt.close(fig)
