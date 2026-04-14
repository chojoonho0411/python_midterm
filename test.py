import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# 1. DATA ACQUISITION
# =========================================================
df = pd.read_csv("AI Job Market Dataset.csv")

print("\n--- Dataset Loaded Successfully ---")
print(df.head())


# =========================================================
# 2. DATA CLEANING
# =========================================================
print("\n--- Data Cleaning ---")

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# =========================================================
# 3. DATA PREPROCESSING
# =========================================================
print("\n--- Data Preprocessing ---")

# Drop identifier column if not useful
if "job_id" in df.columns:
    df = df.drop(columns=["job_id"])

# ---------------------------------------------------------
# 3A. Feature Engineering
# ---------------------------------------------------------
df["total_skills"] = (
    df["skills_python"]
    + df["skills_sql"]
    + df["skills_ml"]
    + df["skills_deep_learning"]
    + df["skills_cloud"]
)

df["is_remote"] = df["remote_type"].apply(lambda x: 1 if x == "Remote" else 0)

# New feature: interaction between experience and total skills
df["experience_skills_interaction"] = df["years_experience"] * df["total_skills"]

# Optional demand-based feature
df["high_demand_job"] = (df["job_openings"] > df["job_openings"].median()).astype(int)

print("\nFeature Engineering Preview:")
print(df[[
    "total_skills",
    "is_remote",
    "experience_skills_interaction",
    "high_demand_job"
]].head())

# ---------------------------------------------------------
# 3B. Encoding
# ---------------------------------------------------------
categorical_cols = [
    "job_title",
    "company_size",
    "company_industry",
    "country",
    "remote_type",
    "experience_level",
    "education_level",
    "hiring_urgency",
]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nEncoded Data Preview:")
print(df_encoded.head())

# ---------------------------------------------------------
# 3C. Scaling
# ---------------------------------------------------------
scale_cols = [
    "years_experience",
    "job_posting_month",
    "job_posting_year",
    "job_openings",
    "total_skills",
    "experience_skills_interaction"
]

scaler = StandardScaler()
df_encoded[scale_cols] = scaler.fit_transform(df_encoded[scale_cols])

print("\nScaled Columns Preview:")
print(df_encoded[scale_cols].head())


# =========================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
print("\n--- EDA ---")

print("\nDataset Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nCorrelation Matrix:")
print(df[[
    "years_experience",
    "salary",
    "job_openings",
    "total_skills",
    "experience_skills_interaction"
]].corr())

# Grouped analysis for stronger report
print("\nAverage Salary by Experience Level:")
print(df.groupby("experience_level")["salary"].mean().sort_values(ascending=False))

print("\nAverage Salary by Education Level:")
print(df.groupby("education_level")["salary"].mean().sort_values(ascending=False))

print("\nAverage Salary by Remote Type:")
print(df.groupby("remote_type")["salary"].mean().sort_values(ascending=False))


# =========================================================
# 5. DATA VISUALIZATION
# =========================================================
print("\n--- Visualizations ---")

# 1. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["salary"], kde=True)
plt.title("Distribution of Salary")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# 2. Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(y=df["years_experience"])
plt.title("Boxplot of Years of Experience")
plt.ylabel("Years of Experience")
plt.show()

# 3. Scatterplot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["years_experience"], y=df["salary"])
plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# 4. Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(
    df[["years_experience", "salary", "job_openings", "total_skills", "experience_skills_interaction"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# 5. Barplot: Average salary by experience level
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="experience_level", y="salary")
plt.title("Average Salary by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.show()

# 6. Barplot: Average salary by education level
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="education_level", y="salary")
plt.title("Average Salary by Education Level")
plt.xlabel("Education Level")
plt.ylabel("Average Salary")
plt.xticks(rotation=45)
plt.show()


# =========================================================
# 6. MODEL BUILDING
# =========================================================
print("\n--- Model Building ---")

X = df_encoded.drop(columns=["salary"])
y = df_encoded["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# Model 1: Linear Regression
# ---------------------------------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# ---------------------------------------------------------
# Model 2: Random Forest Regressor
# ---------------------------------------------------------
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# ---------------------------------------------------------
# Model 3: Decision Tree Regressor
# ---------------------------------------------------------
dt_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=8
)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)


# =========================================================
# 7. MODEL EVALUATION
# =========================================================
print("\n--- Model Evaluation ---")

# Linear Regression Metrics
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

print("\nLinear Regression Results:")
print("RMSE:", lr_rmse)
print("MAE:", lr_mae)
print("R^2:", lr_r2)

# Random Forest Metrics
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("\nRandom Forest Regressor Results:")
print("RMSE:", rf_rmse)
print("MAE:", rf_mae)
print("R^2:", rf_r2)

# Decision Tree Metrics
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_preds))
dt_mae = mean_absolute_error(y_test, dt_preds)
dt_r2 = r2_score(y_test, dt_preds)

print("\nDecision Tree Regressor Results:")
print("RMSE:", dt_rmse)
print("MAE:", dt_mae)
print("R^2:", dt_r2)

# ---------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------
print("\n--- Cross Validation (5-Fold, R^2) ---")
lr_cv = cross_val_score(lr_model, X, y, cv=5, scoring="r2")
rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring="r2")
dt_cv = cross_val_score(dt_model, X, y, cv=5, scoring="r2")

print("Linear Regression CV Mean R^2:", lr_cv.mean())
print("Random Forest CV Mean R^2:", rf_cv.mean())
print("Decision Tree CV Mean R^2:", dt_cv.mean())

# ---------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Decision Tree"],
    "RMSE": [lr_rmse, rf_rmse, dt_rmse],
    "MAE": [lr_mae, rf_mae, dt_mae],
    "R^2": [lr_r2, rf_r2, dt_r2],
    "CV Mean R^2": [lr_cv.mean(), rf_cv.mean(), dt_cv.mean()]
})

print("\n--- Model Comparison Table ---")
print(results.sort_values(by="R^2", ascending=False))


# =========================================================
# 8. FEATURE IMPORTANCE
# =========================================================
print("\n--- Feature Importance (Random Forest) ---")

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x="Importance", y="Feature")
plt.title("Top 10 Important Features in Salary Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


# =========================================================
# 9. RESIDUAL ANALYSIS
# =========================================================
print("\n--- Residual Analysis ---")

residuals = y_test - rf_preds

plt.figure(figsize=(8, 5))
sns.scatterplot(x=rf_preds, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residual Plot for Random Forest")
plt.xlabel("Predicted Salary")
plt.ylabel("Residuals")
plt.show()


# =========================================================
# 10. FINAL MODEL SUMMARY
# =========================================================
print("\n--- Final Model Summary ---")

best_model_row = results.sort_values(by="R^2", ascending=False).iloc[0]
print(f"Best model based on R^2: {best_model_row['Model']}")
print(f"R^2: {best_model_row['R^2']}")
print(f"RMSE: {best_model_row['RMSE']}")
print(f"MAE: {best_model_row['MAE']}")
print(f"CV Mean R^2: {best_model_row['CV Mean R^2']}")