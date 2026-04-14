import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================================================
# 1. DATA ACQUISITION
# =========================================================
# Load dataset from CSV file
df = pd.read_csv("AI Job Market Dataset.csv")

print("\n--- Dataset Loaded Successfully ---")
print(df.head())


# =========================================================
# 2. DATA CLEANING
# =========================================================
print("\n--- Data Cleaning ---")

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values
# Numeric columns -> fill with median
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns -> fill with mode
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

print("\nFeature Engineering Preview:")
print(df[["total_skills", "is_remote"]].head())

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
    "total_skills"
]].corr())


# =========================================================
# 5. DATA VISUALIZATION
# Requirement: At least 3 techniques
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
    df[["years_experience", "salary", "job_openings", "total_skills"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()


# =========================================================
# 6. MODEL BUILDING
# Requirement: At least 2 ML algorithms
# We will predict salary (regression problem)
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


# =========================================================
# 7. MODEL EVALUATION
# Requirement: At least 2 evaluation metrics
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

# Compare models
print("\n--- Model Comparison ---")
if rf_rmse < lr_rmse:
    print("Random Forest performed better based on lower RMSE.")
else:
    print("Linear Regression performed better based on lower RMSE.")
