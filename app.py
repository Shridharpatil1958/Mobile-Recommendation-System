# =============================
# Mobile Recommendation System
# End-to-End Data Science Project
# Author: Yash Desai (customizable)
# =============================

# --------- 1. IMPORT LIBRARIES ---------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib
import warnings
warnings.filterwarnings('ignore')


# --------- 2. LOAD DATA ---------
# Update path if needed
DATA_PATH = "Smartphones_cleaned_dataset.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())


# --------- 3. DATA CLEANING ---------
# Check missing values
print("\nMissing values (%):")
print((df.isna().sum() / len(df)) * 100)

# Fill missing numerical values with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
cat_cols = df.select_dtypes(include=['object', 'bool']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove duplicates if any
df.drop_duplicates(inplace=True)

print("\nAfter cleaning shape:", df.shape)


# --------- 4. EXPLORATORY DATA ANALYSIS (EDA) ---------
plt.figure()
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.show()

plt.figure()
sns.boxplot(x='brand_name', y='price', data=df)
plt.xticks(rotation=90)
plt.title('Price by Brand')
plt.show()

plt.figure()
sns.scatterplot(x='battery_capacity', y='rating', data=df)
plt.title('Battery vs Rating')
plt.show()


# --------- 5. FEATURE SELECTION ---------
# Target variable (rating prediction)
TARGET = 'rating'

FEATURES = df.drop(columns=['model', TARGET])
y = df[TARGET]

categorical_features = FEATURES.select_dtypes(include=['object', 'bool']).columns
numerical_features = FEATURES.select_dtypes(include=['int64', 'float64']).columns


# --------- 6. PREPROCESSING PIPELINE ---------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)


# --------- 7. TRAIN-TEST SPLIT ---------
X_train, X_test, y_train, y_test = train_test_split(
    FEATURES, y, test_size=0.2, random_state=42
)


# --------- 8. MACHINE LEARNING MODELS ---------
models = {
    "Linear Regression": LinearRegression(),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    print(f"\n{name} Performance")
    print(results[name])


# --------- 9. SAVE BEST MODEL ---------
# Select best model based on R2
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_model = models[best_model_name]

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

final_pipeline.fit(FEATURES, y)

joblib.dump(final_pipeline, 'rating_prediction_model.pkl')
print(f"\nSaved best model: {best_model_name}")


# --------- 10. CONTENT-BASED RECOMMENDATION SYSTEM ---------
# Using Nearest Neighbors (Similarity based)

reco_features = [
    'price', 'battery_capacity', 'ram_capacity', 'internal_memory',
    'screen_size', 'refresh_rate', 'primary_camera_rear'
]

reco_df = df[['brand_name', 'model'] + reco_features]

scaler = StandardScaler()
X_reco = scaler.fit_transform(reco_df[reco_features])

nn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
nn_model.fit(X_reco)

# Save recommendation models
joblib.dump(nn_model, 'mobile_recommendation_knn.pkl')
joblib.dump(scaler, 'reco_scaler.pkl')


# --------- 11. RECOMMENDATION FUNCTION ---------
def recommend_mobiles(user_input: dict, top_n=5):
    """
    user_input example:
    {
        'price': 20000,
        'battery_capacity': 5000,
        'ram_capacity': 8,
        'internal_memory': 128,
        'screen_size': 6.5,
        'refresh_rate': 120,
        'primary_camera_rear': 64
    }
    """
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)

    distances, indices = nn_model.kneighbors(user_scaled, n_neighbors=top_n)

    recommendations = reco_df.iloc[indices[0]][['brand_name', 'model']]
    return recommendations


# --------- 12. SAMPLE RECOMMENDATION ---------
user_preferences = {
    'price': 25000,
    'battery_capacity': 5000,
    'ram_capacity': 8,
    'internal_memory': 128,
    'screen_size': 6.6,
    'refresh_rate': 120,
    'primary_camera_rear': 64
}

print("\nRecommended Mobiles:")
print(recommend_mobiles(user_preferences))


# --------- 13. STREAMLIT WEB APPLICATION (UPDATED & ERROR-FREE) ---------
# Save this file as app.py

import streamlit as st
import joblib
import pandas as pd

# ---------------- LOAD FILES ----------------
df_train = pd.read_csv('Smartphones_cleaned_dataset.csv')

rating_model = joblib.load('rating_prediction_model.pkl')
reco_model = joblib.load('mobile_recommendation_knn.pkl')
reco_scaler = joblib.load('reco_scaler.pkl')

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Mobile Recommendation System", layout="centered")

st.title("📱 Mobile Recommendation System")
st.write("Predict mobile rating and get recommendations based on your preferences")

# ---------------- USER INPUT ----------------
st.sidebar.header("Enter Your Preferences")

price = st.sidebar.number_input("Budget (₹)", 5000, 150000, 25000)
battery = st.sidebar.number_input("Battery Capacity (mAh)", 2000, 7000, 5000)
ram = st.sidebar.number_input("RAM (GB)", 2, 24, 8)
storage = st.sidebar.number_input("Storage (GB)", 16, 1024, 128)
screen = st.sidebar.number_input("Screen Size (inches)", 4.0, 7.5, 6.6)
refresh = st.sidebar.number_input("Refresh Rate (Hz)", 60, 240, 120)
camera_rear = st.sidebar.number_input("Primary Rear Camera (MP)", 8, 200, 64)

user_input = {
    'price': price,
    'battery_capacity': battery,
    'ram_capacity': ram,
    'internal_memory': storage,
    'screen_size': screen,
    'refresh_rate': refresh,
    'primary_camera_rear': camera_rear
}

# ---------------- BUILD FULL FEATURE ROW ----------------
feature_cols = df_train.drop(columns=['rating', 'model']).columns
user_df = pd.DataFrame(columns=feature_cols)

# Fill provided inputs
for k, v in user_input.items():
    if k in user_df.columns:
        user_df.loc[0, k] = v

# Fill missing columns safely
for col in user_df.columns:
    if user_df[col].isna().any():
        if df_train[col].dtype in ['int64', 'float64']:
            user_df[col].fillna(df_train[col].median(), inplace=True)
        else:
            user_df[col].fillna(df_train[col].mode()[0], inplace=True)

# ---------------- BUTTON ACTION ----------------
if st.button("🔍 Predict & Recommend"):

    # ---- Rating Prediction ----
    predicted_rating = rating_model.predict(user_df)[0]

    st.subheader("⭐ Predicted User Rating")
    st.success(f"{predicted_rating:.2f} / 5")

    # ---- Recommendation System ----
    reco_features = [
        'price', 'battery_capacity', 'ram_capacity', 'internal_memory',
        'screen_size', 'refresh_rate', 'primary_camera_rear'
    ]

    reco_df = df_train[['brand_name', 'model'] + reco_features]

    user_reco_df = user_df[reco_features]
    user_scaled = reco_scaler.transform(user_reco_df)

    distances, indices = reco_model.kneighbors(user_scaled, n_neighbors=5)

    recommendations = reco_df.iloc[indices[0]].reset_index(drop=True)

    st.subheader("📌 Recommended Mobiles")
    st.dataframe(recommendations)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built by Yash Desai | Data Science Project")