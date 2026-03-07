"""
Data Preprocessing Module
Handles data cleaning, missing value imputation, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import config

class DataPreprocessor:
    """
    Handles all data preprocessing tasks including:
    - Loading data
    - Cleaning (missing values, duplicates)
    - Feature engineering
    - Encoding categorical variables
    - Scaling numerical features
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath=None):
        """Load smartphone dataset"""
        if filepath is None:
            filepath = config.DATASET_PATH
        
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def clean_data(self, df):
        """
        Clean the dataset:
        - Handle missing values
        - Remove duplicates
        - Fix data types
        """
        print("\n=== Data Cleaning ===")
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Convert boolean columns
        for col in config.BOOLEAN_FEATURES:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({
                    'TRUE': 1, 'True': 1, True: 1, 1: 1,
                    'FALSE': 0, 'False': 0, False: 0, 0: 0
                }).fillna(0).astype(int)
        
        # Handle missing values in numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                if col == 'fast_charging':
                    # For fast charging, fill with 0 if not available
                    df_clean[col] = df_clean[col].fillna(0)
                elif col == 'rating':
                    # For rating (target), fill with median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    # For other numeric columns, fill with median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                print(f"Filled {missing_count} missing values in '{col}'")
        
        # Handle missing values in categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                df_clean[col] = df_clean[col].fillna('unknown')
                print(f"Filled {missing_count} missing values in '{col}'")
        
        # Clean text columns
        if 'brand_name' in df_clean.columns:
            df_clean['brand_name'] = df_clean['brand_name'].str.lower().str.strip()
        if 'processor_brand' in df_clean.columns:
            df_clean['processor_brand'] = df_clean['processor_brand'].str.lower().str.strip()
        if 'os' in df_clean.columns:
            df_clean['os'] = df_clean['os'].str.lower().str.strip()
        
        print(f"\nCleaned data shape: {df_clean.shape}")
        print(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def engineer_features(self, df):
        """
        Create new features to improve model performance
        """
        print("\n=== Feature Engineering ===")
        df_eng = df.copy()
        
        # Price category
        df_eng['price_category'] = pd.cut(
            df_eng['price'],
            bins=[0, 15000, 30000, 50000, np.inf],
            labels=['budget', 'mid-range', 'premium', 'flagship']
        )
        
        # RAM category
        df_eng['ram_category'] = pd.cut(
            df_eng['ram_capacity'],
            bins=[0, 4, 8, 12, np.inf],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Battery category
        df_eng['battery_category'] = pd.cut(
            df_eng['battery_capacity'],
            bins=[0, 4000, 5000, 6000, np.inf],
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        # Camera quality score (combined front + rear)
        df_eng['camera_score'] = (
            df_eng['primary_camera_rear'] * 0.7 + 
            df_eng['primary_camera_front'] * 0.3
        )
        
        # Performance score
        df_eng['performance_score'] = (
            df_eng['ram_capacity'] * 0.4 +
            df_eng['processor_speed'] * 10 * 0.3 +
            df_eng['num_cores'] * 2 * 0.3
        )
        
        # Display quality score
        df_eng['display_score'] = (
            df_eng['screen_size'] * 10 * 0.4 +
            df_eng['refresh_rate'] * 0.3 +
            (df_eng['resolution_width'] * df_eng['resolution_height'] / 1000000) * 0.3
        )
        
        # Value for money (rating per 10000 rupees)
        df_eng['value_score'] = df_eng['rating'] / (df_eng['price'] / 10000)
        
        # Total storage
        df_eng['total_storage'] = df_eng['internal_memory'] + df_eng.get('extended_upto', 0).fillna(0)
        
        print(f"Created {len(['price_category', 'ram_category', 'battery_category', 'camera_score', 'performance_score', 'display_score', 'value_score', 'total_storage'])} new features")
        
        return df_eng
    
    def prepare_features(self, df, fit=True):
        """
        Prepare features for ML models:
        - Encode categorical variables
        - Scale numerical features
        """
        print("\n=== Feature Preparation ===")
        df_prep = df.copy()
        
        # Select features for modeling
        feature_cols = []
        
        # Numerical features
        for col in config.NUMERIC_FEATURES:
            if col in df_prep.columns:
                feature_cols.append(col)
        
        # Engineered numerical features
        engineered_numeric = [
            'camera_score', 'performance_score', 'display_score', 
            'value_score', 'total_storage'
        ]
        for col in engineered_numeric:
            if col in df_prep.columns:
                feature_cols.append(col)
        
        # Boolean features
        for col in config.BOOLEAN_FEATURES:
            if col in df_prep.columns:
                feature_cols.append(col)
        
        # Encode categorical features
        for col in config.CATEGORICAL_FEATURES:
            if col in df_prep.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_prep[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_prep[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df_prep[f'{col}_encoded'] = df_prep[col].apply(
                            lambda x: self.label_encoders[col].transform([str(x)])[0] 
                            if str(x) in self.label_encoders[col].classes_ 
                            else -1
                        )
                feature_cols.append(f'{col}_encoded')
        
        # Encode engineered categorical features
        engineered_categorical = ['price_category', 'ram_category', 'battery_category']
        for col in engineered_categorical:
            if col in df_prep.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_prep[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_prep[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df_prep[f'{col}_encoded'] = df_prep[col].apply(
                            lambda x: self.label_encoders[col].transform([str(x)])[0] 
                            if str(x) in self.label_encoders[col].classes_ 
                            else -1
                        )
                feature_cols.append(f'{col}_encoded')
        
        # Create feature matrix
        X = df_prep[feature_cols].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = feature_cols
        else:
            X_scaled = self.scaler.transform(X)
        
        print(f"Prepared {len(feature_cols)} features for modeling")
        
        return X_scaled, feature_cols
    
    def split_data(self, df):
        """Split data into train and test sets"""
        # Prepare features
        X, feature_names = self.prepare_features(df, fit=True)
        y = df[config.TARGET_COLUMN].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE
        )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def save_preprocessor(self):
        """Save scaler and encoders"""
        joblib.dump(self.scaler, config.SCALER_PATH)
        joblib.dump(self.label_encoders, config.MODELS_DIR / 'label_encoders.pkl')
        joblib.dump(self.feature_names, config.FEATURE_NAMES_PATH)
        print(f"\nPreprocessor saved to {config.MODELS_DIR}")
    
    def load_preprocessor(self):
        """Load saved scaler and encoders"""
        self.scaler = joblib.load(config.SCALER_PATH)
        self.label_encoders = joblib.load(config.MODELS_DIR / 'label_encoders.pkl')
        self.feature_names = joblib.load(config.FEATURE_NAMES_PATH)
        print("Preprocessor loaded successfully")


def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("MOBILE RECOMMENDATION SYSTEM - DATA PREPROCESSING")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data()
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Engineer features
    df_final = preprocessor.engineer_features(df_clean)
    
    # Split data and prepare features
    X_train, X_test, y_train, y_test, feature_names = preprocessor.split_data(df_final)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    # Save processed data
    df_final.to_csv(config.DATA_DIR / 'processed_data.csv', index=False)
    print(f"\nProcessed data saved to {config.DATA_DIR / 'processed_data.csv'}")
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return df_final, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()