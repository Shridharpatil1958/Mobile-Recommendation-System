"""
Model Training Module
Trains multiple ML models for rating prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import time
import config
from data_preprocessing import DataPreprocessor

class ModelTrainer:
    """
    Trains and evaluates multiple ML models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """Initialize all ML models"""
        print("\n=== Initializing Models ===")
        
        self.models = {
            'Linear Regression': LinearRegression(**config.MODEL_CONFIGS['Linear Regression']['params']),
            'KNN': KNeighborsRegressor(**config.MODEL_CONFIGS['KNN']['params']),
            'Random Forest': RandomForestRegressor(**config.MODEL_CONFIGS['Random Forest']['params']),
            'Gradient Boosting': GradientBoostingRegressor(**config.MODEL_CONFIGS['Gradient Boosting']['params'])
        }
        
        print(f"Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  ✓ {name}")
    
    def train_model(self, model_name, model, X_train, y_train):
        """Train a single model"""
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")
        
        return model, training_time
    
    def evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test):
        """Evaluate model performance"""
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Store results
        results = {
            'model_name': model_name,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        # Print results
        print(f"\n  {model_name} - Performance Metrics:")
        print(f"    Train MAE:  {train_mae:.4f} | Test MAE:  {test_mae:.4f}")
        print(f"    Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        print(f"    Train R²:   {train_r2:.4f} | Test R²:   {test_r2:.4f}")
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.initialize_models()
        
        all_results = []
        
        for model_name, model in self.models.items():
            # Train
            trained_model, training_time = self.train_model(model_name, model, X_train, y_train)
            
            # Evaluate
            results = self.evaluate_model(model_name, trained_model, X_train, y_train, X_test, y_test)
            results['training_time'] = training_time
            
            all_results.append(results)
            self.results[model_name] = results
            
            # Update model
            self.models[model_name] = trained_model
        
        # Create results dataframe
        results_df = pd.DataFrame(all_results)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def select_best_model(self, results_df):
        """Select the best model based on test R² score"""
        print("\n" + "="*60)
        print("SELECTING BEST MODEL")
        print("="*60)
        
        # Sort by test R² (descending)
        results_df_sorted = results_df.sort_values('test_r2', ascending=False)
        
        best_model_name = results_df_sorted.iloc[0]['model_name']
        best_test_r2 = results_df_sorted.iloc[0]['test_r2']
        best_test_mae = results_df_sorted.iloc[0]['test_mae']
        best_test_rmse = results_df_sorted.iloc[0]['test_rmse']
        
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test R²:   {best_test_r2:.4f}")
        print(f"   Test MAE:  {best_test_mae:.4f}")
        print(f"   Test RMSE: {best_test_rmse:.4f}")
        
        print("\n📊 Model Rankings (by Test R²):")
        for idx, row in results_df_sorted.iterrows():
            print(f"   {row['model_name']:20s} - R²: {row['test_r2']:.4f}")
        
        return self.best_model, self.best_model_name
    
    def save_models(self):
        """Save all trained models and results"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        # Save best model
        joblib.dump(self.best_model, config.BEST_MODEL_PATH)
        print(f"✓ Best model saved: {config.BEST_MODEL_PATH}")
        
        # Save all models
        for model_name, model in self.models.items():
            model_path = config.MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_path)
            print(f"✓ {model_name} saved: {model_path}")
        
        # Save results
        joblib.dump(self.results, config.MODEL_METRICS_PATH)
        print(f"✓ Model metrics saved: {config.MODEL_METRICS_PATH}")
        
        # Save best model name
        with open(config.MODELS_DIR / 'best_model_name.txt', 'w') as f:
            f.write(self.best_model_name)
        print(f"✓ Best model name saved")
    
    def load_best_model(self):
        """Load the best trained model"""
        self.best_model = joblib.load(config.BEST_MODEL_PATH)
        with open(config.MODELS_DIR / 'best_model_name.txt', 'r') as f:
            self.best_model_name = f.read().strip()
        print(f"✓ Loaded best model: {self.best_model_name}")
        return self.best_model


def main():
    """Main training pipeline"""
    print("="*60)
    print("MOBILE RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("="*60)
    
    # Load processed data
    print("\nLoading processed data...")
    df = pd.read_csv(config.DATA_DIR / 'processed_data.csv')
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor()
    
    X_train, X_test, y_train, y_test, feature_names = preprocessor.split_data(df)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train all models
    results_df = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Select best model
    best_model, best_model_name = trainer.select_best_model(results_df)
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return trainer, results_df


if __name__ == "__main__":
    main()