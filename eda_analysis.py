"""
Exploratory Data Analysis Module
Generates comprehensive visualizations and statistical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class EDAAnalyzer:
    """
    Performs comprehensive exploratory data analysis
    """
    
    def __init__(self, df):
        self.df = df
        self.output_dir = config.EDA_PLOTS_DIR
        
    def generate_summary_statistics(self):
        """Generate and display summary statistics"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print("\n--- Dataset Shape ---")
        print(f"Rows: {self.df.shape[0]}")
        print(f"Columns: {self.df.shape[1]}")
        
        print("\n--- Data Types ---")
        print(self.df.dtypes.value_counts())
        
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
        
        print("\n--- Numerical Features Statistics ---")
        print(self.df.describe())
        
        print("\n--- Target Variable (Rating) Distribution ---")
        print(f"Mean Rating: {self.df['rating'].mean():.2f}")
        print(f"Median Rating: {self.df['rating'].median():.2f}")
        print(f"Std Rating: {self.df['rating'].std():.2f}")
        print(f"Min Rating: {self.df['rating'].min():.2f}")
        print(f"Max Rating: {self.df['rating'].max():.2f}")
        
        print("\n--- Categorical Features ---")
        for col in ['brand_name', 'processor_brand', 'os']:
            if col in self.df.columns:
                print(f"\n{col.upper()} - Top 5:")
                print(self.df[col].value_counts().head())
    
    def plot_target_distribution(self):
        """Plot rating distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.df['rating'].dropna(), bins=30, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Rating')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Rating Distribution')
        axes[0].axvline(self.df['rating'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["rating"].mean():.2f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(self.df['rating'].dropna(), vert=True)
        axes[1].set_ylabel('Rating')
        axes[1].set_title('Rating Box Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '/images/photo1765958862.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: /images/photo1765958862.jpg")
    
    def plot_price_analysis(self):
        """Analyze price distribution and relationship with rating"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Price distribution
        axes[0, 0].hist(self.df['price'], bins=50, color='lightcoral', edgecolor='black')
        axes[0, 0].set_xlabel('Price (₹)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Price Distribution')
        
        # Price vs Rating scatter
        axes[0, 1].scatter(self.df['price'], self.df['rating'], alpha=0.5, color='steelblue')
        axes[0, 1].set_xlabel('Price (₹)')
        axes[0, 1].set_ylabel('Rating')
        axes[0, 1].set_title('Price vs Rating')
        
        # Price by brand (top 10 brands)
        top_brands = self.df['brand_name'].value_counts().head(10).index
        df_top_brands = self.df[self.df['brand_name'].isin(top_brands)]
        brand_price = df_top_brands.groupby('brand_name')['price'].mean().sort_values(ascending=False)
        axes[1, 0].barh(brand_price.index, brand_price.values, color='teal')
        axes[1, 0].set_xlabel('Average Price (₹)')
        axes[1, 0].set_title('Average Price by Brand (Top 10)')
        
        # Price categories
        price_bins = [0, 15000, 30000, 50000, np.inf]
        price_labels = ['Budget', 'Mid-Range', 'Premium', 'Flagship']
        self.df['price_cat'] = pd.cut(self.df['price'], bins=price_bins, labels=price_labels)
        price_cat_counts = self.df['price_cat'].value_counts()
        axes[1, 1].pie(price_cat_counts.values, labels=price_cat_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Price Category Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '/images/photo1765958861.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: /images/photo1765958861.jpg")
    
    def plot_brand_analysis(self):
        """Analyze brand distribution and ratings"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Top brands by count
        top_brands = self.df['brand_name'].value_counts().head(10)
        axes[0, 0].barh(top_brands.index, top_brands.values, color='mediumpurple')
        axes[0, 0].set_xlabel('Number of Models')
        axes[0, 0].set_title('Top 10 Brands by Model Count')
        
        # Average rating by brand
        brand_ratings = self.df.groupby('brand_name')['rating'].mean().sort_values(ascending=False).head(10)
        axes[0, 1].barh(brand_ratings.index, brand_ratings.values, color='lightgreen')
        axes[0, 1].set_xlabel('Average Rating')
        axes[0, 1].set_title('Top 10 Brands by Average Rating')
        
        # Brand market share (by price)
        brand_revenue = self.df.groupby('brand_name')['price'].sum().sort_values(ascending=False).head(8)
        axes[1, 0].pie(brand_revenue.values, labels=brand_revenue.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Market Share by Total Price (Top 8 Brands)')
        
        # OS distribution
        os_counts = self.df['os'].value_counts()
        axes[1, 1].bar(os_counts.index, os_counts.values, color=['#3DDC84', '#A2AAAD', '#666666', '#999999'])
        axes[1, 1].set_xlabel('Operating System')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Operating System Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '/images/photo1765958861.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: /images/photo1765958861.jpg")
    
    def plot_hardware_analysis(self):
        """Analyze hardware specifications"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # RAM distribution
        ram_counts = self.df['ram_capacity'].value_counts().sort_index()
        axes[0, 0].bar(ram_counts.index, ram_counts.values, color='coral')
        axes[0, 0].set_xlabel('RAM (GB)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('RAM Distribution')
        
        # Storage distribution
        storage_counts = self.df['internal_memory'].value_counts().sort_index()
        axes[0, 1].bar(storage_counts.index, storage_counts.values, color='skyblue')
        axes[0, 1].set_xlabel('Storage (GB)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Internal Storage Distribution')
        
        # Battery capacity
        axes[0, 2].hist(self.df['battery_capacity'], bins=30, color='lightgreen', edgecolor='black')
        axes[0, 2].set_xlabel('Battery Capacity (mAh)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Battery Capacity Distribution')
        
        # Processor brand
        processor_counts = self.df['processor_brand'].value_counts().head(8)
        axes[1, 0].barh(processor_counts.index, processor_counts.values, color='gold')
        axes[1, 0].set_xlabel('Count')
        axes[1, 0].set_title('Processor Brand Distribution')
        
        # Screen size
        axes[1, 1].hist(self.df['screen_size'], bins=30, color='plum', edgecolor='black')
        axes[1, 1].set_xlabel('Screen Size (inches)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Screen Size Distribution')
        
        # Refresh rate
        refresh_counts = self.df['refresh_rate'].value_counts().sort_index()
        axes[1, 2].bar(refresh_counts.index, refresh_counts.values, color='lightcoral')
        axes[1, 2].set_xlabel('Refresh Rate (Hz)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Refresh Rate Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '/images/photo1765958861.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: /images/photo1765958861.jpg")
    
    def plot_camera_analysis(self):
        """Analyze camera specifications"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Rear camera distribution
        axes[0, 0].hist(self.df['primary_camera_rear'], bins=30, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('Primary Rear Camera (MP)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Rear Camera Distribution')
        
        # Front camera distribution
        axes[0, 1].hist(self.df['primary_camera_front'], bins=30, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('Primary Front Camera (MP)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Front Camera Distribution')
        
        # Camera vs Rating
        axes[1, 0].scatter(self.df['primary_camera_rear'], self.df['rating'], alpha=0.5, color='green')
        axes[1, 0].set_xlabel('Primary Rear Camera (MP)')
        axes[1, 0].set_ylabel('Rating')
        axes[1, 0].set_title('Rear Camera vs Rating')
        
        # Number of cameras
        camera_counts = self.df['num_rear_cameras'].value_counts().sort_index()
        axes[1, 1].bar(camera_counts.index, camera_counts.values, color='purple')
        axes[1, 1].set_xlabel('Number of Rear Cameras')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Number of Rear Cameras Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '/images/photo1765958861.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: /images/photo1765958861.jpg")
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap for numerical features"""
        # Select numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        # Plot heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '/images/photo1765958861.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: /images/photo1765958861.jpg")
        
        # Print top correlations with rating
        if 'rating' in correlation_matrix.columns:
            rating_corr = correlation_matrix['rating'].sort_values(ascending=False)
            print("\n--- Top Features Correlated with Rating ---")
            print(rating_corr.head(10))
    
    def plot_feature_vs_rating(self):
        """Plot key features vs rating"""
        features = ['price', 'battery_capacity', 'ram_capacity', 'screen_size']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            axes[idx].scatter(self.df[feature], self.df['rating'], alpha=0.5, color='steelblue')
            axes[idx].set_xlabel(feature.replace('_', ' ').title())
            axes[idx].set_ylabel('Rating')
            axes[idx].set_title(f'{feature.replace("_", " ").title()} vs Rating')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '/images/photo1765958861.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: /images/photo1765958861.jpg")
    
    def plot_connectivity_features(self):
        """Analyze connectivity features"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 5G availability
        has_5g_counts = self.df['has_5g'].value_counts()
        axes[0].pie(has_5g_counts.values, labels=['Has 5G', 'No 5G'], autopct='%1.1f%%', 
                    colors=['#4CAF50', '#FFC107'], startangle=90)
        axes[0].set_title('5G Availability')
        
        # NFC availability
        has_nfc_counts = self.df['has_nfc'].value_counts()
        axes[1].pie(has_nfc_counts.values, labels=['Has NFC', 'No NFC'], autopct='%1.1f%%',
                    colors=['#2196F3', '#FF9800'], startangle=90)
        axes[1].set_title('NFC Availability')
        
        # Fast charging
        fast_charging_counts = self.df['fast_charging_available'].value_counts()
        axes[2].pie(fast_charging_counts.values, labels=['Fast Charging', 'No Fast Charging'], 
                    autopct='%1.1f%%', colors=['#9C27B0', '#E91E63'], startangle=90)
        axes[2].set_title('Fast Charging Availability')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '/images/photo1765958861.jpg', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: /images/photo1765958861.jpg")
    
    def generate_all_plots(self):
        """Generate all EDA visualizations"""
        print("\n" + "="*60)
        print("GENERATING EDA VISUALIZATIONS")
        print("="*60)
        
        self.plot_target_distribution()
        self.plot_price_analysis()
        self.plot_brand_analysis()
        self.plot_hardware_analysis()
        self.plot_camera_analysis()
        self.plot_correlation_heatmap()
        self.plot_feature_vs_rating()
        self.plot_connectivity_features()
        
        print("\n" + "="*60)
        print(f"All plots saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main EDA pipeline"""
    print("="*60)
    print("MOBILE RECOMMENDATION SYSTEM - EDA")
    print("="*60)
    
    # Load processed data
    df = pd.read_csv(config.DATA_DIR / 'processed_data.csv')
    print(f"\nLoaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize analyzer
    analyzer = EDAAnalyzer(df)
    
    # Generate summary statistics
    analyzer.generate_summary_statistics()
    
    # Generate all visualizations
    analyzer.generate_all_plots()
    
    print("\n" + "="*60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()