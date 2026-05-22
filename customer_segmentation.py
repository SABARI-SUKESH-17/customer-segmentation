"""
Customer Segmentation using Machine Learning
============================================

This module implements customer segmentation using K-Means clustering
and other machine learning techniques.

Author: Your Name
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class CustomerSegmentation:
    """
    A class to perform customer segmentation using K-Means clustering.
    
    Attributes:
        data (pd.DataFrame): Customer data
        scaler (StandardScaler): Scaler for feature normalization
        model (KMeans): Trained K-Means model
        clusters (np.array): Cluster assignments for each customer
    """
    
    def __init__(self, data_path):
        """
        Initialize the segmentation class and load data.
        
        Args:
            data_path (str): Path to the customer data CSV file
        """
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.model = None
        self.clusters = None
        print(f"Data loaded successfully! Shape: {self.data.shape}")
    
    def explore_data(self):
        """Display basic data exploration and statistics."""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print("\nFirst few rows:")
        print(self.data.head())
        
        print("\nData Info:")
        print(self.data.info())
        
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        print("\nMissing Values:")
        print(self.data.isnull().sum())
    
    def preprocess_data(self, feature_columns):
        """
        Preprocess data for clustering.
        
        Args:
            feature_columns (list): List of columns to use for clustering
        
        Returns:
            np.array: Normalized feature matrix
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Select features
        X = self.data[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        print(f"\nMissing values handled. Shape: {X.shape}")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        print("Features normalized using StandardScaler")
        
        return X_scaled
    
    def find_optimal_clusters(self, X_scaled, k_range=range(2, 11)):
        """
        Find optimal number of clusters using Elbow method.
        
        Args:
            X_scaled (np.array): Normalized feature matrix
            k_range (range): Range of cluster numbers to test
        
        Returns:
            int: Optimal number of clusters
        """
        print("\n" + "="*50)
        print("FINDING OPTIMAL CLUSTERS")
        print("="*50)
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Plot Elbow curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (WCSS)', fontsize=12)
        axes[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/elbow_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\nOptimal number of clusters: {optimal_k}")
        print(f"Silhouette Score: {max(silhouette_scores):.4f}")
        
        return optimal_k
    
    def train_model(self, X_scaled, n_clusters=3):
        """
        Train K-Means clustering model.
        
        Args:
            X_scaled (np.array): Normalized feature matrix
            n_clusters (int): Number of clusters
        """
        print("\n" + "="*50)
        print("TRAINING K-MEANS MODEL")
        print("="*50)
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = self.model.fit_predict(X_scaled)
        
        # Add cluster labels to original data
        self.data['Cluster'] = self.clusters
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, self.clusters)
        davies_bouldin = davies_bouldin_score(X_scaled, self.clusters)
        
        print(f"\nNumber of clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"Inertia: {self.model.inertia_:.2f}")
    
    def analyze_segments(self, feature_columns):
        """
        Analyze and display segment characteristics.
        
        Args:
            feature_columns (list): Feature columns to analyze
        """
        print("\n" + "="*50)
        print("SEGMENT ANALYSIS")
        print("="*50)
        
        segment_profiles = self.data.groupby('Cluster')[feature_columns].mean()
        
        print("\nSegment Profiles (Mean Values):")
        print(segment_profiles.round(2))
        
        # Segment sizes
        print("\nSegment Sizes:")
        print(self.data['Cluster'].value_counts().sort_index())
        
        # Save segment profiles
        segment_profiles.to_csv('output/segment_profiles.csv')
        print("\nSegment profiles saved to output/segment_profiles.csv")
        
        return segment_profiles
    
    def visualize_segments(self, feature_columns):
        """
        Create visualizations of customer segments.
        
        Args:
            feature_columns (list): Features to visualize
        """
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Cluster distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Cluster distribution
        cluster_counts = self.data['Cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='steelblue')
        axes[0, 0].set_xlabel('Cluster', fontsize=11)
        axes[0, 0].set_ylabel('Number of Customers', fontsize=11)
        axes[0, 0].set_title('Cluster Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Feature comparison - Feature 1 vs Feature 2
        if len(feature_columns) >= 2:
            scatter = axes[0, 1].scatter(
                self.data[feature_columns[0]], 
                self.data[feature_columns[1]], 
                c=self.data['Cluster'], 
                cmap='viridis', 
                s=50, 
                alpha=0.6
            )
            axes[0, 1].set_xlabel(feature_columns[0], fontsize=11)
            axes[0, 1].set_ylabel(feature_columns[1], fontsize=11)
            axes[0, 1].set_title(f'{feature_columns[0]} vs {feature_columns[1]}', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')
        
        # 3. Box plot for first numeric feature
        self.data.boxplot(column=feature_columns[0], by='Cluster', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Cluster', fontsize=11)
        axes[1, 0].set_ylabel(feature_columns[0], fontsize=11)
        axes[1, 0].set_title(f'{feature_columns[0]} by Cluster', fontsize=12, fontweight='bold')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=0)
        
        # 4. Heatmap of segment profiles
        segment_profiles = self.data.groupby('Cluster')[feature_columns].mean()
        segment_profiles_normalized = (segment_profiles - segment_profiles.min()) / (segment_profiles.max() - segment_profiles.min())
        
        sns.heatmap(segment_profiles_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 1], cbar_kws={'label': 'Normalized Value'})
        axes[1, 1].set_title('Segment Profile Heatmap (Normalized)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Cluster', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('output/segment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to output/segment_analysis.png")
    
    def export_results(self, output_path='output/customer_segments.csv'):
        """
        Export segmented data to CSV.
        
        Args:
            output_path (str): Path to save the output file
        """
        self.data.to_csv(output_path, index=False)
        print(f"\nSegmented data exported to {output_path}")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*50)
    print("CUSTOMER SEGMENTATION PROJECT")
    print("="*50)
    
    # Initialize segmentation
    segmentation = CustomerSegmentation('data/raw/customer_data.csv')
    
    # Explore data
    segmentation.explore_data()
    
    # Define features for clustering
    feature_columns = [
        'Age',
        'Annual_Income',
        'Purchase_Frequency',
        'Average_Order_Value',
        'Customer_Tenure'
    ]
    
    # Preprocess data
    X_scaled = segmentation.preprocess_data(feature_columns)
    
    # Find optimal clusters
    optimal_k = segmentation.find_optimal_clusters(X_scaled)
    
    # Train model
    segmentation.train_model(X_scaled, n_clusters=optimal_k)
    
    # Analyze segments
    segmentation.analyze_segments(feature_columns)
    
    # Visualize segments
    segmentation.visualize_segments(feature_columns)
    
    # Export results
    segmentation.export_results()
    
    print("\n" + "="*50)
    print("SEGMENTATION COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()
