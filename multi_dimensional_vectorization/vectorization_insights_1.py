
Content is user-generated and unverified.
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class MultiDimensionalInsightExtractor:
    """
    A comprehensive algorithm for extracting actionable insights from multi-dimensional vectorized data.
    
    This class implements multiple analytical techniques to discover patterns, anomalies,
    relationships, and actionable insights from high-dimensional data.
    """
    
    def __init__(self, target_column: Optional[str] = None):
        """
        Initialize the insight extractor.
        
        Args:
            target_column: Name of target variable for supervised analysis (optional)
        """
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.insights = {}
        self.processed_data = None
        self.original_data = None
        
    def fit_transform(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main method to extract insights from multi-dimensional data.
        
        Args:
            data: Input DataFrame with vectorized features
            
        Returns:
            Dictionary containing all extracted insights
        """
        self.original_data = data.copy()
        print("🔍 Starting Multi-Dimensional Insight Extraction...")
        
        # Step 1: Data Preprocessing and Validation
        self._preprocess_data(data)
        
        # Step 2: Dimensionality Analysis
        self._analyze_dimensionality()
        
        # Step 3: Feature Importance Analysis
        self._analyze_feature_importance()
        
        # Step 4: Clustering and Pattern Discovery
        self._discover_patterns()
        
        # Step 5: Anomaly Detection
        self._detect_anomalies()
        
        # Step 6: Correlation and Relationship Analysis
        self._analyze_relationships()
        
        # Step 7: Statistical Insights
        self._extract_statistical_insights()
        
        # Step 8: Generate Actionable Recommendations
        self._generate_recommendations()
        
        print("✅ Insight extraction completed!")
        return self.insights
    
    def _preprocess_data(self, data: pd.DataFrame):
        """Preprocess and validate the input data."""
        print("📊 Preprocessing data...")
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_clean = data[numeric_columns].fillna(data[numeric_columns].median())
        
        # Scale the data
        if self.target_column and self.target_column in data_clean.columns:
            features = data_clean.drop(columns=[self.target_column])
            target = data_clean[self.target_column]
            self.processed_features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=features.columns,
                index=features.index
            )
            self.target = target
        else:
            self.processed_features = pd.DataFrame(
                self.scaler.fit_transform(data_clean),
                columns=data_clean.columns,
                index=data_clean.index
            )
            self.target = None
        
        self.processed_data = self.processed_features.copy()
        if self.target is not None:
            self.processed_data[self.target_column] = self.target
            
        self.insights['data_info'] = {
            'n_samples': len(data_clean),
            'n_features': len(self.processed_features.columns),
            'feature_names': list(self.processed_features.columns),
            'missing_values_handled': data.isnull().sum().sum(),
            'data_shape': self.processed_features.shape
        }
    
    def _analyze_dimensionality(self):
        """Analyze the dimensionality and perform dimension reduction."""
        print("🔄 Analyzing dimensionality...")
        
        n_features = self.processed_features.shape[1]
        n_samples = self.processed_features.shape[0]
        
        # PCA Analysis
        pca = PCA()
        pca_data = pca.fit_transform(self.processed_features)
        
        # Find optimal number of components (95% variance)
        cumvar_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumvar_ratio >= 0.95) + 1
        
        # Reduced dimensionality data
        pca_reduced = PCA(n_components=min(n_components_95, n_features))
        self.pca_data = pca_reduced.fit_transform(self.processed_features)
        
        self.insights['dimensionality'] = {
            'original_dimensions': n_features,
            'recommended_dimensions': n_components_95,
            'variance_explained_95': cumvar_ratio[n_components_95-1],
            'top_components_variance': pca.explained_variance_ratio_[:5].tolist(),
            'dimensionality_reduction_benefit': (n_features - n_components_95) / n_features,
            'principal_components': pca_reduced.components_[:3].tolist(),  # Top 3 PCs
            'feature_loadings': dict(zip(
                self.processed_features.columns,
                pca_reduced.components_[0]  # First PC loadings
            ))
        }
    
    def _analyze_feature_importance(self):
        """Analyze feature importance using various methods."""
        print("⭐ Analyzing feature importance...")
        
        feature_importance = {}
        
        # Method 1: Variance-based importance
        variances = self.processed_features.var()
        feature_importance['variance_based'] = dict(variances.sort_values(ascending=False))
        
        # Method 2: Correlation with target (if available)
        if self.target is not None:
            correlations = self.processed_features.corrwith(self.target).abs()
            feature_importance['target_correlation'] = dict(correlations.sort_values(ascending=False))
        
        # Method 3: Random Forest importance (if target available)
        if self.target is not None:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(self.processed_features, self.target)
            rf_importance = dict(zip(self.processed_features.columns, rf.feature_importances_))
            feature_importance['random_forest'] = dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True))
        
        # Method 4: Mutual information (if target available)
        if self.target is not None:
            mi_scores = mutual_info_regression(self.processed_features, self.target)
            mi_importance = dict(zip(self.processed_features.columns, mi_scores))
            feature_importance['mutual_information'] = dict(sorted(mi_importance.items(), key=lambda x: x[1], reverse=True))
        
        self.insights['feature_importance'] = feature_importance
        
        # Identify top features across methods
        all_features = set()
        for method_features in feature_importance.values():
            all_features.update(list(method_features.keys())[:5])  # Top 5 from each method
        
        self.insights['top_features'] = list(all_features)
    
    def _discover_patterns(self):
        """Discover patterns through clustering analysis."""
        print("🎯 Discovering patterns through clustering...")
        
        # K-Means clustering with optimal K
        silhouette_scores = []
        k_range = range(2, min(11, len(self.processed_features) // 2))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.processed_features)
            score = silhouette_score(self.processed_features, cluster_labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Final clustering
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans_final.fit_predict(self.processed_features)
        
        # DBSCAN for density-based clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(self.processed_features)
        
        # Analyze cluster characteristics
        cluster_profiles = {}
        for i in range(optimal_k):
            cluster_mask = cluster_labels == i
            cluster_data = self.processed_features[cluster_mask]
            
            cluster_profiles[f'cluster_{i}'] = {
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(self.processed_features) * 100),
                'centroid': cluster_data.mean().to_dict(),
                'key_features': self._identify_cluster_key_features(cluster_data)
            }
        
        self.insights['patterns'] = {
            'optimal_clusters': optimal_k,
            'silhouette_scores': silhouette_scores,
            'cluster_profiles': cluster_profiles,
            'dbscan_clusters': int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
            'dbscan_noise_points': int(list(dbscan_labels).count(-1))
        }
        
        self.cluster_labels = cluster_labels
    
    def _identify_cluster_key_features(self, cluster_data: pd.DataFrame) -> Dict[str, float]:
        """Identify key features that characterize a cluster."""
        # Compare cluster mean to global mean
        global_mean = self.processed_features.mean()
        cluster_mean = cluster_data.mean()
        
        feature_importance = abs(cluster_mean - global_mean).sort_values(ascending=False)
        return dict(feature_importance.head(3))
    
    def _detect_anomalies(self):
        """Detect anomalies and outliers in the data."""
        print("🚨 Detecting anomalies...")
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(self.processed_features)
        anomaly_scores = iso_forest.score_samples(self.processed_features)
        
        # Statistical outliers (Z-score based)
        z_scores = np.abs(stats.zscore(self.processed_features))
        statistical_outliers = (z_scores > 3).any(axis=1)
        
        # Distance-based outliers
        distances = pdist(self.processed_features.values)
        distance_matrix = squareform(distances)
        avg_distances = np.mean(distance_matrix, axis=1)
        distance_threshold = np.percentile(avg_distances, 90)
        distance_outliers = avg_distances > distance_threshold
        
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        
        self.insights['anomalies'] = {
            'isolation_forest_anomalies': len(anomaly_indices),
            'statistical_outliers': int(statistical_outliers.sum()),
            'distance_based_outliers': int(distance_outliers.sum()),
            'anomaly_percentage': float(len(anomaly_indices) / len(self.processed_features) * 100),
            'top_anomaly_indices': anomaly_indices[:10].tolist(),
            'anomaly_scores_stats': {
                'min': float(anomaly_scores.min()),
                'max': float(anomaly_scores.max()),
                'mean': float(anomaly_scores.mean()),
                'std': float(anomaly_scores.std())
            }
        }
        
        self.anomaly_labels = anomaly_labels
        self.anomaly_scores = anomaly_scores
    
    def _analyze_relationships(self):
        """Analyze relationships and correlations between features."""
        print("🔗 Analyzing feature relationships...")
        
        # Correlation matrix
        corr_matrix = self.processed_features.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        # Feature redundancy analysis
        redundant_features = set()
        for pair in high_corr_pairs:
            if abs(pair['correlation']) > 0.9:
                redundant_features.add(pair['feature2'])  # Remove second feature
        
        self.insights['relationships'] = {
            'correlation_matrix_shape': corr_matrix.shape,
            'high_correlation_pairs': high_corr_pairs,
            'potentially_redundant_features': list(redundant_features),
            'avg_correlation': float(corr_matrix.abs().mean().mean()),
            'max_correlation': float(corr_matrix.abs().max().max()),
            'correlation_distribution': {
                'mean': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                'std': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].std())
            }
        }
    
    def _extract_statistical_insights(self):
        """Extract statistical insights from the data."""
        print("📈 Extracting statistical insights...")
        
        # Descriptive statistics
        desc_stats = self.processed_features.describe()
        
        # Distribution analysis
        skewness = self.processed_features.skew()
        kurtosis = self.processed_features.kurtosis()
        
        # Normality tests (Shapiro-Wilk for small samples)
        normality_results = {}
        if len(self.processed_features)  0.05
                    }
                except:
                    normality_results[col] = {'error': 'Could not perform test'}
        
        self.insights['statistical'] = {
            'descriptive_stats': desc_stats.to_dict(),
            'skewness': dict(skewness.sort_values(ascending=False)),
            'kurtosis': dict(kurtosis.sort_values(ascending=False)),
            'normality_tests': normality_results,
            'highly_skewed_features': list(skewness[abs(skewness) > 2].index),
            'heavy_tailed_features': list(kurtosis[kurtosis > 3].index)
        }
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on insights."""
        print("💡 Generating actionable recommendations...")
        
        recommendations = []
        
        # Dimensionality recommendations
        if self.insights['dimensionality']['dimensionality_reduction_benefit'] > 0.3:
            recommendations.append({
                'category': 'Dimensionality Reduction',
                'priority': 'High',
                'action': f"Consider reducing dimensions from {self.insights['dimensionality']['original_dimensions']} to {self.insights['dimensionality']['recommended_dimensions']}",
                'benefit': f"Reduce complexity by {self.insights['dimensionality']['dimensionality_reduction_benefit']:.1%} while retaining 95% of variance"
            })
        
        # Feature selection recommendations
        if self.insights['relationships']['potentially_redundant_features']:
            recommendations.append({
                'category': 'Feature Selection',
                'priority': 'Medium',
                'action': f"Remove redundant features: {', '.join(self.insights['relationships']['potentially_redundant_features'][:3])}",
                'benefit': "Reduce multicollinearity and improve model interpretability"
            })
        
        # Anomaly handling recommendations
        anomaly_pct = self.insights['anomalies']['anomaly_percentage']
        if anomaly_pct > 5:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'action': f"Investigate {anomaly_pct:.1f}% anomalous data points",
                'benefit': "Improve data quality and model robustness"
            })
        
        # Clustering insights recommendations
        optimal_k = self.insights['patterns']['optimal_clusters']
        if optimal_k > 1:
            recommendations.append({
                'category': 'Segmentation',
                'priority': 'Medium',
                'action': f"Leverage {optimal_k} natural data clusters for segmentation strategies",
                'benefit': "Enable targeted approaches for different data segments"
            })
        
        # Feature engineering recommendations
        highly_skewed = self.insights['statistical']['highly_skewed_features']
        if highly_skewed:
            recommendations.append({
                'category': 'Feature Engineering',
                'priority': 'Medium',
                'action': f"Apply transformations to highly skewed features: {', '.join(highly_skewed[:3])}",
                'benefit': "Improve data distribution and model performance"
            })
        
        # High-level strategic recommendations
        if self.target is not None:
            recommendations.append({
                'category': 'Model Strategy',
                'priority': 'High',
                'action': f"Focus modeling efforts on top features: {', '.join(self.insights['top_features'][:3])}",
                'benefit': "Maximize predictive power while reducing complexity"
            })
        
        self.insights['recommendations'] = recommendations
    
    def get_summary_report(self) -> str:
        """Generate a comprehensive summary report of insights."""
        report = "📋 MULTI-DIMENSIONAL DATA INSIGHTS SUMMARY\n"
        report += "=" * 50 + "\n\n"
        
        # Data overview
        data_info = self.insights['data_info']
        report += f"📊 Data Overview:\n"
        report += f"   • Samples: {data_info['n_samples']:,}\n"
        report += f"   • Features: {data_info['n_features']}\n"
        report += f"   • Data Quality: {data_info['missing_values_handled']} missing values handled\n\n"
        
        # Key findings
        report += "🔍 Key Findings:\n"
        
        # Dimensionality insights
        dim_info = self.insights['dimensionality']
        report += f"   • Dimensionality: Can reduce from {dim_info['original_dimensions']} to {dim_info['recommended_dimensions']} dimensions\n"
        
        # Pattern insights
        pattern_info = self.insights['patterns']
        report += f"   • Patterns: {pattern_info['optimal_clusters']} natural clusters identified\n"
        
        # Anomaly insights
        anomaly_info = self.insights['anomalies']
        report += f"   • Anomalies: {anomaly_info['anomaly_percentage']:.1f}% anomalous data points detected\n"
        
        # Feature insights
        if self.insights['relationships']['potentially_redundant_features']:
            report += f"   • Redundancy: {len(self.insights['relationships']['potentially_redundant_features'])} potentially redundant features\n"
        
        report += "\n"
        
        # Top recommendations
        report += "💡 Top Recommendations:\n"
        for i, rec in enumerate(self.insights['recommendations'][:3], 1):
            report += f"   {i}. [{rec['priority']}] {rec['action']}\n"
            report += f"      → {rec['benefit']}\n"
        
        return report
    
    def plot_insights(self, figsize: Tuple[int, int] = (15, 10)):
        """Create visualizations of key insights."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Multi-Dimensional Data Insights Dashboard', fontsize=16, fontweight='bold')
        
        # 1. PCA Variance Explained
        pca = PCA()
        pca.fit(self.processed_features)
        cumvar_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        axes[0, 0].plot(range(1, len(cumvar_ratio) + 1), cumvar_ratio, 'bo-')
        axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        axes[0, 0].set_xlabel('Number of Components')
        axes[0, 0].set_ylabel('Cumulative Variance Explained')
        axes[0, 0].set_title('PCA: Variance Explained')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance (if available)
        if 'variance_based' in self.insights['feature_importance']:
            importance = self.insights['feature_importance']['variance_based']
            top_features = list(importance.keys())[:10]
            importance_values = [importance[f] for f in top_features]
            
            axes[0, 1].barh(range(len(top_features)), importance_values)
            axes[0, 1].set_yticks(range(len(top_features)))
            axes[0, 1].set_yticklabels(top_features)
            axes[0, 1].set_xlabel('Variance')
            axes[0, 1].set_title('Top 10 Features by Variance')
        
        # 3. Cluster Distribution
        cluster_info = self.insights['patterns']['cluster_profiles']
        cluster_sizes = [cluster_info[c]['size'] for c in cluster_info.keys()]
        cluster_labels = list(cluster_info.keys())
        
        axes[0, 2].pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%')
        axes[0, 2].set_title('Cluster Size Distribution')
        
        # 4. Correlation Heatmap (top features)
        top_features = self.insights.get('top_features', self.processed_features.columns[:10])
        corr_subset = self.processed_features[top_features].corr()
        
        im = axes[1, 0].imshow(corr_subset, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 0].set_xticks(range(len(top_features)))
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_xticklabels(top_features, rotation=45, ha='right')
        axes[1, 0].set_yticklabels(top_features)
        axes[1, 0].set_title('Feature Correlation Heatmap')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Anomaly Scores Distribution
        axes[1, 1].hist(self.anomaly_scores, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=np.percentile(self.anomaly_scores, 10), color='r', linestyle='--', label='Anomaly Threshold')
        axes[1, 1].set_xlabel('Anomaly Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Anomaly Score Distribution')
        axes[1, 1].legend()
        
        # 6. Data Distribution (first PC)
        if hasattr(self, 'pca_data'):
            axes[1, 2].hist(self.pca_data[:, 0], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 2].set_xlabel('First Principal Component')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Distribution of First Principal Component')
        
        plt.tight_layout()
        return fig

# Example usage function
def analyze_sample_data():
    """Example function showing how to use the MultiDimensionalInsightExtractor."""
    
    # Generate sample multi-dimensional data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    # Create correlated features and clusters
    cluster_centers = np.random.randn(3, n_features) * 2
    labels = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])
    
    data = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        data[i] = cluster_centers[labels[i]] + np.random.randn(n_features) * 0.5
    
    # Add some noise and correlations
    data[:, 1] = data[:, 0] * 0.8 + np.random.randn(n_samples) * 0.3  # Correlated feature
    data[:, 2] = data[:, 0] * 0.9 + np.random.randn(n_samples) * 0.2  # Highly correlated
    
    # Create target variable (optional)
    target = (data[:, 0] * 0.3 + data[:, 3] * 0.5 + data[:, 5] * 0.2 + 
              np.random.randn(n_samples) * 0.1)
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_names)
    df['target'] = target
    
    # Initialize and run the insight extractor
    extractor = MultiDimensionalInsightExtractor(target_column='target')
    insights = extractor.fit_transform(df)
    
    # Print summary report
    print(extractor.get_summary_report())
    
    # Create visualizations
    fig = extractor.plot_insights()
    plt.show()
    
    return extractor, insights

# Run example if script is executed directly
if __name__ == "__main__":
    extractor, insights = analyze_sample_data()
