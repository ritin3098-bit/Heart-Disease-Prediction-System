"""
Enhanced Heart Disease Data Analysis
Includes all visualizations and confusion matrix generation
Updated with correct paths for project structure
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

# Set style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class EnhancedHeartAnalyzer:
    """Enhanced analyzer with confusion matrix support"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df.columns = [col.lower().replace(' ', '_') for col in self.df.columns]
            
            # Convert target to binary
            if 'heart_disease' in self.df.columns:
                self.df['target'] = self.df['heart_disease'].apply(
                    lambda x: 1 if str(x).lower() in ['presence', '1', 'yes', 'true'] else 0
                )
            
            print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_model_data(self):
        """Prepare data for model training"""
        # Select numeric features
        feature_cols = [col for col in self.df.columns 
                       if col not in ['heart_disease', 'target'] 
                       and self.df[col].dtype in ['int64', 'float64']]
        
        X = self.df[feature_cols]
        y = self.df['target']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def train_models(self):
        """Train multiple models for comparison"""
        print("\n=== Training Models ===")
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        self.predictions['Random Forest'] = rf.predict(self.X_test)
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        self.predictions['Logistic Regression'] = lr.predict(self.X_test)
        
        # Support Vector Machine
        print("Training SVM...")
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(self.X_train, self.y_train)
        self.models['SVM'] = svm
        self.predictions['SVM'] = svm.predict(self.X_test)
        
        print("All models trained successfully!")
    
    def create_confusion_matrices(self, output_dir):
        """Create confusion matrix visualizations for all models"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n=== Generating Confusion Matrices ===")
        
        for model_name, y_pred in self.predictions.items():
            # Calculate confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       cbar_kws={'label': 'Count'},
                       square=True, linewidths=2, linecolor='black')
            
            plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.2%}', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.xticks([0.5, 1.5], ['No Disease (0)', 'Disease (1)'])
            plt.yticks([0.5, 1.5], ['No Disease (0)', 'Disease (1)'])
            
            # Add text annotations for percentages
            for i in range(2):
                for j in range(2):
                    percentage = cm[i, j] / cm.sum() * 100
                    plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                            ha='center', va='center', fontsize=10, color='gray')
            
            plt.tight_layout()
            filename = f"{output_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {filename}")
            
            # Print classification report
            print(f"\n{model_name} Classification Report:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=['No Disease', 'Disease']))
    
    def create_comparison_plot(self, output_dir):
        """Create a comparison plot of all models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics for all models
        metrics = []
        for model_name, y_pred in self.predictions.items():
            accuracy = accuracy_score(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Calculate sensitivity (recall) and specificity
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        bars1 = ax.bar(x - width, metrics_df['Accuracy'], width, label='Accuracy', color='skyblue')
        bars2 = ax.bar(x, metrics_df['Sensitivity'], width, label='Sensitivity', color='lightcoral')
        bars3 = ax.bar(x + width, metrics_df['Specificity'], width, label='Specificity', color='lightgreen')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Model'])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved: {output_dir}/model_comparison.png")
        print("\n=== Model Performance Summary ===")
        print(metrics_df.to_string(index=False))
    
    def generate_all_visualizations(self, output_dir):
        """Generate all visualizations including the ones you already have"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n=== Generating All Visualizations ===")
        
        # 1. Target Distribution
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='target', data=self.df, palette='viridis')
        plt.title('Distribution of Heart Disease Cases', fontsize=14, fontweight='bold')
        plt.xlabel('Heart Disease (0 = No, 1 = Yes)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        total = float(len(self.df))
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 3,
                   '{:1.1f}%'.format((height/total)*100),
                   ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/target_distribution.png")
        
        # 2. Feature Correlation Matrix
        df_viz = self.df.drop(columns=['heart_disease'], errors='ignore')
        numeric_cols = df_viz.select_dtypes(include=['int64', 'float64']).columns
        
        plt.figure(figsize=(16, 12))
        corr = df_viz[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
                   cbar_kws={"shrink": .8, "label": "Correlation"})
        
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/correlation_matrix.png")
        
        print("All visualizations generated successfully!")


def main():
    """Main execution function"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to script location or use absolute paths
    # Option 1: If script is in project root
    data_path = os.path.join(script_dir, 'data', 'Heart_Disease_Prediction.csv')
    output_dir = os.path.join(script_dir, 'analysis_results')
    
    # Option 2: Use absolute path (comment out Option 1 if using this)
    # data_path = r"C:\Users\LOQ\Desktop\Disease Prediction System\heart_disese_prediction\data\Heart_Disease_Prediction.csv"
    # output_dir = r"C:\Users\LOQ\Desktop\Disease Prediction System\heart_disese_prediction\analysis_results"
    
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Data file not found at {data_path}")
        print("Please update the data_path in the script.")
        return
    
    # Initialize analyzer
    analyzer = EnhancedHeartAnalyzer(data_path)
    
    # Load data
    if not analyzer.load_data():
        print("Failed to load data. Exiting...")
        return
    
    # Prepare data for modeling
    analyzer.prepare_model_data()
    
    # Train models
    analyzer.train_models()
    
    # Generate confusion matrices
    analyzer.create_confusion_matrices(output_dir)
    
    # Create comparison plot
    analyzer.create_comparison_plot(output_dir)
    
    # Generate all other visualizations
    analyzer.generate_all_visualizations(output_dir)
    
    print(f"\n✓ Analysis complete! Check the '{output_dir}' directory.")
    print(f"✓ Generated files:")
    print(f"   - Confusion matrices for all models")
    print(f"   - Model comparison chart")
    print(f"   - Target distribution plot")
    print(f"   - Feature correlation matrix")


if __name__ == "__main__":
    main()