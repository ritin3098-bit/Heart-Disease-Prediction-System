import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class KaggleHeartDiseasePredictor:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}

        # Kaggle Heart Disease Dataset features
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        # Alternative feature names
        self.alternative_features = {
            'trestbps': ['trtbps', 'resting_bp', 'restbp'],
            'chol': ['cholesterol', 'chol_level'],
            'thalach': ['thalachh', 'max_hr', 'thalach_max'],
            'oldpeak': ['oldpeak_depression', 'st_depression'],
            'thal': ['thalassemia', 'thal_type']
        }

    def detect_dataset_format(self, df):
        """Automatically detect and standardize column names"""
        detected_features = {}

        # Map standard feature names to possible variations in the dataset
        column_mappings = {
            'target': ['heart disease', 'target', 'output', 'num', 'diagnosis'],
            'age': ['age'],
            'sex': ['sex', 'gender'],
            'cp': ['chest pain type', 'cp', 'chest_pain_type', 'chest_pain'],
            'trestbps': ['bp', 'trestbps', 'trtbps', 'resting_bp', 'restbp', 'blood pressure'],
            'chol': ['cholesterol', 'chol', 'serum_cholesterol'],
            'fbs': ['fbs over 120', 'fbs', 'fasting_bs', 'fasting_blood_sugar'],
            'restecg': ['ekg results', 'restecg', 'resting_ecg', 'rest_ecg'],
            'thalach': ['max hr', 'thalach', 'thalachh', 'maximum_hr', 'max_heart_rate'],
            'exang': ['exercise angina', 'exang', 'exercise_angina'],
            'oldpeak': ['st depression', 'oldpeak', 'depression'],
            'slope': ['slope of st', 'slope', 'st_slope', 'peak_exerc_st'],
            'ca': ['number of vessels fluro', 'ca', 'major_vessels', 'vessels'],
            'thal': ['thallium', 'thal', 'thalassemia']
        }

        # Convert all column names to lowercase with underscores for consistent matching
        df_columns_lower = [str(col).lower().replace(' ', '_') for col in df.columns]
        
        # Map each standard name to the actual column name in the dataset
        for standard_name, possible_names in column_mappings.items():
            for possible_name in possible_names:
                # Convert possible name to lowercase with underscores for comparison
                possible_name_lower = str(possible_name).lower().replace(' ', '_')
                if possible_name_lower in df_columns_lower:
                    # Get the original column name with correct capitalization
                    original_col = df.columns[df_columns_lower.index(possible_name_lower)]
                    detected_features[standard_name] = original_col
                    break

        # Special handling for target variable
        if 'target' not in detected_features and 'heart disease' in [col.lower() for col in df.columns]:
            original_col = [col for col in df.columns if col.lower() == 'heart disease'][0]
            detected_features['target'] = original_col

        return detected_features

    def load_and_prepare_kaggle_data(self, data_path='data/Heart_Disease_Prediction.csv'):
        """Load and prepare Kaggle heart disease dataset"""
        try:
            # Read the dataset
            df = pd.read_csv(data_path)
            print(f"Dataset loaded: {df.shape}")
            print(f"Original columns: {list(df.columns)}")

            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Standardize column names (lowercase with underscores)
            df.columns = [str(col).lower().replace(' ', '_').strip() for col in df.columns]
            
            # Define the expected column mappings based on your dataset
            column_mapping = {
                'age': 'age',
                'sex': 'sex',
                'cp': 'chest_pain_type',
                'trestbps': 'bp',
                'chol': 'cholesterol',
                'fbs': 'fbs_over_120',
                'restecg': 'ekg_results',
                'thalach': 'max_hr',
                'exang': 'exercise_angina',
                'oldpeak': 'st_depression',
                'slope': 'slope_of_st',
                'ca': 'number_of_vessels_fluro',
                'thal': 'thallium',
                'target': 'heart_disease'  # This is our target variable
            }
            
            # Rename columns to standard names
            df_renamed = df.rename(columns={v: k for k, v in column_mapping.items() 
                                          if v in df.columns})
            
            # Ensure we have all required columns
            required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                              'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in df_renamed.columns]
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                print("Available columns:", df_renamed.columns.tolist())
            
            # Handle target variable
            target_col = 'target'
            if target_col not in df_renamed.columns and 'heart_disease' in df_renamed.columns:
                df_renamed[target_col] = df_renamed['heart_disease']
                print("Using 'heart_disease' as the target variable")
            
            if target_col not in df_renamed.columns:
                raise ValueError("Could not identify target column in the dataset")
            
            # Convert target to binary (0/1)
            if df_renamed[target_col].dtype == 'object':
                # Handle string values like 'Presence'/'Absence' or 'Yes'/'No'
                unique_values = df_renamed[target_col].unique()
                print(f"Target unique values: {unique_values}")
                
                if 'Presence' in unique_values or 'presence' in [str(x).lower() for x in unique_values]:
                    # Convert 'Presence' to 1 and 'Absence' to 0
                    df_renamed[target_col] = df_renamed[target_col].apply(
                        lambda x: 1 if str(x).lower() == 'presence' else 0
                    )
                elif 'Yes' in unique_values or 'yes' in [str(x).lower() for x in unique_values]:
                    # Convert 'Yes' to 1 and 'No' to 0
                    df_renamed[target_col] = df_renamed[target_col].apply(
                        lambda x: 1 if str(x).lower() == 'yes' else 0
                    )
                else:
                    # If it's not a recognized string format, try to convert to numeric
                    df_renamed[target_col] = pd.to_numeric(df_renamed[target_col], errors='coerce')
            
            # Convert target to binary (0/1) if it's not already
            if df_renamed[target_col].nunique() > 2:
                # If target has multiple values, convert to binary
                # Assuming that any value > 0 indicates heart disease
                df_renamed[target_col] = (df_renamed[target_col] > 0).astype(int)
            
            # Select only the features we need
            available_features = [f for f in required_columns if f in df_renamed.columns]
            print(f"Available features: {available_features}")
            
            if not available_features:
                raise ValueError("No valid features found in the dataset")
                
            # Update feature names
            self.feature_names = available_features
            
            # Select only the features we need
            selected_columns = available_features + [target_col]
            df_renamed = df_renamed[selected_columns].copy()
            
            # Print data types and sample values for debugging
            print("\nData types and sample values:")
            for col in df_renamed.columns:
                print(f"{col}: {df_renamed[col].dtype}, Sample: {df_renamed[col].iloc[0] if len(df_renamed) > 0 else 'N/A'}")
            
            # Convert all columns to numeric, coercing errors
            for col in df_renamed.columns:
                # Try to convert to numeric, non-numeric become NaN
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
                
                # Fill remaining NaN values with column mean for numeric columns
                if df_renamed[col].dtype in ['int64', 'float64']:
                    # Only fill if there are missing values
                    if df_renamed[col].isna().any():
                        mean_val = df_renamed[col].mean()
                        print(f"Filling {df_renamed[col].isna().sum()} missing values in {col} with mean: {mean_val:.2f}")
                        df_renamed[col].fillna(mean_val, inplace=True)
            
            # Drop any remaining rows with NaN values (should be very few or none)
            initial_rows = len(df_renamed)
            df_renamed = df_renamed.dropna()
            if len(df_renamed) < initial_rows:
                print(f"Dropped {initial_rows - len(df_renamed)} rows with missing values")
            
            # Separate features and target
            X = df_renamed[available_features].copy()
            y = df_renamed[target_col].copy()
            
            # Ensure we have data left after processing
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No valid data remaining after preprocessing")
            
            print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
            print(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            print(f"Error in load_and_prepare_kaggle_data: {str(e)}")
            # Print first few rows for debugging
            if 'df_renamed' in locals() and not df_renamed.empty:
                print("\nFirst few rows of processed data:")
                print(df_renamed.head())
            raise

            X = df_renamed[available_features].copy()
            y = df_renamed['target'].copy()

            X = self._handle_missing_values(X)
            X = self._handle_categorical_features(X)
            X, y = self._validate_and_clean_data(X, y)

            print(f"Final dataset shape: X={X.shape}, y={y.shape}")
            print(f"Target distribution: {y.value_counts().to_dict()}")

            return X, y

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Kaggle dataset not found at {data_path}. "
                "Please place the dataset in the 'data/' folder."
            )

    def _handle_missing_values(self, X):
        """
        Handle missing values in the input data.
        
        Args:
            X (DataFrame): Input data with potential missing values
            
        Returns:
            DataFrame: Processed data with no missing values
        """
        if X.empty:
            print("⚠️ Warning: Empty DataFrame received in _handle_missing_values")
            return pd.DataFrame(columns=self.feature_names).astype(float)
            
        # Create a copy to avoid SettingWithCopyWarning
        X_processed = X.copy()
        
        # Ensure all expected columns are present and have the correct type
        for col in self.feature_names:
            if col not in X_processed.columns:
                # Initialize missing columns with NaN
                X_processed[col] = np.nan
        
        # Get numeric and categorical columns that exist in the DataFrame
        numeric_cols = [col for col in self.feature_names 
                       if col in X_processed.columns and 
                       pd.api.types.is_numeric_dtype(X_processed[col])]
        
        categorical_cols = [col for col in self.feature_names 
                          if col in X_processed.columns and 
                          not pd.api.types.is_numeric_dtype(X_processed[col])]
        
        # Debug info
        print("\n=== Handling Missing Values ===")
        print(f"Numeric columns: {numeric_cols}")
        print(f"Categorical columns: {categorical_cols}")
        print("Missing values before processing:")
        print(X_processed.isnull().sum())
        
        # Handle numeric columns
        if numeric_cols:
            try:
                # First, ensure all numeric columns are actually numeric
                for col in numeric_cols:
                    X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                
                # Only process columns that have at least one non-null value
                valid_numeric_cols = [col for col in numeric_cols 
                                    if X_processed[col].notna().any()]
                
                if not valid_numeric_cols:
                    print("⚠️ No valid numeric columns found for imputation")
                    # If no valid numeric columns, initialize with zeros
                    for col in numeric_cols:
                        X_processed[col] = 0.0
                else:
                    # Fill remaining NaNs with column median
                    for col in valid_numeric_cols:
                        # Calculate median on non-null values only
                        non_null_values = X_processed[col].dropna()
                        if len(non_null_values) > 0:
                            median_val = non_null_values.median()
                            X_processed[col] = X_processed[col].fillna(median_val)
                        else:
                            X_processed[col] = 0.0  # Fallback if all values are NaN
                    
                    # Scale the numeric columns if scaler is available
                    if hasattr(self, 'scaler') and self.scaler is not None:
                        try:
                            # Ensure we only pass numeric columns with no missing values
                            scaling_data = X_processed[valid_numeric_cols].values
                            if np.isnan(scaling_data).any():
                                print("⚠️ NaN values detected before scaling, filling with 0")
                                scaling_data = np.nan_to_num(scaling_data, nan=0.0)
                            
                            scaled_data = self.scaler.transform(scaling_data)
                            X_processed[valid_numeric_cols] = scaled_data
                        except Exception as scale_error:
                            print(f"⚠️ Error during scaling: {scale_error}")
                            # If scaling fails, use the unscaled values
                            pass
                
            except Exception as e:
                print(f"⚠️ Error processing numeric columns: {e}")
                # If there's an error, initialize with zeros
                for col in numeric_cols:
                    X_processed[col] = 0.0
        
        # Handle categorical columns
        for col in categorical_cols:
            try:
                # Convert to string and replace NaN with 'unknown'
                X_processed[col] = X_processed[col].astype(str)
                X_processed[col] = X_processed[col].replace('nan', 'unknown')
                
                # If the column is empty or all 'unknown', set a default value
                if (X_processed[col] == 'unknown').all():
                    X_processed[col] = 'unknown'
                
                # If we have label encoders, transform the categorical values
                if hasattr(self, 'label_encoders') and col in self.label_encoders:
                    try:
                        # Get unique values to avoid unseen labels
                        known_labels = set(self.label_encoders[col].classes_)
                        # Replace unknown labels with the most frequent one or a default
                        X_processed[col] = X_processed[col].apply(
                            lambda x: x if x in known_labels else self.label_encoders[col].classes_[0]
                        )
                        X_processed[col] = self.label_encoders[col].transform(X_processed[col])
                    except Exception as le_error:
                        print(f"⚠️ Error encoding {col}: {le_error}")
                        # If encoding fails, use a default value (0)
                        X_processed[col] = 0
            except Exception as cat_error:
                print(f"⚠️ Error processing categorical column {col}: {cat_error}")
                X_processed[col] = 'unknown'
        
        # Ensure all expected columns are present and in the correct order
        missing_cols = [col for col in self.feature_names if col not in X_processed.columns]
        if missing_cols:
            print(f"⚠️ Adding missing columns: {missing_cols}")
            for col in missing_cols:
                X_processed[col] = 0.0  # Default value for missing columns
        
        # Reorder columns to match feature_names
        X_processed = X_processed.reindex(columns=self.feature_names, fill_value=0.0)
        
        # Final check for any remaining NaN values
        if X_processed.isnull().any().any():
            print("⚠️ NaN values still present after processing, filling with 0")
            X_processed = X_processed.fillna(0.0)
        
        # Debug info
        print("\nMissing values after processing:")
        print(X_processed.isnull().sum())
        print("\nProcessed data sample:")
        print(X_processed.head())
        
        return X_processed

    def _handle_categorical_features(self, X):
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        return X

    def _validate_and_clean_data(self, X, y):
        initial_shape = X.shape[0]
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        y = y.loc[X.index]
        final_shape = X.shape[0]
        if final_shape != initial_shape:
            print(f"Removed {initial_shape - final_shape} invalid rows.")
        return X.reset_index(drop=True), y.reset_index(drop=True)

    def predict(self, input_data):
        """
        Make predictions using the trained ensemble model.
        
        Args:
            input_data: List or array of input features
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if not hasattr(self, 'ensemble_model') or self.ensemble_model is None:
            raise ValueError("Model has not been trained. Please train the model first.")
            
        # Convert input to numpy array if it's not already
        input_array = np.array(input_data)
        
        # Reshape if single sample
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
            
        # Make predictions
        predictions = self.ensemble_model.predict(input_array)
        probabilities = self.ensemble_model.predict_proba(input_array)
        
        return predictions, probabilities
        

    def train_models_with_kaggle_data(self, hyperparameter_tuning=True):
        """
        Train machine learning models on the Kaggle Heart Disease dataset.
        
        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            
        Returns:
            dict: Dictionary containing model results
        """
        # 1️⃣ Load and prepare data
        try:
            X, y = self.load_and_prepare_kaggle_data()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return {}
            
        # 2️⃣ Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 3️⃣ Handle missing values and scale features
        X_train = self._handle_missing_values(X_train)
        X_test = self._handle_missing_values(X_test)
        
        X_train = self._handle_categorical_features(X_train)
        X_test = self._handle_categorical_features(X_test)
        
        # 4️⃣ Scale numeric features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 5️⃣ Define base models
        base_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        model_results = {}

        # 6️⃣ Train and evaluate each model
        for name, model in base_models.items():
            print(f"🔹 Training {name}...")
            try:
                # Models that need scaled input
                if name in ['Logistic Regression', 'SVM', 'KNN']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Evaluate performance
                accuracy = accuracy_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)

                model_results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': float(cv_scores.mean()) if 'cv_scores' in locals() else 0.0,
                    'cv_std': float(cv_scores.std()) if 'cv_scores' in locals() else 0.0,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }

                self.models[name] = model
                print(f"✅ {name}: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}")

            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue

        # 7️⃣ Build and train Ensemble (Voting Classifier)
        print("\n🔹 Training Ensemble (Voting Classifier)...")
        try:
            if not self.models:
                raise ValueError("No models were successfully trained for the ensemble.")
                
            estimators = [(name.replace(' ', '_').lower(), model) for name, model in self.models.items()]
            self.ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
            self.ensemble_model.fit(X_train_scaled, y_train)
            self.models['Ensemble'] = self.ensemble_model

            y_pred_ensemble = self.ensemble_model.predict(X_test_scaled)
            y_pred_proba_ensemble = self.ensemble_model.predict_proba(X_test_scaled)[:, 1]
            acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
            auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
            print(f"✅ Ensemble: Accuracy={acc_ensemble:.4f}, AUC={auc_ensemble:.4f}")
            
            # Calculate cross-validation scores for the ensemble
            try:
                cv_scores = cross_val_score(self.ensemble_model, X_train_scaled, y_train, cv=5)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except Exception as e:
                print(f"⚠️ Could not calculate cross-validation scores for ensemble: {e}")
                cv_mean = 0.0
                cv_std = 0.0
                
            # Add ensemble to results
            model_results['Ensemble'] = {
                'model': self.ensemble_model,
                'accuracy': acc_ensemble,
                'auc_score': auc_ensemble,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred_ensemble,
                'probabilities': y_pred_proba_ensemble
            }
            
        except Exception as e:
            print(f"❌ Error training Ensemble: {e}")

        # 8️⃣ Save all models and scaler
        self._save_models()

        # 9️⃣ Generate evaluation report
        self._generate_evaluation_report(model_results, X_test, y_test)

        print("\n🎉 All models (including ensemble) trained and saved successfully!")
        return model_results

    def _save_models(self):
        """Save all trained models and preprocessing objects to disk"""
        os.makedirs('ml_models/trained_models', exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            joblib.dump(model, f'ml_models/trained_models/{filename}')
        
        # Save preprocessing objects
        joblib.dump(self.scaler, 'ml_models/trained_models/scaler.pkl')
        joblib.dump(self.imputer, 'ml_models/trained_models/imputer.pkl')
        joblib.dump(self.label_encoders, 'ml_models/trained_models/label_encoders.pkl')
        joblib.dump(self.feature_names, 'ml_models/trained_models/feature_names.pkl')
        
        # Save ensemble model if it exists
        if hasattr(self, 'ensemble_model') and self.ensemble_model is not None:
            joblib.dump(self.ensemble_model, 'ml_models/trained_models/ensemble_model.pkl')

    def load_models(self):
        """Load trained models and preprocessing objects from disk"""
        try:
            folder = 'ml_models/trained_models'
            
            # Check if the models directory exists
            if not os.path.exists(folder):
                print(f"❌ Models directory not found at: {os.path.abspath(folder)}")
                return False
                
            print(f"🔍 Looking for model files in: {os.path.abspath(folder)}")
            
            # List all files in the directory for debugging
            if not os.listdir(folder):
                print("❌ No model files found in the directory.")
                return False
                
            print(f"📂 Found files: {os.listdir(folder)}")
            
            # Load preprocessing objects first
            required_files = ['scaler.pkl', 'imputer.pkl', 'label_encoders.pkl', 'feature_names.pkl']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(folder, f))]
            
            if missing_files:
                print(f"❌ Missing required model files: {', '.join(missing_files)}")
                return False
            
            try:
                print("🔧 Loading preprocessing objects...")
                self.scaler = joblib.load(os.path.join(folder, 'scaler.pkl'))
                self.imputer = joblib.load(os.path.join(folder, 'imputer.pkl'))
                self.label_encoders = joblib.load(os.path.join(folder, 'label_encoders.pkl'))
                self.feature_names = joblib.load(os.path.join(folder, 'feature_names.pkl'))
                
                # Load individual models
                self.models = {}
                model_files = [f for f in os.listdir(folder) 
                             if f.endswith('.pkl') 
                             and not any(k in f for k in ['scaler', 'imputer', 'label_encoders', 'feature_names', 'ensemble'])]
                
                if not model_files:
                    print("⚠️ No trained model files found.")
                    return False
                
                for file in model_files:
                    try:
                        model_name = os.path.splitext(file)[0].replace('_', ' ').title()
                        print(f"🔍 Loading model: {model_name}")
                        self.models[model_name] = joblib.load(os.path.join(folder, file))
                    except Exception as e:
                        print(f"⚠️ Error loading model {file}: {e}")
                
                # Load ensemble model if it exists
                ensemble_path = os.path.join(folder, 'ensemble_model.pkl')
                if os.path.exists(ensemble_path):
                    print("🔍 Loading ensemble model...")
                    self.ensemble_model = joblib.load(ensemble_path)
                    self.models['Ensemble'] = self.ensemble_model
                
                if not self.models:
                    print("❌ No models were successfully loaded.")
                    return False
                    
                print(f"✅ Successfully loaded {len(self.models)} models.")
                return True
                
            except Exception as e:
                print(f"❌ Error loading model files: {e}")
                import traceback
                traceback.print_exc()
                return False
            
        except Exception as e:
            print(f"❌ Unexpected error in load_models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, input_data):
        """
        Make predictions using the trained models.

        Args:
            input_data (list or dict): Input data for prediction. Can be a list of values
                                     in the order of feature_names or a dictionary with feature names as keys.

        Returns:
            tuple: (predictions, probabilities) where predictions is a dictionary of model names to predictions
                  and probabilities is a dictionary of model names to probability arrays.
        """
        try:
            # If input is a single value, convert to list
            if not isinstance(input_data, (list, dict, np.ndarray)):
                input_data = [input_data]
                
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                # If input is a dictionary, convert to DataFrame with one row
                input_df = pd.DataFrame([input_data])
            elif isinstance(input_data, (list, np.ndarray)):
                # If input is a list/array, assume it's in the order of feature_names
                if len(input_data) != len(self.feature_names):
                    # If we have exactly one list, assume it's a single sample
                    if len(input_data) == 1 and isinstance(input_data[0], (list, np.ndarray)):
                        input_data = input_data[0]
                    
                    if len(input_data) != len(self.feature_names):
                        raise ValueError(
                            f"Expected {len(self.feature_names)} features, got {len(input_data)}. "
                            f"Expected features: {self.feature_names}"
                        )
                
                # Create DataFrame with proper column names
                input_df = pd.DataFrame([input_data], columns=self.feature_names)
            else:
                raise ValueError("Input data must be a list, numpy array, or dictionary")

            # Ensure all columns are present and in the correct order
            for col in self.feature_names:
                if col not in input_df.columns:
                    input_df[col] = np.nan  # Will be filled by _handle_missing_values

            # Reorder columns to match training data
            input_df = input_df[self.feature_names]

            # Convert all columns to numeric, coercing errors
            for col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

            # Debug: Print input data before processing
            print("\n=== Input Data Before Processing ===")
            print(input_df)
            print("Feature names:", self.feature_names)
            print("Input data shape:", input_df.shape)
            print("Input data types:", input_df.dtypes)
            print("Missing values:\n", input_df.isnull().sum())

            # Handle missing values and scale
            X_processed = self._handle_missing_values(input_df)
            
            # Debug: Print processed data
            print("\n=== Processed Data ===")
            print("Processed data shape:", X_processed.shape)
            print("Processed data types:", X_processed.dtypes)
            print("Missing values after processing:\n", X_processed.isnull().sum().sum())

            # Ensure we have a 2D array for prediction
            if len(X_processed.shape) == 1:
                X_processed = X_processed.reshape(1, -1)

            # Make predictions with each model
            predictions = {}
            probabilities = {}

            for name, model in self.models.items():
                try:
                    # Debug: Print model info
                    print(f"\n=== Making prediction with {name} ===")
                    print(f"Model type: {type(model)}")
                    
                    # Get prediction and probability
                    pred = model.predict(X_processed)
                    
                    # Handle different model types that might not have predict_proba
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_processed)
                        # Get probability of positive class (index 1)
                        prob_positive = float(proba[0][1])  # Convert numpy float to Python float
                    else:
                        # For models without predict_proba, use decision function if available
                        if hasattr(model, 'decision_function'):
                            decision = model.decision_function(X_processed)
                            # Convert decision function to probability-like score between 0 and 1
                            prob_positive = 1 / (1 + np.exp(-decision[0]))
                        else:
                            # Fallback to 1.0 if prediction is 1, else 0.0
                            prob_positive = 1.0 if pred[0] == 1 else 0.0
                    
                    # Store results
                    predictions[name] = int(pred[0])  # Convert numpy int to Python int
                    probabilities[name] = prob_positive
                    
                    # Debug: Print prediction results
                    print(f"Prediction: {predictions[name]}")
                    print(f"Probability: {probabilities[name]:.4f}")
                    
                except Exception as e:
                    print(f"\n❌ Error making prediction with {name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    predictions[name] = -1  # Indicate error
                    probabilities[name] = 0.0

            return predictions, probabilities

        except Exception as e:
            print(f"\n❌ Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return default values in case of error
            default_pred = {name: -1 for name in self.models.keys()}
            default_prob = {name: 0.0 for name in self.models.keys()}
            return default_pred, default_prob

    def _generate_evaluation_report(self, model_results, X_test, y_test):
        os.makedirs('data/analysis', exist_ok=True)
        report_data = []
        for name, result in model_results.items():
            # Skip if the model doesn't have required keys
            if 'accuracy' not in result or 'auc_score' not in result:
                print(f"⚠️ Skipping evaluation for {name} - missing required metrics")
                continue
            report_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'AUC Score': result['auc_score'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        df_report = pd.DataFrame(report_data)
        df_report.to_csv('data/analysis/model_evaluation_report.csv', index=False)
        print("\n✅ Model evaluation report saved at: data/analysis/model_evaluation_report.csv\n")


# ✅ Instantiate for import in views.py
kaggle_heart_predictor = KaggleHeartDiseasePredictor()
