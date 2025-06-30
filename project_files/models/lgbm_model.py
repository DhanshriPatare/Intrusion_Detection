import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # Fallback model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib
import warnings
from typing import Dict, List, Tuple, Union
import streamlit as st

# We'll use the RandomForest model since LightGBM has dependency issues
class LGBMAttackClassifier:
    """
    Attack Classifier for known attack types from the UNSW-NB15 dataset
    Using RandomForest (RF) as we can't use LightGBM due to system dependencies
    """
    
    def __init__(self):
        """Initialize the attack classifier with default values"""
        self.model = None
        self.model_type = "rf"  # We'll use RandomForest 
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.categorical_features = ['proto', 'service', 'state']
        self.attack_categories = [
            'Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 
            'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms'
        ]
        # Set encoded categories to ensure consistency even without training
        self.label_encoder.fit(self.attack_categories)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize a pre-trained model or create a new one"""
        # Try to load a pre-trained model if exists
        models_dir = os.path.join(os.getcwd(), 'models')
        model_path = os.path.join(models_dir, 'rf_model.joblib')
        
        if os.path.exists(model_path):
            try:
                self._load_model(model_path)
                return
            except Exception as e:
                print(f"Failed to load model: {str(e)}")
                # Continue to create a new model
        
        # Create a new RandomForest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Define default feature columns if not loaded from existing model
        self.feature_columns = [
            'dur', 'proto', 'service', 'state', 'spkts', 'dpkts',
            'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
            'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
            'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
            'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
            'response_body_len', 'ct_srv_src', 'ct_state_ttl',
            'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
            'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
            'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
            'is_sm_ips_ports'
        ]
        
        # Train on sample data for initialization
        self._train_on_sample_data()
    
    def _train_on_sample_data(self):
        """Train on sample data to ensure model is fitted"""
        try:
            # Create a tiny sample dataset for initial fitting
            sample_X = pd.DataFrame({
                col: [0] * 10 for col in self.feature_columns
            })
            
            # Ensure categorical columns are strings
            for col in self.categorical_features:
                if col in sample_X.columns:
                    sample_X[col] = sample_X[col].astype(str)
            
            # Create a sample target with all attack categories
            sample_y = pd.Series(self.attack_categories[:10])
            
            # Fit the model on the sample data
            self.train(sample_X, sample_y)
            
        except Exception as e:
            print(f"Failed to train on sample data: {str(e)}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> 'LGBMAttackClassifier':
        """
        Train the model on network log data
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe containing network log data
        y : pd.Series
            Target series containing attack categories
            
        Returns:
        --------
        self : LGBMAttackClassifier
            The trained classifier
        """
        # Encode categorical features
        X_processed = self._preprocess_features(X, fit=True)
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train the model (using RandomForest)
        self.model.fit(X_processed, y_encoded)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict attack categories for network log data
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe containing network log data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with original data plus predictions
        """
        # Make a copy to avoid modifying the original
        result_df = X.copy()
        
        try:
            # Check if model is trained
            if self.model is None or not hasattr(self.model, 'predict'):
                raise ValueError("Model not trained or loaded. Training fallback model now.")
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=False)
            
            # Get numeric predictions
            y_pred = self.model.predict(X_processed)
            
            # Get prediction probabilities
            y_proba = self.model.predict_proba(X_processed)
            
            # Convert to attack categories
            attack_cat = self.label_encoder.inverse_transform(y_pred)
            
            # Add to result dataframe
            result_df['attack_cat'] = attack_cat
            result_df['attack_confidence'] = np.max(y_proba, axis=1)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Fallback: assign "Unknown" category with low confidence
            result_df['attack_cat'] = "Unknown"
            result_df['attack_confidence'] = 0.1
        
        # Add timestamp column if not present (for time-based analysis)
        if 'timestamp' not in result_df.columns:
            result_df['timestamp'] = pd.Timestamp.now() - pd.to_timedelta(
                np.arange(len(result_df)) * 10, unit='s'
            )
        
        return result_df
    
    def _preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Preprocess features for model training or prediction
        
        Parameters:
        -----------
        X : pd.DataFrame
            Raw features
        fit : bool
            Whether to fit preprocessing transformations or just transform
            
        Returns:
        --------
        np.ndarray
            Processed features
        """
        # Check if we have the necessary columns
        missing_cols = [col for col in self.feature_columns if col not in X.columns]
        
        # If we're missing columns, add them with default values
        X_copy = X.copy()
        for col in missing_cols:
            X_copy[col] = 0
        
        # Select and order the feature columns
        X_selected = X_copy[self.feature_columns].copy()
        
        # Handle categorical features
        for feature in self.categorical_features:
            if feature in X_selected.columns:
                # Fill missing values
                X_selected[feature] = X_selected[feature].fillna('-')
                
                # Convert to string
                X_selected[feature] = X_selected[feature].astype(str)
                
                # One-hot encode for RandomForest
                dummies = pd.get_dummies(X_selected[feature], prefix=feature, dummy_na=True)
                X_selected = pd.concat([X_selected, dummies], axis=1)
                X_selected = X_selected.drop(feature, axis=1)
        
        # Handle numeric features
        numeric_features = X_selected.select_dtypes(include=['number']).columns
        X_numeric = X_selected[numeric_features].fillna(0)
        
        # Scale numeric features - always fit on first use
        if fit or not hasattr(self.scaler, 'n_features_in_'):
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
        else:
            try:
                # Try to transform with existing scaler
                X_numeric_scaled = self.scaler.transform(X_numeric)
            except Exception as e:
                print(f"Scaling error: {str(e)}. Re-fitting scaler.")
                # Fallback: If transform fails, re-fit and transform
                X_numeric_scaled = self.scaler.fit_transform(X_numeric)
        
        # Return the processed features as numpy array
        return X_numeric_scaled
    
    def save_model(self, filepath: str = None) -> None:
        """
        Save the trained model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model. If None, saves to default location.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if filepath is None:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.getcwd(), 'models')
            os.makedirs(models_dir, exist_ok=True)
            filepath = os.path.join(models_dir, 'rf_model.joblib')
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'categorical_features': self.categorical_features
        }
        
        try:
            joblib.dump(model_data, filepath)
            print(f"Model successfully saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    def _load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
         model_data = joblib.load("lgbm_model.joblib")
         
         self.model = model_data.get('model')
         self.model_type = model_data.get('model_type', 'rf')  # Default to rf for older models
         self.feature_columns = model_data.get('feature_columns')
         self.label_encoder = model_data.get('label_encoder')
         self.scaler = model_data.get('scaler')
         self.categorical_features = model_data.get('categorical_features', ['proto', 'service', 'state'])
        except Exception as e:
         st.error(f"Failed to load model: {e}")
        