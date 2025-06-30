import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime, timedelta

class UserBehaviorAnalyzer:
    """
    Class for analyzing user behavior and detecting anomalies
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def train(self, features_df, contamination=0.05):
        """
        Train the user behavior anomaly detection model
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing extracted user behavior features
        contamination : float
            Expected proportion of anomalies in the dataset
        
        Returns:
        --------
        self
            Trained model instance
        """
        if features_df.empty:
            raise ValueError("Empty features dataframe provided for training")
        
        # Remove any non-numeric columns except user_id
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'user_id' in features_df.columns:
            self.feature_columns = ['user_id'] + numeric_cols
        else:
            self.feature_columns = numeric_cols
        
        # Extract features for training (exclude user_id)
        X = features_df[numeric_cols]
        total_samples = len(X)
        normal_count = int((1 - contamination) * total_samples)
        y = np.ones(total_samples)
        y[normal_count:] = -1  # simulate contamination

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        # Scale features

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        #X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest model
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=42
        )
        
        y_pred=self.model.fit(X_test_scaled)
        
        print("=== Test Set Evaluation ===")
        print(classification_report(y_test, y_pred, target_names=["Anomaly", "Normal"]))

        return self
    
    def predict(self, features_df):
        """
        Predict anomalies in user behavior
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing extracted user behavior features
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with user_id and anomaly prediction (-1 for anomalies, 1 for normal)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if features_df.empty:
            return pd.DataFrame(columns=['user_id', 'prediction'])
        
        # Extract user_ids
        user_ids = features_df['user_id'].values if 'user_id' in features_df.columns else np.arange(len(features_df))
        
        # Get numeric columns for prediction
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure all required feature columns are present
        for col in set(self.feature_columns) - set(['user_id']):
            if col not in numeric_cols:
                features_df[col] = 0
        
        # Extract features for prediction
        X = features_df[numeric_cols]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Combine user_ids and predictions
        result = pd.DataFrame({
            'user_id': user_ids,
            'prediction': predictions
        })
        
        return result
    
    def predict_anomaly_score(self, features_df):
        """
        Calculate anomaly scores for user behavior
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            DataFrame containing extracted user behavior features
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with user_id and anomaly scores (negative values are more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if features_df.empty:
            return pd.DataFrame(columns=['user_id', 'anomaly_score'])
        
        # Extract user_ids
        user_ids = features_df['user_id'].values if 'user_id' in features_df.columns else np.arange(len(features_df))
        
        # Get numeric columns for prediction
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ensure all required feature columns are present
        for col in set(self.feature_columns) - set(['user_id']):
            if col not in numeric_cols:
                features_df[col] = 0
        
        # Extract features for prediction
        X = features_df[numeric_cols]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Calculate anomaly scores
        scores = self.model.decision_function(X_scaled)
        
        # Combine user_ids and scores
        result = pd.DataFrame({
            'user_id': user_ids,
            'anomaly_score': scores
        })
        
        return result
    
    def save_model(self, filepath):
        """
        Save the trained model to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        
        Returns:
        --------
        self
            Model instance with loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        return self

class UserActivityProfiler:
    """
    Class for profiling user activity patterns and detecting deviations
    """
    def __init__(self):
        self.user_profiles = {}
    
    def build_profiles(self, activity_df):
        """
        Build profiles for users based on their activity patterns
        
        Parameters:
        -----------
        activity_df : pd.DataFrame
            DataFrame containing user activity data
        
        Returns:
        --------
        self
            Updated instance with user profiles
        """
        if activity_df.empty:
            return self
        
        # Group by user_id
        user_groups = activity_df.groupby('user_id')
        
        for user_id, user_data in user_groups:
            # Calculate time-based patterns
            if 'timestamp' in user_data.columns:
                user_data['timestamp'] = pd.to_datetime(user_data['timestamp'])
                user_data['hour'] = user_data['timestamp'].dt.hour
                user_data['day_of_week'] = user_data['timestamp'].dt.dayofweek
                
                # Calculate hourly activity distribution
                hour_dist = user_data['hour'].value_counts().sort_index() / len(user_data)
                
                # Calculate day of week distribution
                dow_dist = user_data['day_of_week'].value_counts().sort_index() / len(user_data)
                
                # Calculate common resources accessed
                if 'resource' in user_data.columns:
                    resource_dist = user_data['resource'].value_counts() / len(user_data)
                else:
                    resource_dist = pd.Series()
                
                # Calculate common actions performed
                if 'action' in user_data.columns:
                    action_dist = user_data['action'].value_counts() / len(user_data)
                else:
                    action_dist = pd.Series()
                
                # Calculate common IP addresses
                if 'ip_address' in user_data.columns:
                    ip_dist = user_data['ip_address'].value_counts() / len(user_data)
                else:
                    ip_dist = pd.Series()
                
                # Store profile
                self.user_profiles[user_id] = {
                    'hour_distribution': hour_dist,
                    'day_of_week_distribution': dow_dist,
                    'resource_distribution': resource_dist,
                    'action_distribution': action_dist,
                    'ip_distribution': ip_dist,
                    'activity_count': len(user_data),
                    'last_activity': user_data['timestamp'].max()
                }
        
        return self
    
    def detect_deviations(self, activity_df, threshold=0.5):
        """
        Detect deviations from user profiles
        
        Parameters:
        -----------
        activity_df : pd.DataFrame
            DataFrame containing recent user activity data
        threshold : float
            Threshold for deviation detection (higher = more sensitive)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with detected deviations
        """
        if activity_df.empty or not self.user_profiles:
            return pd.DataFrame()
        
        deviations = []
        
        for _, activity in activity_df.iterrows():
            user_id = activity['user_id']
            
            # Skip if no profile exists for this user
            if user_id not in self.user_profiles:
                continue
                
            profile = self.user_profiles[user_id]
            
            # Initialize deviation score
            deviation_score = 0
            deviation_reasons = []
            
            # Check time-based deviation
            timestamp = pd.to_datetime(activity['timestamp'])
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            
            # Check hour distribution
            if hour in profile['hour_distribution']:
                hour_prob = profile['hour_distribution'][hour]
                if hour_prob < 0.05:  # Unusual hour
                    deviation_score += 0.3
                    deviation_reasons.append(f"Unusual hour: {hour}")
            else:
                deviation_score += 0.5
                deviation_reasons.append(f"Activity at new hour: {hour}")
            
            # Check day of week distribution
            if day_of_week in profile['day_of_week_distribution']:
                dow_prob = profile['day_of_week_distribution'][day_of_week]
                if dow_prob < 0.05:  # Unusual day
                    deviation_score += 0.3
                    deviation_reasons.append(f"Unusual day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week]}")
            else:
                deviation_score += 0.5
                deviation_reasons.append(f"Activity on new day: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week]}")
            
            # Check resource
            if 'resource' in activity and 'resource_distribution' in profile:
                resource = activity['resource']
                if resource in profile['resource_distribution']:
                    resource_prob = profile['resource_distribution'][resource]
                    if resource_prob < 0.02:  # Rarely accessed resource
                        deviation_score += 0.4
                        deviation_reasons.append(f"Rarely accessed resource: {resource}")
                else:
                    deviation_score += 0.6
                    deviation_reasons.append(f"Access to new resource: {resource}")
            
            # Check action
            if 'action' in activity and 'action_distribution' in profile:
                action = activity['action']
                if action in profile['action_distribution']:
                    action_prob = profile['action_distribution'][action]
                    if action_prob < 0.05:  # Unusual action
                        deviation_score += 0.2
                        deviation_reasons.append(f"Unusual action: {action}")
                else:
                    deviation_score += 0.4
                    deviation_reasons.append(f"New action type: {action}")
            
            # Check IP address
            if 'ip_address' in activity and 'ip_distribution' in profile:
                ip = activity['ip_address']
                if ip in profile['ip_distribution']:
                    ip_prob = profile['ip_distribution'][ip]
                    if ip_prob < 0.05:  # Unusual IP
                        deviation_score += 0.5
                        deviation_reasons.append(f"Unusual IP address: {ip}")
                else:
                    deviation_score += 0.7
                    deviation_reasons.append(f"New IP address: {ip}")
            
            # Check if the deviation score exceeds the threshold
            if deviation_score >= threshold:
                deviation_entry = activity.to_dict()
                deviation_entry['deviation_score'] = deviation_score
                deviation_entry['deviation_reasons'] = '; '.join(deviation_reasons)
                deviations.append(deviation_entry)
        
        # Convert to DataFrame
        if deviations:
            return pd.DataFrame(deviations)
        else:
            return pd.DataFrame()
    
    def update_profiles(self, activity_df, learning_rate=0.1):
        """
        Update user profiles with new activity data
        
        Parameters:
        -----------
        activity_df : pd.DataFrame
            DataFrame containing new user activity data
        learning_rate : float
            Rate at which to incorporate new data (0 to 1)
        
        Returns:
        --------
        self
            Updated instance with updated user profiles
        """
        # Build temporary profiles from new data
        temp_profiler = UserActivityProfiler()
        temp_profiler.build_profiles(activity_df)
        
        # Update existing profiles
        for user_id, new_profile in temp_profiler.user_profiles.items():
            if user_id in self.user_profiles:
                # Update existing profile
                for key in new_profile:
                    if key in ['hour_distribution', 'day_of_week_distribution', 
                              'resource_distribution', 'action_distribution', 'ip_distribution']:
                        # For distributions, blend the old and new
                        old_dist = self.user_profiles[user_id][key]
                        new_dist = new_profile[key]
                        
                        # Combine indices
                        all_indices = set(old_dist.index) | set(new_dist.index)
                        
                        # Create blended distribution
                        blended_dist = pd.Series(index=all_indices)
                        for idx in all_indices:
                            old_val = old_dist.get(idx, 0)
                            new_val = new_dist.get(idx, 0)
                            blended_dist[idx] = (1 - learning_rate) * old_val + learning_rate * new_val
                        
                        # Normalize
                        if blended_dist.sum() > 0:
                            blended_dist = blended_dist / blended_dist.sum()
                        
                        self.user_profiles[user_id][key] = blended_dist
                    elif key == 'activity_count':
                        # Add new activities to count
                        self.user_profiles[user_id][key] += new_profile[key]
                    elif key == 'last_activity':
                        # Update if newer
                        if new_profile[key] > self.user_profiles[user_id][key]:
                            self.user_profiles[user_id][key] = new_profile[key]
            else:
                # Add new profile
                self.user_profiles[user_id] = new_profile
                       
        return self