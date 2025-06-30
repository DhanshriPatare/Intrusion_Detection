import re
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, Union, List

class UnknownThreatClassifier:
    """
    Detects and classifies unknown attack patterns not covered by UNSW dataset
    using Isolation Forest and advanced pattern matching.
    """
    
    def __init__(self):
        # Real-world attack patterns not in UNSW (expanded list)
        self.threat_patterns = {
            'Credential Stuffing': [
                (r'failed login for \w+ from \d+\.\d+\.\d+\.\d+', 3),
                (r'authentication attempt with \d+ passwords', 3),
                (r'failed password for', 3),
                (r'multiple failed login attempts', 3),
                (r'brute force', 3)
            ],
            'API Abuse': [
                (r'api endpoint \S+ called \d+ times from \d+\.\d+\.\d+\.\d+', 2),
                (r'unusual api parameter: \S+=', 2),
                (r'api rate limit exceeded', 2),
                (r'unusual api call pattern', 2)
            ],
            'Cloud Misconfig': [
                (r'public access enabled for \S+ bucket', 3),
                (r'security group \S+ allows 0\.0\.0\.0/0', 3),
                (r'critical configuration changes', 2),
                (r'public bucket exposure', 3)
            ],
            'Lateral Movement': [
                (r'connection from \d+\.\d+\.\d+\.\d+ to internal \S+', 2),
                (r'smb session established from \S+ to \S+', 2),
                (r'unusual process behavior', 2),
                (r'access to multiple systems', 2)
            ],
            'Cryptojacking': [
                (r'unexpected cpu spike from process \S+', 2),
                (r'crypto miner process detected', 3),
                (r'high cpu usage', 2),
                (r'mining process', 3)
            ],
            'Supply Chain': [
                (r'dependency \S+ contains malicious code', 3),
                (r'package \S+ modified after installation', 2),
                (r'unusual package behavior', 2),
                (r'unauthorized modification', 3)
            ],
            'Data Exfiltration': [
                (r'unusual data transfer', 3),
                (r'large file download', 2),
                (r'unauthorized access to sensitive', 3),
                (r'suspicious download pattern', 2)
            ],
            'Privilege Escalation': [
                (r'elevated privileges', 3),
                (r'sudo command', 2),
                (r'permission change', 2),
                (r'unauthorized admin access', 3)
            ],
            'Web Attacks': [
                (r'sql injection', 3),
                (r'xss attack', 3),
                (r'csrf detected', 3),
                (r'path traversal', 3),
                (r'\.\./\.\./\.\./', 3)
            ]
        }
        
        # Feature extraction pipeline
        self.feature_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                analyzer='word'
            )),
            ('detector', IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Initialize with a small sample dataset to prevent "not fitted" errors
        sample_logs = pd.DataFrame({
            'message': [
                'User login successful',
                'Database connection established',
                'File uploaded successfully',
                'User logged out',
                'System backup completed'
            ]
        })
        self.train(sample_logs)
        
        # Thresholds
        self.pattern_threshold = 1.5  # Minimum pattern score
        self.anomaly_threshold = -0.5  # Isolation Forest score threshold

    def train(self, normal_logs: pd.DataFrame):
        """
        Train on normal logs to establish baseline patterns.
        
        Parameters:
            normal_logs: DataFrame with 'message' column containing clean logs
        """
        # Ensure message column exists and is string type
        if 'message' not in normal_logs.columns:
            # Try to create a message column from available data
            if len(normal_logs.columns) > 0:
                normal_logs['message'] = normal_logs.apply(
                    lambda row: ' '.join([f"{k}={v}" for k, v in row.items() if pd.notna(v)]), 
                    axis=1
                )
            else:
                normal_logs['message'] = ["Empty log entry"]
        
        # Convert message column to string type
        normal_logs['message'] = normal_logs['message'].astype(str)
        
        # Fit the model
        self.feature_pipeline.fit(normal_logs['message'])
        
        try:
            joblib.dump(self.feature_pipeline, 'unknown_threat_model.joblib')
        except Exception as e:
            print(f"Could not save model: {e}")

    def detect(self, log_entry: Union[Dict, pd.Series, str]) -> Dict:
        """
        Detect unknown threats in a log entry.
        
        Parameters:
            log_entry: Dictionary, Series, or string log entry
            
        Returns:
            {
                'category': str,
                'confidence': float (0-3),
                'evidence': List[str],
                'is_unknown': bool
            }
        """
        try:
            # Extract message from log entry
            if isinstance(log_entry, dict):
                message = log_entry.get('message', '')
                if not message and len(log_entry) > 0:
                    # Create message from all fields
                    message = ' '.join([f"{k}={v}" for k, v in log_entry.items() if pd.notna(v)])
            elif isinstance(log_entry, pd.Series):
                if 'message' in log_entry:
                    message = log_entry['message']
                else:
                    # Create message from all fields
                    message = ' '.join([f"{k}={v}" for k, v in log_entry.items() if pd.notna(v)])
            elif isinstance(log_entry, str):
                message = log_entry
            else:
                message = str(log_entry)
            
            # Ensure message is a string
            message = str(message)
            
            # Initialize results
            results = {
                'category': 'Normal',
                'confidence': 0,
                'evidence': [],
                'is_unknown': False
            }
            
            # Pattern Matching (Known Unknowns)
            for category, patterns in self.threat_patterns.items():
                matches = []
                for pattern, score in patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        matches.append(pattern)
                
                if matches:
                    pattern_scores = [score for pattern, score in patterns if pattern in matches]
                    if pattern_scores:
                        max_score = max(pattern_scores)
                        if max_score >= self.pattern_threshold:
                            results.update({
                                'category': category,
                                'confidence': min(max_score, 3),
                                'evidence': matches
                            })
            
            # Anomaly Detection (Unknown Unknowns)
            if results['category'] == 'Normal':
                try:
                    anomaly_score = self.feature_pipeline.decision_function([message])[0]
                    if anomaly_score < self.anomaly_threshold:
                        results.update({
                            'category': 'Uncategorized Threat',
                            'confidence': self._score_to_confidence(anomaly_score),
                            'is_unknown': True,
                            'evidence': ['Anomalous pattern detected']
                        })
                except Exception as e:
                    # Fall back to simple pattern matching
                    if any(keyword in message.lower() for keyword in ['error', 'fail', 'denied', 'attack', 'warning', 'threat', 'unauthorized']):
                        results.update({
                            'category': 'Possible Threat',
                            'confidence': 1.0,
                            'is_unknown': True,
                            'evidence': [f'Basic pattern match (anomaly detection failed: {str(e)})']
                        })
            
            return results
        except Exception as e:
            # Fallback for any unexpected errors
            return {
                'category': 'Normal',
                'confidence': 0,
                'evidence': [f'Error in detection: {str(e)}'],
                'is_unknown': False
            }

    def _score_to_confidence(self, score: float) -> float:
        """Convert anomaly score to confidence level (0-3)"""
        # Lower (more negative) scores indicate higher anomaly
        confidence = min(3, max(0, abs(score) * 3))
        return confidence

    def detect_batch(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Process multiple logs efficiently"""
        results = []
        
        # Check if the dataframe has a message column
        if 'message' not in logs_df.columns:
            # Create message column from all other columns
            logs_df = logs_df.copy()
            logs_df['message'] = logs_df.apply(
                lambda row: ' '.join([f"{k}={v}" for k, v in row.items() if pd.notna(v) and k != 'message']), 
                axis=1
            )
        
        for _, log in logs_df.iterrows():
            try:
                result = self.detect(log)
                results.append(result)
            except Exception as e:
                # Add fallback entry on error
                results.append({
                    'category': 'Normal',
                    'confidence': 0,
                    'evidence': [f'Error: {str(e)}'],
                    'is_unknown': False
                })
        
        result_df = pd.DataFrame(results)
        
        # Copy original columns from the input dataframe if they're not already in the result
        for col in logs_df.columns:
            if col not in result_df.columns:
                result_df[col] = logs_df[col].values
        
        return result_df

    @classmethod
    def load(cls, model_path: str):
        """Load trained classifier"""
        try:
            classifier = cls()
            classifier.feature_pipeline = joblib.load(model_path)
            return classifier
        except Exception as e:
            print(f"Error loading model: {e}")
            # Return a new instance with default settings
            return cls()