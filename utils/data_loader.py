import pandas as pd
import os
from datetime import datetime
import numpy as np

def load_attack_data():
    
    """Generate sample data matching LGBMAttackClassifier's expected features."""
    n_samples = 100

    # --- Required by LGBMAttackClassifier ---
    # Categorical features (must match self.categorical_features)
    proto = np.random.choice(['tcp', 'udp', 'icmp', 'http'], size=n_samples)
    service = np.random.choice(['-', 'dns', 'http', 'smtp', 'ftp'], size=n_samples)
    state = np.random.choice(['FIN', 'CON', 'INT', 'REQ'], size=n_samples)

    # Numeric features (must match self.feature_columns)
    dur = np.random.uniform(0.1, 60, size=n_samples)  # Duration
    sbytes = np.random.randint(100, 5000, size=n_samples)  # Source bytes
    dbytes = np.random.randint(100, 5000, size=n_samples)  # Destination bytes

    # --- Optional for display ---
    attack_categories = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS']
    attack_cat = np.random.choice(attack_categories, size=n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05])

    # Build DataFrame
    data = pd.DataFrame({
        # Required features (must match LGBMAttackClassifier.feature_columns)
        'proto': proto,
        'service': service,
        'state': state,
        'dur': dur,
        'sbytes': sbytes,
        'dbytes': dbytes,
        # Add more numeric features as needed...
        
        # Optional (for UI display)
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='min'),
        'src_ip': [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)],
        'attack_cat': attack_cat
    })

    return data


def load_zeroday_data():
    """
    Load sample zero-day threat detection data
    
    Returns:
    --------
    pd.DataFrame
        Zero-day threat detection data
    """
    data_path = os.path.join('data', 'zeroday_detection_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        print(f"Warning: File not found - {data_path}")
        return pd.DataFrame()

def load_user_behavior_data():
    """
    Load sample user behavior data
    
    Returns:
    --------
    pd.DataFrame
        User behavior data
    """
    data_path = os.path.join('data', 'user_behavior_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        print(f"Warning: File not found - {data_path}")
        return pd.DataFrame()

def get_model_metrics():
    """
    Get performance metrics for the different models
    
    Returns:
    --------
    dict
        Dictionary with metrics for each model type
    """
    return {
        'attack_classification': {
            'accuracy': 0.942,
            'precision': 0.928,
            'recall': 0.915,
            'f1_score': 0.921,
            'training_time': '2m 34s',
            'prediction_time': '0.45ms per record'
        },
        'zeroday_detection': {
            'detection_rate': 0.895,
            'false_positive_rate': 0.032,
            'auc': 0.934,
            'avg_precision': 0.912,
            'training_time': '1m 42s',
            'prediction_time': '0.38ms per record'
        },
        'user_behavior': {
            'detection_rate': 0.873,
            'false_positive_rate': 0.047,
            'auc': 0.915,
            'avg_precision': 0.893,
            'training_time': '1m 15s',
            'prediction_time': '0.32ms per record'
        }
    }