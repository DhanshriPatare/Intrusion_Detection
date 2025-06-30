import pandas as pd
import numpy as np
import datetime
import re
from typing import Dict, List, Union, Tuple, Optional
import json
import csv,io

@staticmethod
def process_csv_logs(csv_content: str) -> pd.DataFrame:
     """
    Process CSV format logs
    
    Parameters:
    -----------
    csv_content : str
        CSV content as string
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with standardized columns
     """
    # Try to read CSV with different delimiters
     try:
        # Read with pandas
        data = pd.read_csv(io.StringIO(csv_content))
     except:
        try:
            # Try with tab delimiter
            data = pd.read_csv(io.StringIO(csv_content), delimiter='\t')
        except:
            try:
                # Try with semicolon delimiter (common in European locales)
                data = pd.read_csv(io.StringIO(csv_content), delimiter=';')
            except:
                # Fallback to basic CSV reader and try to identify delimiter
                try:
                    dialect = csv.Sniffer().sniff(csv_content[:1000])
                    reader = csv.reader(io.StringIO(csv_content), dialect)
                    headers = next(reader)
                    rows = [row for row in reader]
                    data = pd.DataFrame(rows, columns=headers)
                except:
                    # Last resort, try to parse as text logs
                    return process_text_logs(csv_content)
    
    # Normalize column names
     data.columns = [col.lower().strip() for col in data.columns]
    
    # Check if the data matches expected UNSW-NB15 format
     required_columns = ['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes']
     is_unsw_format = all(col in data.columns for col in required_columns)
    
     if is_unsw_format:
        # Use as is, it's already in the expected format
        return data
    
    # If it doesn't match UNSW format, try to map to a standardized format
     if 'timestamp' not in data.columns and 'time' in data.columns:
        data['timestamp'] = data['time']
     elif 'timestamp' not in data.columns and 'date' in data.columns:
        data['timestamp'] = data['date']
    
    # Extract source/destination if available
     if 'src_ip' in data.columns and 'src_port' in data.columns:
        data['src'] = data['src_ip'] + ':' + data['src_port'].astype(str)
     elif 'source_ip' in data.columns and 'source_port' in data.columns:
        data['src'] = data['source_ip'] + ':' + data['source_port'].astype(str)
    
     if 'dst_ip' in data.columns and 'dst_port' in data.columns:
        data['dst'] = data['dst_ip'] + ':' + data['dst_port'].astype(str)
     elif 'destination_ip' in data.columns and 'destination_port' in data.columns:
        data['dst'] = data['destination_ip'] + ':' + data['destination_port'].astype(str)
    
    # Look for user information
     user_cols = [col for col in data.columns if 'user' in col]
     if user_cols:
        data['user_id'] = data[user_cols[0]]
    
    # Add message column for text analysis if needed
     if 'message' not in data.columns:
        if 'log' in data.columns:
            data['message'] = data['log']
        elif 'event' in data.columns:
            data['message'] = data['event']
        else:
            # Create message from available columns
            message_cols = [col for col in data.columns if data[col].dtype == 'object' 
                           and col not in ['timestamp', 'src', 'dst', 'user_id']]
            if message_cols:
                data['message'] = data[message_cols].astype(str).apply(' '.join, axis=1)
            else:
                # Default to JSON representation of row
                data['message'] = data.apply(lambda row: json.dumps(row.to_dict()), axis=1)
    
     return data
class DataProcessor:
    """
    Utility class for processing and preparing data for the models
    """
    
    def __init__(self):
        self.categorical_features = ['proto', 'service', 'state']
    
    def preprocess_network_data(self, data):
        """
        Preprocess network traffic data for anomaly detection
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw network traffic data
            
        Returns:
        --------
        pd.DataFrame
            Processed data ready for model input
        """
        # Create a copy of the data
        df = data.copy()
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('-')
            else:
                df[col] = df[col].fillna(0)
        
        # Ensure required categorical features exist
        for feature in ['proto', 'service', 'state']:
            if feature not in df.columns:
                if feature == 'proto':
                    df[feature] = 'tcp'  # Default protocol
                elif feature == 'service':
                    df[feature] = 'http'  # Default service
                else:
                    df[feature] = 'unknown'
        
        # Convert categorical features to string
        for feature in self.categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].astype(str)
        
        # Ensure IP addresses are present and extract subnet features
        # Add source and destination IPs if missing
        if 'src_ip' not in df.columns:
            if 'ip_address' in df.columns:
                df['src_ip'] = df['ip_address']
            else:
                df['src_ip'] = '0.0.0.0'  # Default source IP
                
        if 'dst_ip' not in df.columns:
            df['dst_ip'] = '192.168.1.1'  # Default destination IP (gateway)
        
        # Extract subnet information - first octet
        df['src_subnet'] = df['src_ip'].astype(str).apply(
            lambda x: x.split('.')[0] if '.' in x and len(x.split('.')) >= 1 else '0')
        
        df['dst_subnet'] = df['dst_ip'].astype(str).apply(
            lambda x: x.split('.')[0] if '.' in x and len(x.split('.')) >= 1 else '0')
        
        # Extract datetime features if timestamp is available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Ensure all required numeric columns exist
        required_columns = [
            'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl',
            'rate', 'sload', 'dload', 'sinpkt', 'dinpkt'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df
    
    def preprocess_user_data(self, data):
        """
        Preprocess user activity data for behavior analysis
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw user activity data
            
        Returns:
        --------
        pd.DataFrame
            Processed user data ready for model input
        """
        # Create a copy of the data
        df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['user_id', 'timestamp', 'action']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'user_id':
                    df[col] = 'unknown_user'
                elif col == 'timestamp':
                    df[col] = datetime.datetime.now()
                else:
                    df[col] = 'unknown'
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract date features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['is_business_hours'] = df['hour'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
        
        return df
    
    def extract_user_behavior_features(self, data):
        """
        Extract features from user activity data for anomaly detection
        
        Parameters:
        -----------
        data : pd.DataFrame
            Processed user activity data
            
        Returns:
        --------
        pd.DataFrame
            Features for the user behavior model
        """
        # Create a copy of the data
        df = data.copy()
        
        # Group by user_id
        user_features = []
        
        for user_id, group in df.groupby('user_id'):
            features = {
                'user_id': user_id,
                'activity_count': len(group),
                'unique_actions': group['action'].nunique(),
                'business_hours_ratio': group['is_business_hours'].mean(),
                'weekend_ratio': group['is_weekend'].mean()
            }
            
            # Add hour distribution features
            for hour in range(24):
                hour_count = len(group[group['hour'] == hour])
                features[f'hour_{hour}_ratio'] = hour_count / len(group) if len(group) > 0 else 0
            
            # Add day of week distribution features
            for day in range(7):
                day_count = len(group[group['day_of_week'] == day])
                features[f'day_{day}_ratio'] = day_count / len(group) if len(group) > 0 else 0
            
            user_features.append(features)
        
        return pd.DataFrame(user_features)
    
    def process_log_messages(self, data):
        """
        Process log messages for unknown threat detection
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw log data with message column
            
        Returns:
        --------
        pd.DataFrame
            Processed log data
        """
        # Create a copy of the data
        df = data.copy()
        
        # Ensure message column exists
        if 'message' not in df.columns:
            raise ValueError("Log data must contain a 'message' column")
        
        # Fill missing values
        df['message'] = df['message'].fillna('')
        
        # Extract timestamps if available
        timestamp_pattern = r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}'
        
        def extract_timestamp(message):
            match = re.search(timestamp_pattern, message)
            if match:
                try:
                    return pd.to_datetime(match.group(0))
                except:
                    return pd.NaT
            return pd.NaT
        
        if 'timestamp' not in df.columns:
            df['timestamp'] = df['message'].apply(extract_timestamp)
        
        # Extract IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        
        def extract_ips(message):
            return re.findall(ip_pattern, message)
        
        df['ip_addresses'] = df['message'].apply(extract_ips)
        df['ip_count'] = df['ip_addresses'].apply(len)
        
        # Extract common log levels
        log_levels = ['ERROR', 'WARNING', 'INFO', 'DEBUG', 'CRITICAL', 'FATAL']
        
        def extract_log_level(message):
            for level in log_levels:
                if level in message.upper():
                    return level
            return 'UNKNOWN'
        
        df['log_level'] = df['message'].apply(extract_log_level)
        
        return df

    @staticmethod
    def process_logs(file_obj: Union[str, bytes, io.BytesIO]) -> pd.DataFrame:
     """
    Process uploaded log files (CSV or text format)
    
    Parameters:
    -----------
    file_obj : Union[str, bytes, io.BytesIO]
        Uploaded file object or path
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with standardized columns
     """
     # Get file content as string
     if isinstance(file_obj, (str, bytes)):
        file_content = file_obj
     else:
         # Reset file pointer to beginning if it's a file object
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        file_content = file_obj.read() if hasattr(file_obj, 'read') else str(file_obj)
    
    # Convert bytes to string if needed
     if isinstance(file_content, bytes):
        file_content = file_content.decode('utf-8', errors='replace')
    
    # Determine file type and process accordingly
     if hasattr(file_obj, 'name') and isinstance(file_obj.name, str):
        file_name = file_obj.name.lower()
        if file_name.endswith('.csv'):
            return process_csv_logs(file_content)
        elif file_name.endswith('.txt'):
            return process_text_logs(file_content)
        else:
            # Try to infer format
            if ',' in file_content[:1000] and '\n' in file_content[:1000]:
                return process_csv_logs(file_content)
            else:
                return process_text_logs(file_content)
     else:
        # Try to infer format
        if ',' in file_content[:1000] and '\n' in file_content[:1000]:
            return process_csv_logs(file_content)
        else:
            return process_text_logs(file_content)
        
   

def process_text_logs(text_content: str) -> pd.DataFrame:
    """
    Process plain text log files
    
    Parameters:
    -----------
    text_content : str
        Text content as string
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with extracted features
    """
    # Split into lines
    lines = text_content.strip().split('\n')
    
    # Process based on log format
    log_format = identify_log_format(lines[:10])
    
    if log_format == 'apache':
        return process_apache_logs(lines)
    elif log_format == 'syslog':
        return process_syslog(lines)
    elif log_format == 'json':
        return process_json_logs(lines)
    else:
        # Generic processing for unknown formats
        return process_generic_logs(lines)

def identify_log_format(sample_lines: List[str]) -> str:
    """
    Identify the log file format based on sample lines
    
    Parameters:
    -----------
    sample_lines : List[str]
        Sample lines from the log file
        
    Returns:
    --------
    str
        Identified log format: 'apache', 'syslog', 'json', or 'unknown'
    """
    # Check for Apache log format
    apache_pattern = r'^\S+ - - \[\d+/\w+/\d+:\d+:\d+:\d+ [\+\-]\d+\] "(?:GET|POST|PUT|DELETE)'
    
    # Check for syslog format
    syslog_pattern = r'^\w{3}\s+\d+\s+\d+:\d+:\d+\s+\S+\s+'
    
    # Check for JSON format
    json_pattern = r'^\s*\{.*\}\s*$'
    
    # Check sample lines against patterns
    format_counts = {'apache': 0, 'syslog': 0, 'json': 0}
    
    for line in sample_lines:
        if re.match(apache_pattern, line):
            format_counts['apache'] += 1
        elif re.match(syslog_pattern, line):
            format_counts['syslog'] += 1
        elif re.match(json_pattern, line):
            format_counts['json'] += 1
    
    # Return the format with the most matches
    if max(format_counts.values()) > 0:
        return max(format_counts.items(), key=lambda x: x[1])[0]
    else:
        return 'unknown'

def process_apache_logs(lines: List[str]) -> pd.DataFrame:
    """
    Process Apache web server logs
    
    Parameters:
    -----------
    lines : List[str]
        List of log lines
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with extracted features
    """
    apache_pattern = r'(\S+) - - \[(\d+/\w+/\d+:\d+:\d+:\d+) [\+\-]\d+\] "(GET|POST|PUT|DELETE|HEAD) (\S+) \S+" (\d+) (\d+|-)'
    
    parsed_logs = []
    for line in lines:
        match = re.match(apache_pattern, line)
        if match:
            ip, timestamp, method, path, status, size = match.groups()
            
            # Parse timestamp
            try:
                ts = pd.to_datetime(timestamp, format='%d/%b/%Y:%H:%M:%S')
            except:
                ts = pd.NaT
            
            parsed_logs.append({
                'src_ip': ip,
                'timestamp': ts,
                'method': method,
                'path': path,
                'status': int(status),
                'size': int(size) if size != '-' else 0,
                'message': line
            })
    
    # Convert to dataframe
    data = pd.DataFrame(parsed_logs)
    
    # Add extracted features for modeling
    if not data.empty:
        # Add source features
        data['src'] = data['src_ip']
        
        # Add user ID (using IP as proxy)
        data['user_id'] = data['src_ip']
        
        # Add protocol (HTTP for Apache logs)
        data['proto'] = 'http'
        
        # Basic feature extraction for modeling
        data['sbytes'] = data['size']  # Size as response bytes
        data['dbytes'] = 0  # No download bytes in typical Apache logs
        data['dur'] = 0  # Duration not available
        
        # Simplify path to service
        data['service'] = data['path'].apply(
            lambda x: x.split('/')[1] if '/' in x and len(x.split('/')) > 1 else '-'
        )
        
        # Map HTTP status to state
        status_to_state = {
            200: 'OK', 
            404: 'NOT_FOUND', 
            500: 'ERROR',
            403: 'FORBIDDEN'
        }
        data['state'] = data['status'].apply(
            lambda x: status_to_state.get(x, 'OTHER')
        )
    
    return data

def process_syslog(lines: List[str]) -> pd.DataFrame:
    """
    Process syslog format logs
    
    Parameters:
    -----------
    lines : List[str]
        List of log lines
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with extracted features
    """
    syslog_pattern = r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+([^:]+):\s+(.*)'
    
    parsed_logs = []
    for line in lines:
        match = re.match(syslog_pattern, line)
        if match:
            timestamp, host, program, message = match.groups()
            
            # Parse timestamp
            try:
                ts = pd.to_datetime(timestamp)
            except:
                ts = pd.NaT
            
            # Extract IPs if present
            src_ip = extract_ip(message)
            dst_ip = extract_ip(message, skip=src_ip)
            
            parsed_logs.append({
                'timestamp': ts,
                'host': host,
                'program': program,
                'message': message,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'full_message': line
            })
    
    # Convert to dataframe
    data = pd.DataFrame(parsed_logs)
    
    # Add extracted features for modeling
    if not data.empty:
        # Format source and destination
        data['src'] = data['src_ip']
        data['dst'] = data['dst_ip']
        
        # Set program as service
        data['service'] = data['program']
        
        # Set protocol (generic)
        data['proto'] = '-'
        
        # Add user ID if possible (extract from message)
        data['user_id'] = data['message'].apply(extract_user)
        
        # Basic feature extraction for modeling
        data['sbytes'] = 0  # Not available
        data['dbytes'] = 0  # Not available
        data['dur'] = 0  # Not available
        data['state'] = '-'  # Not available
    
    return data

def process_json_logs(lines: List[str]) -> pd.DataFrame:
    """
    Process JSON format logs
    
    Parameters:
    -----------
    lines : List[str]
        List of log lines
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with extracted features
    """
    parsed_logs = []
    for line in lines:
        try:
            log_entry = json.loads(line)
            parsed_logs.append(log_entry)
        except json.JSONDecodeError:
            continue
    
    # Convert to dataframe
    data = pd.DataFrame(parsed_logs)
    
    # Standardize column names
    if not data.empty:
        # Map common JSON log fields to standard format
        field_mapping = {
            '@timestamp': 'timestamp',
            'time': 'timestamp',
            'source.ip': 'src_ip',
            'destination.ip': 'dst_ip',
            'src_ip': 'src_ip',
            'dst_ip': 'dst_ip',
            'source.port': 'src_port',
            'destination.port': 'dst_port',
            'protocol': 'proto',
            'event.action': 'action',
            'user.name': 'user_id',
            'username': 'user_id'
        }
        
        # Rename columns based on mapping
        for old_name, new_name in field_mapping.items():
            if old_name in data.columns and new_name not in data.columns:
                data[new_name] = data[old_name]
        
        # Extract features for modeling
        if 'proto' not in data.columns and 'protocol' in data.columns:
            data['proto'] = data['protocol']
        
        if 'service' not in data.columns:
            if 'app_name' in data.columns:
                data['service'] = data['app_name']
            elif 'application' in data.columns:
                data['service'] = data['application']
            else:
                data['service'] = '-'
        
        # Set numeric fields if not present
        if 'sbytes' not in data.columns:
            data['sbytes'] = 0
        if 'dbytes' not in data.columns:
            data['dbytes'] = 0
        if 'dur' not in data.columns:
            data['dur'] = 0
        
        # Set state field if not present
        if 'state' not in data.columns:
            if 'status' in data.columns:
                data['state'] = data['status']
            elif 'result' in data.columns:
                data['state'] = data['result']
            else:
                data['state'] = '-'
    
    return data

def process_generic_logs(lines: List[str]) -> pd.DataFrame:
    """
    Process generic/unknown format logs
    
    Parameters:
    -----------
    lines : List[str]
        List of log lines
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with basic extracted features
    """
    parsed_logs = []
    for line in lines:
        # Extract timestamp if present
        timestamp = extract_timestamp(line)
        
        # Extract IPs if present
        src_ip = extract_ip(line)
        dst_ip = extract_ip(line, skip=src_ip)
        
        # Extract user if present
        user_id = extract_user(line)
        
        parsed_logs.append({
            'timestamp': timestamp,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'user_id': user_id,
            'message': line
        })
    
    # Convert to dataframe
    data = pd.DataFrame(parsed_logs)
    
    # Add additional features for modeling
    if not data.empty:
        # Format source and destination with default ports
        data['src'] = data['src_ip'].apply(lambda x: f"{x}:0" if x else "-")
        data['dst'] = data['dst_ip'].apply(lambda x: f"{x}:0" if x else "-")
        
        # Set generic protocol and service
        data['proto'] = '-'
        data['service'] = '-'
        
        # Set numeric features to default values
        data['sbytes'] = 0
        data['dbytes'] = 0
        data['dur'] = 0
        data['state'] = '-'
    
    return data

def extract_timestamp(text: str) -> pd.Timestamp:
    """Extract timestamp from text if present"""
    # Check for common timestamp formats
    timestamp_patterns = [
        r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)',  # ISO format
        r'(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}(?:\s[+-]\d{4})?)',  # Apache format
        r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',  # Syslog format
        r'(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}:\d{2})'  # Common log format
    ]
    
    for pattern in timestamp_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return pd.to_datetime(match.group(1))
            except:
                continue
    
    return pd.NaT

def extract_ip(text: str, skip: str = None) -> str:
    """Extract IP address from text"""
    # IPv4 pattern
    ipv4_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    
    # Find all IPv4 addresses
    ips = re.findall(ipv4_pattern, text)
    
    # Filter out the one to skip
    if ips and skip:
        ips = [ip for ip in ips if ip != skip]
    
    # Return the first match or empty string
    return ips[0] if ips else ""

def extract_user(text: str) -> str:
    """Extract username/user ID from text"""
    # Username patterns
    user_patterns = [
        r'user[=:]\s*[\'\"]?([a-zA-Z0-9_\-\.@]+)[\'\"]?',
        r'username[=:]\s*[\'\"]?([a-zA-Z0-9_\-\.@]+)[\'\"]?',
        r'login[=:]\s*[\'\"]?([a-zA-Z0-9_\-\.@]+)[\'\"]?',
        r'as [\'\"]?([a-zA-Z0-9_\-\.@]+)[\'\"]?[@]',
        r'for [\'\"]?([a-zA-Z0-9_\-\.@]+)[\'\"]? from'
    ]
    
    for pattern in user_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return ""

def extract_user_behavior_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract user behavior features for analysis
    
    Parameters:
    -----------
    data : pd.DataFrame
        Processed log data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with user behavior features
    """
    # Check if data has necessary columns
    if data.empty or 'user_id' not in data.columns:
        return pd.DataFrame()
    
    # Group data by user_id
    user_stats = []
    
    for user_id, user_data in data.groupby('user_id'):
        # Skip empty user IDs
        if not user_id or user_id == "":
            continue
        
        # Calculate basic statistics
        user_features = {
            'user_id': user_id,
            'login_count': len(user_data),
            'resource_count': 0,
            'action_count': 0,
            'unique_ip_count': 0,
            'weekend_logins': 0,
            'night_logins': 0,
            'failed_logins': 0,
            'admin_actions': 0,
            'sensitive_access': 0,
            'unusual_location': 0,
            'session_duration_mean': 0,
            'session_duration_std': 0,
            'bytes_sent_mean': 0,
            'bytes_received_mean': 0
        }
        
        # Extract unique IPs
        if 'src_ip' in user_data.columns:
            user_features['unique_ip_count'] = user_data['src_ip'].nunique()
        
        # Extract time-based features if timestamp available
        if 'timestamp' in user_data.columns:
            timestamps = pd.to_datetime(user_data['timestamp'], errors='coerce')
            valid_timestamps = timestamps[~timestamps.isna()]
            
            if not valid_timestamps.empty:
                # Weekend logins
                user_features['weekend_logins'] = sum(valid_timestamps.dt.dayofweek.isin([5, 6]))
                
                # Night logins (10pm - 6am)
                user_features['night_logins'] = sum((valid_timestamps.dt.hour >= 22) | (valid_timestamps.dt.hour < 6))
        
        # Extract protocol/service specific features
        if 'service' in user_data.columns:
            service_counts = user_data['service'].value_counts()
            user_features['resource_count'] = len(service_counts)
            
            # Admin actions (based on service name)
            admin_services = ['admin', 'config', 'management', 'security']
            admin_count = sum(user_data['service'].str.contains('|'.join(admin_services), case=False, na=False))
            user_features['admin_actions'] = admin_count
        
        # State-based features
        if 'state' in user_data.columns:
            # Failed logins
            failed_states = ['FAIL', 'ERROR', 'FORBIDDEN', 'NOT_FOUND']
            failed_count = sum(user_data['state'].str.contains('|'.join(failed_states), case=False, na=False))
            user_features['failed_logins'] = failed_count
        
        # Byte-based features
        if 'sbytes' in user_data.columns:
            user_features['bytes_sent_mean'] = user_data['sbytes'].mean()
        
        if 'dbytes' in user_data.columns:
            user_features['bytes_received_mean'] = user_data['dbytes'].mean()
        
        # Duration-based features
        if 'dur' in user_data.columns:
            user_features['session_duration_mean'] = user_data['dur'].mean()
            user_features['session_duration_std'] = user_data['dur'].std()
        
        # Action count based on message content
        if 'message' in user_data.columns:
            action_words = ['login', 'logout', 'access', 'create', 'delete', 'update', 'upload', 'download']
            action_count = 0
            for word in action_words:
                action_count += sum(user_data['message'].str.contains(word, case=False, na=False))
            user_features['action_count'] = action_count
            
            # Sensitive access based on message content
            sensitive_words = ['password', 'credit', 'admin', 'root', 'sudo', 'sensitive', 'personal', 'key']
            sensitive_count = 0
            for word in sensitive_words:
                sensitive_count += sum(user_data['message'].str.contains(word, case=False, na=False))
            user_features['sensitive_access'] = sensitive_count
        
        # Calculate entropy of user activity
        user_features['activity_entropy'] = calc_entropy(user_features)
        
        user_stats.append(user_features)
    
    # Convert to dataframe
    user_behavior_df = pd.DataFrame(user_stats)
    
    return user_behavior_df

def calc_entropy(probabilities: dict) -> float:
    """
    Calculate entropy of a distribution
    
    Parameters:
    -----------
    probabilities : dict
        Dictionary of counts
        
    Returns:
    --------
    float
        Entropy value
    """
    # Select numeric features
    numeric_features = ['login_count', 'resource_count', 'unique_ip_count', 
                        'weekend_logins', 'night_logins', 'failed_logins',
                        'admin_actions', 'sensitive_access']
    
    # Extract values
    values = []
    for feature in numeric_features:
        if feature in probabilities and isinstance(probabilities[feature], (int, float)):
            values.append(probabilities[feature])
    
    if not values:
        return 0.0
    
    # Calculate total
    total = sum(values)
    
    # Avoid division by zero
    if total == 0:
        return 0.0
    
    # Calculate probabilities
    probs = [value / total for value in values]
    
    # Calculate entropy
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
    
    return entropy