import pandas as pd
import re
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any

class LogParser:
    """Utility class for parsing different log formats"""
    
    def __init__(self):
        # Regular expressions for different log formats
        self.apache_regex = r'(\S+) (\S+) (\S+) \[(.*?)\] "(.*?)" (\d+) (\d+)'
        self.syslog_regex = r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+([^:]+)(?:\[(\d+)\])?:\s+(.*)'
        
    def parse_logs(self, log_lines: List[str]) -> Tuple[pd.DataFrame, str]:
        """
        Parse logs in various formats (Apache/Nginx, Syslog, JSON)
        
        Parameters:
        -----------
        log_lines : List[str]
            List of log lines to parse
            
        Returns:
        --------
        tuple
            (DataFrame of parsed logs, detected format)
        """
        # Try to determine the format based on the first non-empty line
        sample_line = ""
        for line in log_lines:
            if line.strip():
                sample_line = line.strip()
                break
        
        # Check if it's JSON
        if sample_line.startswith('{') and sample_line.endswith('}'):
            try:
                json.loads(sample_line)
                return self._parse_json_logs(log_lines), 'json'
            except:
                pass
                
        # Check if it's Apache/Nginx format
        if re.match(self.apache_regex, sample_line):
            return self._parse_apache_logs(log_lines), 'common'
            
        # Check if it's Syslog format
        if re.match(self.syslog_regex, sample_line):
            return self._parse_syslog_logs(log_lines), 'syslog'
            
        # If no format detected, try to parse as generic log
        return self._parse_generic_logs(log_lines), 'generic'
    
    def _parse_apache_logs(self, log_lines: List[str]) -> pd.DataFrame:
        """Parse Apache/Nginx common format logs"""
        parsed_logs = []
        
        for line in log_lines:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(self.apache_regex, line)
            if match:
                ip, ident, user, timestamp, request, status, size = match.groups()
                
                # Extract method, path, protocol from request
                request_parts = request.split()
                method = request_parts[0] if len(request_parts) > 0 else ""
                path = request_parts[1] if len(request_parts) > 1 else ""
                protocol = request_parts[2] if len(request_parts) > 2 else ""
                
                parsed_logs.append({
                    'ip_address': ip,
                    'timestamp': timestamp,
                    'request': request,
                    'method': method,
                    'path': path,
                    'protocol': protocol,
                    'status': status,
                    'size': size,
                    'message': line
                })
        
        return pd.DataFrame(parsed_logs)
    
    def _parse_syslog_logs(self, log_lines: List[str]) -> pd.DataFrame:
        """Parse Syslog format logs"""
        parsed_logs = []
        
        for line in log_lines:
            line = line.strip()
            if not line:
                continue
                
            match = re.match(self.syslog_regex, line)
            if match:
                timestamp, host, program, pid, message = match.groups()
                
                # Extract IP addresses from message
                ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
                ip_addresses = re.findall(ip_pattern, message)
                ip_address = ip_addresses[0] if ip_addresses else ""
                
                # Extract user info if present
                user_pattern = r'user (\S+)'
                user_match = re.search(user_pattern, message, re.IGNORECASE)
                user_id = user_match.group(1) if user_match else ""
                
                parsed_logs.append({
                    'timestamp': timestamp,
                    'host': host,
                    'program': program,
                    'pid': pid if pid else "",
                    'message': message,
                    'ip_address': ip_address,
                    'user_id': user_id,
                    'level': self._detect_log_level(message)
                })
        
        return pd.DataFrame(parsed_logs)
    
    def _parse_json_logs(self, log_lines: List[str]) -> pd.DataFrame:
        """Parse JSON format logs"""
        parsed_logs = []
        
        for line in log_lines:
            line = line.strip()
            if not line:
                continue
                
            try:
                log_json = json.loads(line)
                # Keep original message
                log_json['message'] = line
                parsed_logs.append(log_json)
            except:
                # If parsing fails, add as raw message
                parsed_logs.append({'message': line})
        
        return pd.DataFrame(parsed_logs)
    
    def _parse_generic_logs(self, log_lines: List[str]) -> pd.DataFrame:
        """Parse generic log format"""
        parsed_logs = []
        
        for line in log_lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract IP addresses if present
            ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            ip_addresses = re.findall(ip_pattern, line)
            ip_address = ip_addresses[0] if ip_addresses else ""
            
            # Extract timestamps if present
            timestamp_patterns = [
                r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}',
                r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}'
            ]
            
            timestamp = ""
            for pattern in timestamp_patterns:
                timestamp_match = re.search(pattern, line)
                if timestamp_match:
                    timestamp = timestamp_match.group(0)
                    break
            
            # Extract user info if present
            user_pattern = r'user (\S+)'
            user_match = re.search(user_pattern, line, re.IGNORECASE)
            user_id = user_match.group(1) if user_match else ""
            
            parsed_logs.append({
                'timestamp': timestamp,
                'ip_address': ip_address,
                'user_id': user_id,
                'level': self._detect_log_level(line),
                'message': line
            })
        
        return pd.DataFrame(parsed_logs)
    
    def _detect_log_level(self, message: str) -> str:
        """Detect log level from message content"""
        if re.search(r'(CRITICAL|FATAL|EMERG)', message, re.IGNORECASE):
            return 'CRITICAL'
        elif re.search(r'ERROR', message, re.IGNORECASE):
            return 'ERROR'
        elif re.search(r'WARNING|WARN', message, re.IGNORECASE):
            return 'WARNING'
        elif re.search(r'INFO|NOTICE', message, re.IGNORECASE):
            return 'INFO'
        elif re.search(r'DEBUG', message, re.IGNORECASE):
            return 'DEBUG'
        return 'INFO'
    
    def extract_features_for_anomaly_detection(self, logs_df: pd.DataFrame, log_format: str) -> pd.DataFrame:
        """
        Extract features from parsed logs for anomaly detection
        
        Parameters:
        -----------
        logs_df : pd.DataFrame
            DataFrame containing parsed logs
        log_format : str
            The format of the logs ('common', 'syslog', 'json', 'generic')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with extracted features for anomaly detection
        """
        features = []
        
        if log_format == 'common':
            # For Apache/Nginx logs
            for _, row in logs_df.iterrows():
                feature = {
                    'ip_address': row.get('ip_address', ''),
                    'status_code': int(row.get('status', 0)),
                    'response_size': int(row.get('size', 0)),
                    'is_get': 1 if row.get('method', '').upper() == 'GET' else 0,
                    'is_post': 1 if row.get('method', '').upper() == 'POST' else 0,
                    'is_error': 1 if row.get('status', '').startswith(('4', '5')) else 0,
                    'has_admin_path': 1 if '/admin' in row.get('path', '') else 0,
                    'has_api_path': 1 if '/api' in row.get('path', '') else 0,
                    'path_depth': len(row.get('path', '').split('/')),
                    'timestamp': row.get('timestamp', '')
                }
                features.append(feature)
                
        elif log_format == 'syslog':
            # For Syslog logs
            for _, row in logs_df.iterrows():
                feature = {
                    'host': row.get('host', ''),
                    'program': row.get('program', ''),
                    'is_auth_log': 1 if 'auth' in row.get('program', '').lower() else 0,
                    'is_cron_log': 1 if 'cron' in row.get('program', '').lower() else 0,
                    'is_kernel_log': 1 if 'kernel' in row.get('program', '').lower() else 0,
                    'is_error_level': 1 if row.get('level', '') in ('ERROR', 'CRITICAL') else 0,
                    'is_warning_level': 1 if row.get('level', '') == 'WARNING' else 0,
                    'has_ip': 1 if row.get('ip_address', '') else 0,
                    'has_user': 1 if row.get('user_id', '') else 0,
                    'timestamp': row.get('timestamp', '')
                }
                features.append(feature)
                
        elif log_format == 'json':
            # For JSON logs
            for _, row in logs_df.iterrows():
                feature = {
                    'level': row.get('level', ''),
                    'is_error_level': 1 if row.get('level', '').lower() in ('error', 'critical') else 0,
                    'is_warning_level': 1 if row.get('level', '').lower() == 'warning' else 0,
                    'has_user': 1 if 'user' in row or 'user_id' in row else 0,
                    'has_ip': 1 if 'ip' in row or 'ip_address' in row else 0,
                    'has_error_message': 1 if 'error' in str(row.get('message', '')).lower() else 0,
                    'timestamp': row.get('timestamp', '')
                }
                features.append(feature)
                
        else:
            # For generic logs
            for _, row in logs_df.iterrows():
                message = str(row.get('message', ''))
                feature = {
                    'is_error_level': 1 if row.get('level', '') in ('ERROR', 'CRITICAL') else 0,
                    'is_warning_level': 1 if row.get('level', '') == 'WARNING' else 0,
                    'has_ip': 1 if row.get('ip_address', '') else 0,
                    'has_user': 1 if row.get('user_id', '') else 0,
                    'has_error_keyword': 1 if 'error' in message.lower() else 0,
                    'has_failed_keyword': 1 if 'fail' in message.lower() else 0,
                    'has_warning_keyword': 1 if 'warn' in message.lower() else 0,
                    'has_exception_keyword': 1 if 'exception' in message.lower() else 0,
                    'message_length': len(message),
                    'timestamp': row.get('timestamp', '')
                }
                features.append(feature)
        
        features_df = pd.DataFrame(features)
        
        # Add timestamp-based features if timestamp is available
        if 'timestamp' in features_df.columns and features_df['timestamp'].any():
            try:
                # Try to convert timestamps to datetime objects
                features_df['hour'] = features_df['timestamp'].apply(
                    lambda x: self._extract_hour(x) if x else 0
                )
                features_df['is_business_hours'] = features_df['hour'].apply(
                    lambda x: 1 if 8 <= x <= 18 else 0
                )
                features_df['is_night'] = features_df['hour'].apply(
                    lambda x: 1 if 0 <= x <= 5 else 0
                )
            except:
                # If timestamp parsing fails, add dummy columns
                features_df['hour'] = 12
                features_df['is_business_hours'] = 1
                features_df['is_night'] = 0
        
        return features_df
    
    def _extract_hour(self, timestamp_str: str) -> int:
        """Extract hour from timestamp string in various formats"""
        # Try common timestamp formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%d/%b/%Y:%H:%M:%S',
            '%b %d %H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                # Try to parse with the current format
                dt = datetime.strptime(timestamp_str.split()[0], fmt)
                return dt.hour
            except:
                continue
        
        # If all formats fail, try to extract hour using regex
        hour_match = re.search(r':(\d{2}):', timestamp_str)
        if hour_match:
            try:
                return int(hour_match.group(1))
            except:
                pass
        
        # Default to 12 if extraction fails
        return 12