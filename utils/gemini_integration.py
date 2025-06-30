import os
from typing import Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

class GeminiAPI:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        try:
            genai.configure(api_key=self.api_key)
            self.model_name = "gemini-1.5-pro-latest"
            
            # Verify model availability
            available_models = [m.name for m in genai.list_models()]
            if f"models/{self.model_name}" not in available_models:
                raise ValueError(f"Model {self.model_name} not available")
                
        except Exception as e:
            raise ConnectionError(f"Gemini configuration failed: {str(e)}")

    def get_recommendation(self, attack_type: str, log_data: Dict) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            prompt = f"""You are a cybersecurity expert analyzing a {attack_type} attack. Provide:
            1. Immediate mitigation steps
            2. Long-term prevention strategies
            3. Recommended tools/technologies
            
            Attack Details:
            - Type: {attack_type}
            - Source IP: {log_data.get('src_ip', 'Unknown')}
            - Target IP: {log_data.get('dst_ip', 'Unknown')}
            - Protocol: {log_data.get('proto', 'Unknown')}
            - Port/Service: {log_data.get('service', 'Unknown')}
            - Timestamp: {log_data.get('timestamp', 'Unknown')}
            
            Provide your response in markdown format."""
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"⚠️ Error: {str(e)}\n\nDefault recommendations:\n1. Isolate affected systems\n2. Review firewall rules"

    def analyze_posture(self, logs_summary: Dict) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            prompt = f"""As a CISO, analyze our security posture based on:
            
            Detection Metrics:
            - Total Events: {logs_summary.get('total_logs', 0)}
            - Threats Detected: {logs_summary.get('detected_threats', 0)}
            - Top Attack Vectors: {', '.join(logs_summary.get('top_attack_types', []))}
            
            Provide:
            1. Risk assessment
            2. Recommended improvements
            3. Comparison to industry benchmarks"""
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"⚠️ Error: {str(e)}\n\nManual review recommended"
