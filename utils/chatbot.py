import google.generativeai as genai
import streamlit as st
import os
from typing import Dict, Any
from utils.gemini_integration import GeminiAPI

class SecurityChatbot:
    def __init__(self):
        try:
            self.gemini = GeminiAPI()
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro-latest",
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048
                }
            )
            self.chat = self.model.start_chat(history=[
                {
                    "role": "user",
                    "parts": ["You are a cybersecurity expert assistant for an enterprise IDS."]
                },
                {
                    "role": "model",
                    "parts": ["I'm your security assistant. How can I help?"]
                }
            ])
            self.available = True
        except Exception as e:
            self.available = False
            self.error_message = str(e)

    def get_response(self, question: str, context: Dict[str, Any] = None) -> str:
        if not self.available:
            return f"Assistant unavailable: {self.error_message}"
        
        try:
            enhanced_prompt = f"{context}\n\n{question}" if context else question
            response = self.chat.send_message(enhanced_prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

def initialize_chatbot() -> SecurityChatbot:
    return SecurityChatbot()

def chatbot_interface(chatbot: SecurityChatbot):
    st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
    <p>Ask security-related questions about detected threats or system events.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help with your security analysis?"}
        ]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if not chatbot.available:
        st.error(f"Chatbot unavailable: {chatbot.error_message}")
        if st.button("Retry Connection"):
            st.session_state.chatbot = initialize_chatbot()
            st.rerun()
    else:
        if prompt := st.chat_input("Ask a security question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    context = {
                        "recent_alerts": st.session_state.get("results", {}),
                        "user_behavior": st.session_state.get("user_behavior_results", {})
                    }
                    response = chatbot.get_response(prompt, context)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})