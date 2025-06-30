# Intrusion Detection System (IDS) using Blockchain and Machine Learning

This project implements an **Intrusion Detection System (IDS)** that combines **Blockchain** for immutable logging and **Machine Learning (ML)** models for anomaly detection. The system aims to detect suspicious activities, such as unauthorized access or malicious activity, by analyzing system logs and using blockchain technology to secure and verify the logs.

## 🎬 Project Demo & Presentation

**Watch the Prototype Demo**  
[![Watch Video](https://img.shields.io/badge/Demo-YouTube-red?logo=youtube)](https://youtu.be/bwHZfG8cHIM)

**View Project PPT**  
[📎 Download Presentation (PDF)](https://drive.google.com/file/d/1yt0gIhuPcKrRoOxBypLFRgCT3vZp9--L/view?usp=sharing)

## 🛠️ Technologies Used
- **Blockchain** (e.g., Ethereum or Hyperledger for secure log storage)
- **Machine Learning** (e.g., LightGBM, Random Forest, SVM, etc. for anomaly detection)
- **Python** (for implementing ML models and blockchain integration)
- **Flask/FastAPI** (for building the backend API)
- **Pandas** & **Scikit-learn** (for data processing and model building)
- **SQL/NoSQL** (for storing user data, logs, and model results)

## 📈 Overview

This system performs the following key tasks:
1. **Data Collection**: It collects system logs (like failed login attempts, use of harmful commands, etc.) from various sources (such as web servers, firewalls, etc.).
2. **Data Preprocessing**: Cleans and processes logs, extracting features like IP address, timestamp, request type, etc.
3. **Model Training**: Uses machine learning algorithms to detect anomalies or intrusions in the logs.
4. **Blockchain Integration**: Stores detected events (intrusions) in a blockchain ledger to ensure immutability and transparency.
5. **Real-time Detection**: Continuously analyzes incoming logs for intrusions and records the results on the blockchain.

## 🚀 Features
- **Real-time Log Monitoring**: The system can detect suspicious activity in real-time.
- **ML Anomaly Detection**: Uses machine learning models to identify unusual patterns or attacks.
- **Blockchain Logging**: All intrusion events are logged on the blockchain for secure and immutable record-keeping.
- **Scalability**: The system is designed to be scalable to handle large amounts of log data.
- **Transparency**: Every detected intrusion is publicly verifiable through the blockchain.

## 🔍 How It Works
1. **Log Data Collection**: System logs are collected from various network and server sources.
2. **Data Preprocessing**: The logs are cleaned and transformed into a format suitable for machine learning.
3. **Anomaly Detection**: The ML model analyzes the logs to identify unusual patterns that could indicate an intrusion.
4. **Blockchain Integration**: Any detected intrusion is recorded on the blockchain, ensuring that the logs cannot be tampered with.
5. **Alert Generation**: Real-time alerts are generated when an intrusion is detected.

## 🧑‍💻 Installation & Setup

# 1. Clone the Repository
```
git clone https://github.com/your-username/intrusion-detection.git
cd intrusion-detection
```
# 2. Create and Activate Virtual Environment
```
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```
# 3. Install Dependencies
```
pip install -r requirements.txt
```
# 4. Run the Streamlit App
```
streamlit run app.py
```
