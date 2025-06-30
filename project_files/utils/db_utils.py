import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

Base = declarative_base()

# Define database models
class NetworkLog(Base):
    """Network traffic logs for intrusion detection"""
    __tablename__ = 'network_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    src_ip = Column(String(50))
    dst_ip = Column(String(50))
    proto = Column(String(10))
    service = Column(String(20))
    dur = Column(Float)
    sbytes = Column(Integer)
    dbytes = Column(Integer)
    sttl = Column(Integer)
    dttl = Column(Integer)
    attack_cat = Column(String(100))
    label = Column(Integer)
    prediction = Column(Integer, nullable=True)
    probability = Column(Float, nullable=True)
    predicted_attack_cat = Column(String(100), nullable=True)
    verified = Column(Boolean, default=False)
    blockchain_hash = Column(String(256), nullable=True)
    
    def __repr__(self):
        return f"<NetworkLog(timestamp='{self.timestamp}', src_ip='{self.src_ip}', " \
               f"attack_cat='{self.attack_cat}', prediction={self.prediction})>"


class UserBehavior(Base):
    """User behavior logs for anomaly detection"""
    __tablename__ = 'user_behaviors'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(String(100))
    ip_address = Column(String(50))
    action = Column(String(100))
    resource = Column(String(100))
    duration = Column(Integer)
    bytes_transferred = Column(Integer)
    location = Column(String(100))
    device_type = Column(String(50))
    session_count = Column(Integer)
    failed_attempts = Column(Integer)
    risk_score = Column(Float)
    prediction = Column(Integer, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    verified = Column(Boolean, default=False)
    blockchain_hash = Column(String(256), nullable=True)
    
    def __repr__(self):
        return f"<UserBehavior(timestamp='{self.timestamp}', user_id='{self.user_id}', " \
               f"action='{self.action}', prediction={self.prediction})>"


class LogEntry(Base):
    """Raw log entries for unknown threat detection"""
    __tablename__ = 'log_entries'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    level = Column(String(20))
    source = Column(String(100))
    message = Column(Text)
    ip_address = Column(String(50))
    user_id = Column(String(100), nullable=True)
    action = Column(String(100))
    status = Column(String(50))
    is_threat = Column(Boolean, nullable=True)
    threat_category = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    verified = Column(Boolean, default=False)
    blockchain_hash = Column(String(256), nullable=True)
    
    def __repr__(self):
        return f"<LogEntry(timestamp='{self.timestamp}', source='{self.source}', " \
               f"is_threat={self.is_threat})>"


class ModelMetrics(Base):
    """Model performance metrics"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    model_type = Column(String(100))
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc = Column(Float, nullable=True)
    training_date = Column(DateTime, default=datetime.datetime.utcnow)
    parameters = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<ModelMetrics(model_name='{self.model_name}', accuracy={self.accuracy})>"


class DatabaseManager:
    """Manage database operations"""
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if self.db_url and self.db_url.startswith("postgres://"):
            self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)
        
        self.engine = None
        self.Session = None
        self.connection_ready = False
    
    def initialize(self):
        """Initialize database connection and create tables"""
        if not self.db_url:
            raise ValueError("Database URL not found in environment variables")
        
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        self.connection_ready = True
        return True
    
    def get_session(self):
        """Get a database session"""
        if not self.connection_ready:
            self.initialize()
        return self.Session()
    
    def add_network_logs(self, logs_df):
        """Add network logs to the database from a DataFrame"""
        if not self.connection_ready:
            self.initialize()
        
        session = self.Session()
        
        try:
            # Convert DataFrame to list of dictionaries
            records = logs_df.to_dict('records')
            
            # Create NetworkLog objects and add to session
            for record in records:
                log = NetworkLog(**record)
                session.add(log)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_user_behaviors(self, behaviors_df):
        """Add user behaviors to the database from a DataFrame"""
        if not self.connection_ready:
            self.initialize()
        
        session = self.Session()
        
        try:
            # Convert DataFrame to list of dictionaries
            records = behaviors_df.to_dict('records')
            
            # Create UserBehavior objects and add to session
            for record in records:
                behavior = UserBehavior(**record)
                session.add(behavior)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_log_entries(self, entries_df):
        """Add log entries to the database from a DataFrame"""
        if not self.connection_ready:
            self.initialize()
        
        session = self.Session()
        
        try:
            # Convert DataFrame to list of dictionaries
            records = entries_df.to_dict('records')
            
            # Create LogEntry objects and add to session
            for record in records:
                entry = LogEntry(**record)
                session.add(entry)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_network_logs(self, limit=100, attack_type=None):
        """Retrieve network logs from the database"""
        if not self.connection_ready:
            self.initialize()
        
        session = self.Session()
        
        try:
            query = session.query(NetworkLog)
            
            if attack_type:
                query = query.filter(NetworkLog.attack_cat == attack_type)
            
            logs = query.order_by(NetworkLog.timestamp.desc()).limit(limit).all()
            
            # Convert to DataFrame
            log_dicts = [log.__dict__ for log in logs]
            for log_dict in log_dicts:
                log_dict.pop('_sa_instance_state', None)
            
            return pd.DataFrame(log_dicts)
        finally:
            session.close()
    
    def get_user_behaviors(self, limit=100, user_id=None):
        """Retrieve user behaviors from the database"""
        if not self.connection_ready:
            self.initialize()
        
        session = self.Session()
        
        try:
            query = session.query(UserBehavior)
            
            if user_id:
                query = query.filter(UserBehavior.user_id == user_id)
            
            behaviors = query.order_by(UserBehavior.timestamp.desc()).limit(limit).all()
            
            # Convert to DataFrame
            behavior_dicts = [behavior.__dict__ for behavior in behaviors]
            for behavior_dict in behavior_dicts:
                behavior_dict.pop('_sa_instance_state', None)
            
            return pd.DataFrame(behavior_dicts)
        finally:
            session.close()
    
    def get_log_entries(self, limit=100, source=None):
        """Retrieve log entries from the database"""
        if not self.connection_ready:
            self.initialize()
        
        session = self.Session()
        
        try:
            query = session.query(LogEntry)
            
            if source:
                query = query.filter(LogEntry.source == source)
            
            entries = query.order_by(LogEntry.timestamp.desc()).limit(limit).all()
            
            # Convert to DataFrame
            entry_dicts = [entry.__dict__ for entry in entries]
            for entry_dict in entry_dicts:
                entry_dict.pop('_sa_instance_state', None)
            
            return pd.DataFrame(entry_dicts)
        finally:
            session.close()
    
    def update_model_metrics(self, model_name, metrics):
        """Update model performance metrics"""
        if not self.connection_ready:
            self.initialize()
        
        session = self.Session()
        
        try:
            # Check if metrics for this model already exist
            existing = session.query(ModelMetrics).filter(
                ModelMetrics.model_name == model_name
            ).first()
            
            if existing:
                # Update existing metrics
                for key, value in metrics.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                # Create new metrics
                metrics['model_name'] = model_name
                new_metrics = ModelMetrics(**metrics)
                session.add(new_metrics)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_model_metrics(self, model_name=None):
        """Retrieve model performance metrics"""
        if not self.connection_ready:
            self.initialize()
        
        session = self.Session()
        
        try:
            query = session.query(ModelMetrics)
            
            if model_name:
                query = query.filter(ModelMetrics.model_name == model_name)
            
            metrics = query.all()
            
            # Convert to DataFrame
            metric_dicts = [metric.__dict__ for metric in metrics]
            for metric_dict in metric_dicts:
                metric_dict.pop('_sa_instance_state', None)
            
            return pd.DataFrame(metric_dicts)
        finally:
            session.close()


# Global instance
db_manager = DatabaseManager()