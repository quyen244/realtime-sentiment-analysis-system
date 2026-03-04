"""
Centralized configuration for ABSA Streaming System
"""
import os

class Config:
    """Main configuration class"""
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    KAFKA_TOPIC = "absa_stream"
    
    # PostgreSQL Configuration
    POSTGRES_HOST = "host.docker.internal"
    POSTGRES_PORT = 5432
    POSTGRES_DB = "absa_db"
    POSTGRES_USER = "postgres"
    POSTGRES_PASSWORD = "1234"
    POSTGRES_TABLE = "sentiment_analysis"
    POSTGRES_MODEL_TABLE = "model_versions"
    
    # JDBC URL for Spark
    POSTGRES_JDBC_URL = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    POSTGRES_DRIVER = "org.postgresql.Driver"
    
    # Model Configuration
    PROJECT_ROOT = "/opt/project"
    
    # Model Configuration
    # (Lưu vào folder models bên trong project đã map)
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    CANDIDATE_MODEL_DIR = os.path.join(MODEL_DIR, "candidates")
    CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "vietnamese_sentiment_model.keras")

    # Data Configuration
    DATA_DIR = os.path.join(PROJECT_ROOT, "archive")
    TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.csv")
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.csv")
    
    # Spark Checkpoint (Cũng nên đưa vào project hoặc volume riêng để không mất)
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "absa_streaming")
    SPARK_APP_NAME = "ABSA_Streaming_System"
    
    # Streaming Configuration
    PROCESSING_TIME = "5 seconds"
    MAX_OFFSETS_PER_TRIGGER = 10
    
    # ABSA Labels
    ASPECT_LABELS = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']
    SENTIMENT_MAP = {
        -1: 'None',
        0: 'Negative', 
        1: 'Positive',
        2: 'Neutral'
    }
    
    # Airflow Configuration
    AIRFLOW_DAG_ID = "absa_model_retraining"
    RETRAINING_SCHEDULE = "0 2 * * 0"  # Every Sunday at 2 AM
    
    # Dashboard Configuration
    DASHBOARD_REFRESH_INTERVAL = 2  # seconds
    DASHBOARD_PORT = 8501
    
    @classmethod
    def get_postgres_connection_string(cls):
        """Return PostgreSQL connection string for SQLAlchemy"""
        return f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
    
    @classmethod
    def get_jdbc_properties(cls):
        """Return JDBC properties for Spark"""
        return {
            "user": cls.POSTGRES_USER,
            "password": cls.POSTGRES_PASSWORD,
            "driver": cls.POSTGRES_DRIVER
        }
