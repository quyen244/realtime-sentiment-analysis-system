import cv2
import json
import time
import base64
import sys
import logging
import argparse
from kafka import KafkaProducer
from kafka.errors import KafkaError
import pandas as pd 

# Cấu hình Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class AbasasProducer:
    def __init__(self, bootstrap_servers, topic, data_path, messages_per_second=10):
        """
        Initialize Kafka Producer for ABSA streaming
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Kafka topic name
            data_path: Path to CSV data file
            messages_per_second: Rate limiting for message sending
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.data_path = data_path
        self.messages_per_second = messages_per_second
        
        # Khởi tạo Kafka Producer
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                request_timeout_ms=20000,
                retries=5
            )
            logging.info(f"Connected to Kafka at {bootstrap_servers}")
        except Exception as e:
            logging.error(f"Failed to connect to Kafka: {e}")
            sys.exit(1)

    def on_send_success(self, record_metadata):
        """Hàm này được gọi khi gửi THÀNH CÔNG"""
        logging.debug(f"Sent to topic: {record_metadata.topic} partition: {record_metadata.partition}")

    def on_send_error(self, excp):
        """Hàm này được gọi khi gửi THẤT BẠI"""
        logging.error(f"Failed to send message: {excp}")

    def start_streaming(self):
        """
        Stream CSV data to Kafka topic
        Sends each row as JSON message with all columns
        """
        try:
            df = pd.read_csv(self.data_path)
            logging.info(f"Loaded {len(df)} records from {self.data_path}")
            
            # Validate required columns
            required_cols = ['Review', 'Price', 'Shipping', 'Outlook', 'Quality', 
                           'Size', 'Shop_Service', 'General', 'Others']
            if not all(col in df.columns for col in required_cols):
                logging.error(f"Missing required columns. Expected: {required_cols}")
                return
            
            delay = 1.0 / self.messages_per_second
            sent_count = 0
            
            for index, row in df.iterrows():
                # Create message with all fields
                message = {
                    'review': str(row['Review'])
                }
                
                # Send to Kafka asynchronously
                future = self.producer.send(self.topic, value=message)
                future.add_callback(self.on_send_success)
                future.add_errback(self.on_send_error)
                
                sent_count += 1
                if sent_count % 10 == 0:
                    logging.info(f"Sent {sent_count}/{len(df)} messages")
                
                # Rate limiting
                time.sleep(delay)
            
            # Ensure all messages are sent
            self.producer.flush()
            logging.info(f"Successfully sent all {sent_count} messages to topic '{self.topic}'")
            
        except FileNotFoundError:
            logging.error(f"Data file not found: {self.data_path}")
        except Exception as e:
            logging.error(f"Error during streaming: {e}", exc_info=True)
        finally:
            self.producer.close()
            logging.info("Producer closed")
       

# Hàm parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Kafka ABSA Stream Producer")

    parser.add_argument('--topic', type=str, default='absa_stream', 
                        help='Kafka topic name (default: absa_stream)')
    parser.add_argument('--server', type=str, default='kafka:29092', 
                        help='Kafka bootstrap servers (default: localhost:9092)')
    parser.add_argument('--data', type=str, 
                        default=r'/app/archive/test_data.csv',
                        help='Path to CSV data file')
    parser.add_argument('--rate', type=int, default=10, 
                        help='Messages per second (default: 10)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create producer instance with proper arguments
    producer = AbasasProducer(
        bootstrap_servers=args.server,
        topic=args.topic,
        data_path=args.data,
        messages_per_second=args.rate
    )
    
    # Start streaming
    producer.start_streaming()
