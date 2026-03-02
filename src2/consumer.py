import sys
import os
import logging
import warnings
import functools
import builtins

# --- 1. GLOBAL MONKEY PATCH (Cho Driver) ---
# Vá lỗi encoding
_original_open = builtins.open
def patched_open(file, mode='r', *args, **kwargs):
    if 'b' not in mode and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return _original_open(file, mode, *args, **kwargs)
builtins.open = patched_open

# --- IMPORTS SAU KHI VÁ ---
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf

# --- CẤU HÌNH MÔI TRƯỜNG ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["PYTHONUTF8"] = "1"
# Quan trọng: Trỏ đúng python executable hiện tại
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

warnings.filterwarnings("ignore")

# Logging Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AbsaConsumer")

# ==========================================
# CONFIGURATION
# ==========================================
class AppConfig:
    KAFKA_BOOTSTRAP = "kafka:29092"
    KAFKA_TOPIC = "absa_stream"
    
    # Đường dẫn file
    VECTORIZER_PATH = r"/app/src2/vectorizer_part.keras" # Lưu ý: Code dưới dùng vocab.pkl, bỏ qua file này nếu không dùng
    ONNX_MODEL_PATH = r"/app/src2/absa_model.onnx"
    VOCAB_PATH = r"/app/src2/vocab_clean.pkl"
    
    CHECKPOINT_DIR = "./spark_checkpoints/absa_streaming"
    
    # Database
    PG_URL = "jdbc:postgresql://host.docker.internal:5432/absa_db"
    PG_TABLE = "sentiment_analysis"
    PG_USER = "postgres"
    PG_PASS = "1234"
    PG_DRIVER = "org.postgresql.Driver"

# ==========================================
# PREPROCESSING & UDF
# ==========================================
def preprocess_text(text):
    import re 
    # Giữ nguyên logic preprocess của bạn
    stopwords = set(['là','ở','và','có','cho','của','trong','được','với','một','những','khi','thì','như','này','đó','các','đã','ra','về','ạ','rồi','gì','nào','sẽ','nha','nhé','à','ơi','ha','vậy','nhỉ','thôi','chứ','đi','thật','luôn','quá','tôi','bạn','ai','cũng','lại','sẽ'])
    word_label = {'dc': 'được','đc': 'được','k': 'không','ko': 'không','kh': 'không','hong': 'không','hok': 'không','hum': 'hôm','mn': 'mọi_người','mik': 'mình','mk': 'mình','j': 'gì','vs': 'với','cx': 'cũng','ntn': 'như_thế_nào','thik': 'thích','thjk': 'thích','dcmm': 'được','ok': 'ổn','oki': 'ổn','oke': 'ổn','okie': 'ổn','ms': 'mới','trc': 'trước','sau': 'sau','qa': 'quá','qaá': 'quá'}
    
    if text is None: return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [word_label.get(token, token) for token in tokens]
    tokens = [token for token in tokens if token and token not in stopwords]
    return ' '.join(tokens)

def create_predict_udf(vocab_path, onnx_path):
    # Khai báo kiểu trả về là Array Integer
    @pandas_udf(ArrayType(IntegerType()))
    def predict_udf(batch_series: pd.Series) -> pd.Series:
        # --- WORKER MONKEY PATCH (QUAN TRỌNG NHẤT) ---
        # Code này chạy trên Worker Node
        import sys
        import numpy as np
        # if not hasattr(np, "_core"):
        #     from types import ModuleType
        #     mock_core = ModuleType("numpy._core")
        #     for attr in dir(np.core):
        #         if not attr.startswith("__"):
        #             try: setattr(mock_core, attr, getattr(np.core, attr))
        #             except: pass
        #     sys.modules["numpy._core"] = mock_core
        #     np._core = mock_core
        
        # Import thư viện bên trong UDF để tránh lỗi serialization
        import onnxruntime as ort
        import pickle
        
        # --- Lazy Loading Model ---
        if not hasattr(predict_udf, 'session'):
            # Load Vocab
            with open(vocab_path, 'rb') as f:
                vocab_list = pickle.load(f)
            predict_udf.vocab_dict = {word: i for i, word in enumerate(vocab_list)}
            predict_udf.max_len = 750
            
            # Load ONNX
            # Tắt logging của ONNX để đỡ rác log
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            predict_udf.session = ort.InferenceSession(onnx_path, sess_options)
            predict_udf.input_name = predict_udf.session.get_inputs()[0].name
            
        # --- Inference Loop ---
        try:
            all_vectors = []
            for text in batch_series:
                cleaned = preprocess_text(text)
                tokens = cleaned.split()
                
                # Vectorize
                vector = [predict_udf.vocab_dict.get(word, 1) for word in tokens] # 1 is UNK
                
                # Padding/Truncating
                if len(vector) < predict_udf.max_len:
                    vector = vector + [0] * (predict_udf.max_len - len(vector))
                else:
                    vector = vector[:predict_udf.max_len]
                all_vectors.append(vector)

            input_data = np.array(all_vectors, dtype=np.int64)
            
            # Run ONNX
            onnx_preds = predict_udf.session.run(None, {predict_udf.input_name: input_data})
            
            # Post-process: Argmax -> Subtract 1 -> Result
            # onnx_preds là list các output (8 outputs cho 8 aspects)
            # Mỗi output là (batch_size, num_classes)
            
            # Stack lại thành (batch_size, 8, num_classes) nếu cần, hoặc xử lý từng cái
            # Cách cũ của bạn: [np.argmax(p, axis=1) for p in onnx_preds] -> list 8 array (batch,)
            
            all_aspect_preds = [np.argmax(p, axis=1) for p in onnx_preds] 
            # Stack columns -> (batch, 8)
            final_matrix = np.stack(all_aspect_preds, axis=1)
            final_matrix = final_matrix - 1 
            
            return pd.Series(final_matrix.tolist())
            
        except Exception as e:
            # In lỗi ra stderr để worker log bắt được
            sys.stderr.write(f"ERROR IN UDF: {str(e)}\n")
            # Trả về mảng -1 nếu lỗi để không làm sập stream
            return pd.Series([[-1]*8 for _ in range(len(batch_series))])

    return predict_udf

# ==========================================
# PIPELINE
# ==========================================
class SentimentAnalysisPipeline:
    def __init__(self):
        self.spark = self._init_spark()
        self.predict_udf = create_predict_udf(AppConfig.VOCAB_PATH, AppConfig.ONNX_MODEL_PATH)
        self.schema = StructType([StructField("review", StringType(), True)])

    def _init_spark(self):
        # Spark package config
        kafka_pkg = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"
        postgres_pkg = "org.postgresql:postgresql:42.5.0"
        
        spark = SparkSession.builder \
            .appName("ABSA_Streaming_System") \
            .config("spark.jars.packages", f"{kafka_pkg},{postgres_pkg}") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "20") \
            .config("spark.executor.memory", "2g") \
            .config("spark.executor.memoryOverhead", "2g") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        return spark

    def read_stream(self):
        return self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", AppConfig.KAFKA_BOOTSTRAP) \
            .option("subscribe", AppConfig.KAFKA_TOPIC) \
            .option("startingOffsets", "earliest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", 10) \
            .load() \
            .selectExpr("CAST(value AS STRING)") \
            .select(F.from_json("value", self.schema).alias("data")) \
            .select("data.review") # Chỉ lấy cột review

    def process_stream(self, df_stream):
        # Apply UDF
        processed = df_stream.withColumn("predictions", self.predict_udf(F.col("review")))
        
        # Extract columns
        return processed.select(
            F.col("review"),
            F.col("predictions")[0].alias("pred_price"),
            F.col("predictions")[1].alias("pred_shipping"),
            F.col("predictions")[2].alias("pred_outlook"),
            F.col("predictions")[3].alias("pred_quality"),
            F.col("predictions")[4].alias("pred_size"),
            F.col("predictions")[5].alias("pred_shop_service"),
            F.col("predictions")[6].alias("pred_general"),
            F.col("predictions")[7].alias("pred_others"),
            F.current_timestamp().alias("processed_at")
        )

    @staticmethod
    def _write_to_postgres(batch_df, batch_id):
        logger.info(f"Writing Batch ID: {batch_id}")
        try:
            batch_df.write \
                .format("jdbc") \
                .option("url", AppConfig.PG_URL) \
                .option("dbtable", AppConfig.PG_TABLE) \
                .option("user", AppConfig.PG_USER) \
                .option("password", AppConfig.PG_PASS) \
                .option("driver", AppConfig.PG_DRIVER) \
                .mode("append") \
                .save()
            logger.info(f"Batch {batch_id} DONE.")
        except Exception as e:
            logger.error(f"Batch {batch_id} FAILED: {e}")

    def start(self):
        df = self.read_stream()
        processed = self.process_stream(df)
        
        query = processed.writeStream \
            .foreachBatch(self._write_to_postgres) \
            .option("checkpointLocation", AppConfig.CHECKPOINT_DIR) \
            .trigger(processingTime="10 seconds") \
            .start()
            
        query.awaitTermination()

if __name__ == "__main__":
    SentimentAnalysisPipeline().start()