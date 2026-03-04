"""
Model Training Script for ABSA System
Trains a new ABSA model and saves it with versioning
"""

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import re
from datetime import datetime
import psycopg2
import json
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config

# ==========================================
# TEXT PREPROCESSING
# ==========================================
def preprocess_text(text):
    """Same preprocessing as in consumer.py"""
    stopwords = set([
        'là','ở','và','có','cho','của','trong','được','với','một',
        'những','khi','thì','như','này','đó','các','đã','ra','về',
        'ạ','rồi','gì','nào','sẽ','nha','nhé','à','ơi','ha','vậy',
        'nhỉ','thôi','chứ','đi','thật','luôn','quá','tôi','bạn','ai','cũng', 'lại', 'sẽ'
    ])
    
    word_label = {
        'dc': 'được','đc': 'được','k': 'không','ko': 'không','kh': 'không',
        'hong': 'không','hok': 'không','hum': 'hôm','mn': 'mọi_người',
        'mik': 'mình','mk': 'mình','j': 'gì','vs': 'với','cx': 'cũng',
        'ntn': 'như_thế_nào','thik': 'thích','thjk': 'thích','dcmm': 'được',
        'ok': 'ổn','oki': 'ổn','oke': 'ổn','okie': 'ổn','ms': 'mới','trc': 'trước',
        'sau': 'sau','qa': 'quá','qaá': 'quá'
    }
    
    text = str(text).lower()
    text = re.sub(r"[^a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [word_label.get(token, token) for token in tokens]
    tokens = [token for token in tokens if token and token not in stopwords]
    
    return ' '.join(tokens)

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
def create_absa_model(vocab_size=10000, embedding_dim=128, max_length=100, num_aspects=8, num_classes=4):
    """
    Create multi-output ABSA model
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        max_length: Maximum sequence length
        num_aspects: Number of aspects (8)
        num_classes: Number of sentiment classes (4: None=-1, Neg=0, Pos=1, Neu=2)
    """
    # Input layer
    text_input = layers.Input(shape=(1,), dtype=tf.string, name='text_input')
    
    # Text vectorization layer
    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_length
    )
    
    # Embedding
    x = vectorize_layer(text_input)
    x = layers.Embedding(vocab_size, embedding_dim)(x)
    
    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    # x = layers.Bidirectional(layers.LSTM(64))(x)
    # x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Output heads for each aspect
    outputs = []
    aspect_names = ['price', 'shipping', 'outlook', 'quality', 'size', 'shop_service', 'general', 'others']
    
    for aspect in aspect_names:
        output = layers.Dense(num_classes, activation='softmax', name=f'output_{aspect}')(x)
        outputs.append(output)
    
    # Create model
    model = keras.Model(inputs=text_input, outputs=outputs, name='ABSA_Model')
    
    return model, vectorize_layer

# ==========================================
# TRAINING FUNCTION
# ==========================================
def train_model(data_path, output_path, epochs=5, batch_size=32):
    """
    Train ABSA model
    
    Args:
        data_path: Path to training CSV
        output_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        metrics: Dictionary of training metrics
    """
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Preprocess reviews
    print("Preprocessing texts...")
    df['processed_review'] = df['Review'].apply(preprocess_text)
    
    # Prepare features and labels
    X = df['processed_review'].values

    print("X : " ,  X[0:5])
    print("type X[0] : " , type(X[0]))
    
    # Map labels: -1->0, 0->1, 1->2, 2->3 (for categorical)
    y_aspects = []
    aspect_cols = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']
    
    for col in aspect_cols:
        y = df[col].values + 1  # Shift to 0-3 range
        y = keras.utils.to_categorical(y, num_classes=4)
        y_aspects.append(y)
    
    # Split data
    print("Splitting data...")
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train, X_val = X[train_idx], X[val_idx]

    # 1. BẮT BUỘC: Ép kiểu sang string để TextVectorization hoạt động
    X_train = np.array(X_train).astype(str)
    X_val = np.array(X_val).astype(str)

    # 2. BẮT BUỘC: Ép kiểu nhãn sang float32 để tránh lỗi "Invalid dtype: object" ở model.fit
    y_train = [y[train_idx].astype('float32') for y in y_aspects]
    y_val = [y[val_idx].astype('float32') for y in y_aspects]
    
    # Create model
    print("Creating model...")
    model, vectorize_layer = create_absa_model()
    
    # Adapt vectorization layer
    print("Adapting text vectorization...")
    vectorize_layer.adapt(X_train)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics={f'output_{i}': ['accuracy'] for i in ['price', 'shipping', 'outlook', 'quality', 'size', 'shop_service', 'general', 'others']}
    )
    
    print("Model summary:")

    print("Loss function : " , model.loss)
    print("Output name : ", model.output_names)


    model.summary()
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    val_results = model.evaluate(X_val, y_val, verbose=0)
    
    # Calculate per-aspect metrics
    predictions = model.predict(X_val, verbose=0)
    
    metrics = {
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'epochs': epochs,
        'final_loss': float(val_results[0]),
    }
    
    # Per-aspect accuracy
    for i, aspect in enumerate(aspect_cols):
        y_true = np.argmax(y_val[i], axis=1)
        y_pred = np.argmax(predictions[i], axis=1)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics[f'accuracy_{aspect.lower()}'] = float(acc)
        metrics[f'f1_score_{aspect.lower()}'] = float(f1)
    
    # Average metrics
    metrics['avg_accuracy'] = float(np.mean([metrics[f'accuracy_{aspect.lower()}'] for aspect in aspect_cols]))
    metrics['avg_f1_score'] = float(np.mean([metrics[f'f1_score_{aspect.lower()}'] for aspect in aspect_cols]))
    
    # Save model
    print(f"\nSaving model to {output_path}")
    model.save(output_path)
    
    print("\nTraining complete!")
    print(f"Average Accuracy: {metrics['avg_accuracy']:.4f}")
    print(f"Average F1-Score: {metrics['avg_f1_score']:.4f}")
    
    return metrics

# ==========================================
# SAVE TO DATABASE
# ==========================================
def save_model_version(version, model_path, metrics):
    """Save model version info to database"""
    try:
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            database=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD
        )
        cursor = conn.cursor()
        
        query = """
            INSERT INTO model_versions (
                version, model_path,
                accuracy_price, accuracy_shipping, accuracy_outlook, accuracy_quality,
                accuracy_size, accuracy_shop_service, accuracy_general, accuracy_others,
                avg_accuracy,
                f1_score_price, f1_score_shipping, f1_score_outlook, f1_score_quality,
                f1_score_size, f1_score_shop_service, f1_score_general, f1_score_others,
                avg_f1_score,
                is_production, notes
            ) VALUES (
                %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s
            )
        """
        
        cursor.execute(query, (
            version, model_path,
            metrics['accuracy_price'], metrics['accuracy_shipping'], 
            metrics['accuracy_outlook'], metrics['accuracy_quality'],
            metrics['accuracy_size'], metrics['accuracy_shop_service'],
            metrics['accuracy_general'], metrics['accuracy_others'],
            metrics['avg_accuracy'],
            metrics['f1_score_price'], metrics['f1_score_shipping'],
            metrics['f1_score_outlook'], metrics['f1_score_quality'],
            metrics['f1_score_size'], metrics['f1_score_shop_service'],
            metrics['f1_score_general'], metrics['f1_score_others'],
            metrics['avg_f1_score'],
            False,  # is_production
            f"Training metrics: {json.dumps(metrics, indent=2)}"
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Model version {version} saved to database")
        
    except Exception as e:
        print(f"Failed to save to database: {e}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ABSA Model")
    parser.add_argument('--data', type=str, default=Config.TRAIN_DATA_PATH,
                       help='Path to training data CSV')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for model')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Generate version and output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v_{timestamp}"
    
    if args.output is None:
        os.makedirs(Config.CANDIDATE_MODEL_DIR, exist_ok=True)
        output_path = os.path.join(Config.CANDIDATE_MODEL_DIR, f"model_{version}.keras")
    else:
        output_path = args.output
    
    print(f'output path: {output_path}')
    # Train model
    metrics = train_model(
        data_path=args.data,
        output_path=output_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save to database  
    save_model_version(version, output_path, metrics)
    
    print(f"\n✓ Training complete! Model saved as: {output_path}")
# python train_model.py --data /opt/project/archive/train_data.csv --epochs 1 --batch-size 32