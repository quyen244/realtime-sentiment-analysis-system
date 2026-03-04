"""
Model Evaluation Script for ABSA System
Compares new candidate model with current production model
"""

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
import psycopg2
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import Config

# ==========================================
# TEXT PREPROCESSING
# ==========================================
def preprocess_text(text):
    """Same preprocessing as in consumer.py and train_model.py"""
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
# EVALUATION FUNCTIONS
# ==========================================
def evaluate_model(model_path, test_data_path):
    """
    Evaluate a model on test data
    
    Returns:
        dict: Metrics for each aspect and overall
    """
    print(f"Evaluating model: {model_path}")
    print(f"Test data: {test_data_path}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load test data
    df = pd.read_csv(test_data_path)
    
    # Preprocess
    X_test = df['Review'].apply(preprocess_text).values
    X_test = np.array(X_test).astype(str)
    
    # Prepare labels
    aspect_cols = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']
    y_test = [df[col].values + 1 for col in aspect_cols]  
    
    # Predict
    predictions = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    metrics = {}
    
    for i, aspect in enumerate(aspect_cols):
        y_true = y_test[i]
        y_pred = np.argmax(predictions[i], axis=1)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics[f'accuracy_{aspect.lower()}'] = float(acc)
        metrics[f'f1_score_{aspect.lower()}'] = float(f1)
        
        print(f"{aspect:15} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # Average metrics
    metrics['avg_accuracy'] = float(np.mean([metrics[f'accuracy_{aspect.lower()}'] for aspect in aspect_cols]))
    metrics['avg_f1_score'] = float(np.mean([metrics[f'f1_score_{aspect.lower()}'] for aspect in aspect_cols]))
    
    print(f"\nAverage Accuracy: {metrics['avg_accuracy']:.4f}")
    print(f"Average F1-Score: {metrics['avg_f1_score']:.4f}")
    
    return metrics

def get_production_model():
    """Get current production model path from database"""
    try:
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            database=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD
        )
        cursor = conn.cursor()
        
        query = "SELECT model_path FROM model_versions WHERE is_production = TRUE LIMIT 1"
        cursor.execute(query)
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            return result[0]
        else:
            return None
            
    except Exception as e:
        print(f"Error getting production model: {e}")
        return None

def compare_models(candidate_path, production_path, test_data_path):
    """
    Compare candidate model with production model
    
    Returns:
        bool: True if candidate is better, False otherwise
    """
    print("="*60)
    print("PRODUCTION MODEL EVALUATION")
    print("="*60)
    prod_metrics = evaluate_model(production_path, test_data_path)
    
    print("\n" + "="*60)
    print("CANDIDATE MODEL EVALUATION")
    print("="*60)
    cand_metrics = evaluate_model(candidate_path, test_data_path)
    
    if prod_metrics is None or cand_metrics is None:
        return False
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"Production Avg Accuracy: {prod_metrics['avg_accuracy']:.4f}")
    print(f"Candidate  Avg Accuracy: {cand_metrics['avg_accuracy']:.4f}")
    print(f"Difference: {cand_metrics['avg_accuracy'] - prod_metrics['avg_accuracy']:+.4f}")
    
    print(f"\nProduction Avg F1-Score: {prod_metrics['avg_f1_score']:.4f}")
    print(f"Candidate  Avg F1-Score: {cand_metrics['avg_f1_score']:.4f}")
    print(f"Difference: {cand_metrics['avg_f1_score'] - prod_metrics['avg_f1_score']:+.4f}")
    
    # Decision: candidate must be better in both accuracy and F1
    is_better = (
        cand_metrics['avg_accuracy'] > prod_metrics['avg_accuracy'] and
        cand_metrics['avg_f1_score'] > prod_metrics['avg_f1_score']
    )
    
    print("\n" + "="*60)
    if is_better:
        print("✓ DECISION: Candidate model is BETTER - Recommend deployment")
    else:
        print("✗ DECISION: Production model is better - Keep current model")
    print("="*60)
    
    return is_better

def update_production_model(new_model_path):
    """Update production model in database"""

    print('Updating production model...')
    print(f'New model path: {new_model_path}')

    try:
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            database=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD
        )
        cursor = conn.cursor()
        
        # Set all models to non-production
        cursor.execute("UPDATE model_versions SET is_production = false")
        
        # Set new model as production
        cursor.execute(
            "UPDATE model_versions SET is_production = true WHERE model_path = %s",
            (new_model_path,)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"\n✓ Production model updated: {new_model_path}")
        
        return True
        
    except Exception as e:
        print(f"Error updating production model: {e}")
        return False

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and Compare ABSA Models")
    parser.add_argument('--candidate', type=str, required=True,
                       help='Path to candidate model')
    parser.add_argument('--test-data', type=str, default=Config.TEST_DATA_PATH,
                       help='Path to test data CSV')
    parser.add_argument('--auto-deploy', action='store_true',
                       help='Automatically deploy if candidate is better')
    
    args = parser.parse_args()
    
    # Get production model
    production_path = get_production_model()

    print("\n" + "="*60)
    print(f"production model path : {production_path}")
    print(f"candidate model path  : {args.candidate}")
    
    if production_path is None:
        print("No production model found. Evaluating candidate only...")
        metrics = evaluate_model(args.candidate, args.test_data)
        if args.auto_deploy:
            print("\nNo production model exists. Deploying candidate as production...")
            update_production_model(args.candidate)
    else:
        # Compare models
        is_better = compare_models(args.candidate, production_path, args.test_data)
        
        if is_better and args.auto_deploy:
            print("\nDeploying new model...")
            update_production_model(args.candidate)
        elif is_better:
            print("\nCandidate is better but auto-deploy is disabled.")
            print("Run with --auto-deploy to deploy automatically.")
        else:
            print("\nKeeping current production model.")
    
    print("\n✓ Evaluation complete!")
# python evaluate_model.py --candidate /opt/project/models/candidates/model_v_20260208_110915.keras --test-data /opt/project/archive/test_data.csv --auto-deploy