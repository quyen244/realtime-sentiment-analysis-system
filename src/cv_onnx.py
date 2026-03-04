import tensorflow as tf
import tf2onnx
import onnx
import builtins
import functools
import os 
import sys 

# Lưu lại hàm open gốc của Python
_original_open = builtins.open

def patched_open(file, mode='r', *args, **kwargs):
    # Nếu mở ở chế độ text (không có 'b' trong mode) và chưa có encoding
    if 'b' not in mode and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return _original_open(file, mode, *args, **kwargs)

# Ghi đè hàm open bằng phiên bản thông minh hơn
builtins.open = patched_open

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
print(sys.getdefaultencoding())
os.environ["PYTHONUTF8"] = "1"

# 1. Load model (Bạn đã có đoạn này)
model_path = r'D:\Projects\Assignment\video-counting\models\vietnamese_sentiment_model (1).keras'
model = tf.keras.models.load_model(model_path, compile=False)

if __name__ == "__main__":
    # ... (Phần tách model và lưu file giữ nguyên) ...

    # Load lại để test
    vectorizer = tf.keras.models.load_model("vectorizer_part.keras", compile=False)
    inference_model = tf.keras.models.load_model("inference_model.keras", compile=False)

    # FIX LỖI 1: Chuyển đầu vào thành Tensor
    text_input = tf.constant(["Sản phẩm tuyệt vời"])
    vectorized_text = vectorizer(text_input) 
    print("Vectorized text shape:", vectorized_text.shape)

    # FIX LỖI 2: Định nghĩa spec cho Inference Model (Input là INT, không phải STRING)
    # Shape là (None, 750) vì đầu ra của vectorizer là 750 số nguyên
    inference_spec = [tf.TensorSpec([None, 750], tf.int64, name='input_ids')]

    # Convert Inference Model sang ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(
        inference_model,
        input_signature=inference_spec,
        opset=15  
    )

    # Save the ONNX model
    onnx.save(onnx_model, "absa_model.onnx")
    print("✅ Keras model successfully converted to absa_model.onnx")

    # --- TEST ONNX ---
    import onnxruntime as ort
    session = ort.InferenceSession("absa_model.onnx")

    # Lấy tên input chính xác từ ONNX (để tránh lỗi Hardcode tên)
    input_name = session.get_inputs()[0].name
    
    # Chuyển vectorized_text sang numpy để đưa vào ONNX
    outputs = session.run(None, {input_name: vectorized_text.numpy()})

    print("ONNX model outputs:", outputs)