import sys
import pickle
import numpy as np
from types import ModuleType

# --- VÁ LỖI ĐỂ ĐỌC ĐƯỢC FILE CŨ ---
# Tạo môi trường giả lập Numpy 2.0 chỉ để lừa Pickle
try:
    import numpy._core
except ImportError:
    # 1. Tạo package giả _core
    mock_core = ModuleType("numpy._core")
    mock_core.__path__ = [] # Dòng này quan trọng: đánh dấu là package
    sys.modules["numpy._core"] = mock_core
    
    # 2. Map multiarray (cái mà pickle đang tìm)
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    mock_core.multiarray = np.core.multiarray

# --- ĐỌC VÀ LƯU LẠI ---
old_path = r"D:\Projects\Assignment\video-counting\src2\vocab.pkl"
new_path = r"D:\Projects\Assignment\video-counting\src2\vocab_clean.pkl"

print(f"Dang doc file: {old_path}")
with open(old_path, 'rb') as f:
    vocab_data = pickle.load(f)

# Nếu vocab_data là numpy array, chuyển về list thuần Python
if isinstance(vocab_data, np.ndarray):
    vocab_data = vocab_data.tolist()
elif isinstance(vocab_data, list):
    # Đảm bảo các phần tử bên trong không phải là numpy scalar
    vocab_data = [x.item() if hasattr(x, "item") else x for x in vocab_data]

print(f"Dang luu file sach: {new_path}")
with open(new_path, 'wb') as f:
    # protocol=4 tương thích tốt với mọi phiên bản Python
    pickle.dump(vocab_data, f, protocol=4) 

print("XONG! Hãy sửa code chính để dùng 'vocab_clean.pkl'")