"""
Script để đọc và hiển thị thông tin các file .pkl trong thư mục models/
"""

import pickle
import os
import sys
from pathlib import Path

# Import config để lấy đường dẫn models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR


def load_pickle_file(filepath):
    """
    Đọc file pickle và trả về object
    
    Args:
        filepath (str): Đường dẫn đến file .pkl
        
    Returns:
        object: Object được load từ file pickle
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"❌ Lỗi khi đọc file {filepath}: {e}")
        return None


def get_model_info(model, model_name):
    """
    Lấy thông tin về model
    
    Args:
        model: Model object
        model_name (str): Tên model
        
    Returns:
        dict: Dictionary chứa thông tin model
    """
    info = {
        'name': model_name,
        'type': type(model).__name__,
        'module': type(model).__module__
    }
    
    # Lấy thông tin đặc biệt cho từng loại model
    if hasattr(model, 'get_params'):
        try:
            info['parameters'] = model.get_params()
        except:
            info['parameters'] = 'Không thể lấy parameters'
    
    if hasattr(model, 'feature_importances_'):
        info['has_feature_importances'] = True
        info['n_features'] = len(model.feature_importances_)
    else:
        info['has_feature_importances'] = False
    
    if hasattr(model, 'coef_'):
        info['has_coefficients'] = True
        info['n_coefficients'] = len(model.coef_) if hasattr(model.coef_, '__len__') else 1
    else:
        info['has_coefficients'] = False
    
    return info


def display_model_info(info):
    """
    Hiển thị thông tin model một cách đẹp mắt
    
    Args:
        info (dict): Dictionary chứa thông tin model
    """
    print("\n" + "="*80)
    print(f"📦 MODEL: {info['name']}")
    print("="*80)
    print(f"  Loại model: {info['type']}")
    print(f"  Module: {info['module']}")
    
    if 'n_features' in info:
        print(f"  Số features: {info['n_features']}")
    
    if 'n_coefficients' in info:
        print(f"  Số coefficients: {info['n_coefficients']}")
    
    if 'parameters' in info and isinstance(info['parameters'], dict):
        print(f"\n  Parameters:")
        for key, value in list(info['parameters'].items())[:10]:  # Hiển thị 10 params đầu
            if isinstance(value, (int, float, str, bool, type(None))):
                print(f"    - {key}: {value}")
            else:
                print(f"    - {key}: {type(value).__name__}")
        if len(info['parameters']) > 10:
            print(f"    ... và {len(info['parameters']) - 10} parameters khác")


def load_all_models(models_dir=None):
    """
    Đọc tất cả các file .pkl trong thư mục models/
    
    Args:
        models_dir (str): Đường dẫn đến thư mục models. Nếu None thì dùng MODELS_DIR từ config
        
    Returns:
        dict: Dictionary chứa tất cả các models đã load
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    print("="*80)
    print("ĐỌC CÁC FILE .PKL TRONG THƯ MỤC MODELS/")
    print("="*80)
    print(f"\n📁 Thư mục: {models_dir}")
    
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(models_dir):
        print(f"\n❌ Thư mục không tồn tại: {models_dir}")
        return {}
    
    # Tìm tất cả file .pkl
    pkl_files = list(Path(models_dir).glob('*.pkl'))
    
    if not pkl_files:
        print(f"\n⚠️ Không tìm thấy file .pkl nào trong {models_dir}")
        return {}
    
    print(f"\n✓ Tìm thấy {len(pkl_files)} file .pkl:")
    for pkl_file in pkl_files:
        print(f"  - {pkl_file.name}")
    
    # Load tất cả models
    models = {}
    models_info = []
    
    print("\n" + "-"*80)
    print("ĐANG ĐỌC CÁC FILE...")
    print("-"*80)
    
    for pkl_file in pkl_files:
        model_name = pkl_file.stem  # Tên file không có extension
        filepath = str(pkl_file)
        
        print(f"\n📖 Đang đọc: {pkl_file.name}...")
        model = load_pickle_file(filepath)
        
        if model is not None:
            models[model_name] = model
            info = get_model_info(model, model_name)
            models_info.append(info)
            display_model_info(info)
            print(f"  ✓ Đã load thành công!")
        else:
            print(f"  ❌ Không thể load model")
    
    # Tóm tắt
    print("\n" + "="*80)
    print("TÓM TẮT")
    print("="*80)
    print(f"✓ Đã load thành công {len(models)}/{len(pkl_files)} models")
    print(f"\nDanh sách models đã load:")
    for name in models.keys():
        print(f"  - {name}")
    
    return models


def load_specific_model(model_filename, models_dir=None):
    """
    Đọc một file .pkl cụ thể
    
    Args:
        model_filename (str): Tên file .pkl (có thể có hoặc không có extension)
        models_dir (str): Đường dẫn đến thư mục models. Nếu None thì dùng MODELS_DIR từ config
        
    Returns:
        object: Model object hoặc None nếu lỗi
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    # Đảm bảo có extension .pkl
    if not model_filename.endswith('.pkl'):
        model_filename += '.pkl'
    
    filepath = os.path.join(models_dir, model_filename)
    
    if not os.path.exists(filepath):
        print(f"❌ Không tìm thấy file: {filepath}")
        return None
    
    print(f"📖 Đang đọc: {model_filename}...")
    model = load_pickle_file(filepath)
    
    if model is not None:
        info = get_model_info(model, model_filename)
        display_model_info(info)
        print(f"✓ Đã load thành công!")
    
    return model


def main():
    """Hàm chính"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Đọc các file .pkl trong thư mục models/')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Tên file model cụ thể cần đọc (ví dụ: baseline_linear_regression_model)')
    parser.add_argument('--dir', '-d', type=str, default=None,
                        help='Đường dẫn đến thư mục models (mặc định: dùng MODELS_DIR từ config)')
    
    args = parser.parse_args()
    
    if args.model:
        # Đọc một model cụ thể
        model = load_specific_model(args.model, args.dir)
        return model
    else:
        # Đọc tất cả models
        models = load_all_models(args.dir)
        return models


if __name__ == "__main__":
    # Nếu chạy trực tiếp không có arguments, load tất cả
    if len(sys.argv) == 1:
        models = load_all_models()
    else:
        models = main()

