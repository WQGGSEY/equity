import yaml
import importlib
from pathlib import Path

def load_config(config_path):
    """YAML 파일을 읽어서 딕셔너리로 반환"""
    path = Path(config_path)
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # base_config가 있다면 읽어서 병합 (간단한 상속 구현)
    if 'base_config' in config:
        base_path = path.parent.parent / Path(config['base_config']).name
        # (경로는 프로젝트 구조에 따라 유연하게 조정 필요)
        # 여기서는 간단히 configs/base.yaml을 찾는다고 가정
        if not base_path.exists():
            base_path = Path("configs/base.yaml")
            
        with open(base_path, 'r', encoding='utf-8') as bf:
            base_config = yaml.safe_load(bf)
            # Base에 현재 설정을 덮어씌움 (Merge)
            base_config.update(config)
            return base_config
            
    return config

def get_strategy_class(module_path, class_name):
    """문자열로 된 모듈/클래스 경로를 실제 클래스 객체로 변환"""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load strategy {class_name} from {module_path}: {e}")