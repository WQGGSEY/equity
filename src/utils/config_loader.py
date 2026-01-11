import yaml
import importlib
import collections.abc
from pathlib import Path

def deep_update(d, u):
    """딕셔너리를 재귀적으로 병합하는 함수 (Deep Merge)"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_config(config_path):
    """YAML 파일을 읽어서 딕셔너리로 반환 (상속 및 Deep Merge 지원)"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # base_config 상속 로직
    if 'base_config' in config:
        base_rel_path = config.pop('base_config') # 키 제거 후 병합
        
        # 1. 같은 폴더에서 찾기
        base_path = path.parent / Path(base_rel_path).name
        # 2. 없으면 지정된 경로(configs/base.yaml) 사용
        if not base_path.exists():
             base_path = Path(base_rel_path)
        
        if base_path.exists():
            with open(base_path, 'r', encoding='utf-8') as bf:
                base_config = yaml.safe_load(bf) or {}
                # Base 위에 현재 Config를 덮어씀 (Deep Merge)
                config = deep_update(base_config, config)
        else:
            print(f"⚠️ Warning: Base config not found at {base_path}")
            
    return config

def get_strategy_class(module_path, class_name):
    """문자열로 된 모듈/클래스 경로를 실제 클래스 객체로 변환"""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load strategy {class_name} from {module_path}: {e}")