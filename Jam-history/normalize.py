
from typing import Any, Dict, List


def normalize(obj: Any) -> Any:
    
    if isinstance(obj, list):
        return [normalize(item) for item in obj]
    
    if isinstance(obj, dict):
        normalized = {}
        for key, value in obj.items():
            if key == 'count':
                continue
            normalized[key] = normalize(value)
        return normalized
    
    if hasattr(obj, '__dict__'):
        obj_dict = obj.__dict__ if hasattr(obj, '__dict__') else obj
        if isinstance(obj_dict, dict):
            normalized = {}
            for key, value in obj_dict.items():
                if key == 'count':
                    continue
                normalized[key] = normalize(value)
            return normalized
    
    return obj
