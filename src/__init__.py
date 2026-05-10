from .dataset import DatasetGenerator
from .utils import draw_detections, load_config, save_results


def __getattr__(name):
    if name == "YOLODetector":
        from .detector import YOLODetector
        return YOLODetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["YOLODetector", "DatasetGenerator", "draw_detections", "load_config", "save_results"]
