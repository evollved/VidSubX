import logging
from typing import Dict, List, Optional
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

logger = logging.getLogger(__name__)


class GPUDetector:
    @staticmethod
    def detect_available_providers() -> Dict[str, bool]:
        """
        Определяет доступные GPU провайдеры
        """
        providers = {
            "cuda": False,
            "directml": False,
            "openvino": False,
            "cpu": True
        }
        
        if not HAS_ORT:
            return providers
            
        try:
            available = ort.get_available_providers()
            providers["cuda"] = "CUDAExecutionProvider" in available
            providers["directml"] = "DmlExecutionProvider" in available
            providers["openvino"] = "OpenVINOExecutionProvider" in available
        except Exception as e:
            logger.error(f"Error detecting providers: {e}")
            
        return providers
    
    @staticmethod
    def get_recommended_provider() -> str:
        """
        Возвращает рекомендуемый провайдер на основе доступности
        """
        providers = GPUDetector.detect_available_providers()
        
        # Приоритет: CUDA > DirectML > OpenVINO > CPU
        if providers["cuda"]:
            return "cuda"
        elif providers["directml"]:
            return "directml"
        elif providers["openvino"]:
            return "openvino"
        else:
            return "cpu"
    
    @staticmethod
    def get_gpu_info() -> List[Dict]:
        """
        Возвращает информацию о доступных GPU
        """
        gpus = []
        try:
            import GPUtil
            gpu_list = GPUtil.getGPUs()
            for gpu in gpu_list:
                gpus.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature,
                    "driver": gpu.driver
                })
        except ImportError:
            logger.debug("GPUtil not installed, GPU info unavailable")
        except Exception as e:
            logger.debug(f"Could not get GPU info: {e}")
            
        return gpus
    
    @staticmethod
    def get_system_gpu_info() -> Dict:
        """
        Полная информация о GPU системе
        """
        info = {
            "available_providers": GPUDetector.detect_available_providers(),
            "recommended_provider": GPUDetector.get_recommended_provider(),
            "gpus": GPUDetector.get_gpu_info(),
            "has_gpu": False
        }
        
        # Определяем, есть ли вообще GPU
        gpu_providers = ["cuda", "directml", "openvino"]
        info["has_gpu"] = any(info["available_providers"][provider] for provider in gpu_providers)
        
        return info


def detect_and_log_gpu_info() -> None:
    """
    Обнаруживает информацию о GPU и логирует её
    """
    gpu_info = GPUDetector.get_system_gpu_info()
    
    logger.info("=" * 50)
    logger.info("GPU SYSTEM INFORMATION")
    logger.info("=" * 50)
    
    # Информация о доступных провайдерах
    providers = gpu_info["available_providers"]
    logger.info("Available ONNX Runtime Providers:")
    for provider, available in providers.items():
        status = "✓ Available" if available else "✗ Not Available"
        logger.info(f"  - {provider.upper()}: {status}")
    
    # Информация о физических GPU
    gpus = gpu_info["gpus"]
    if gpus:
        logger.info(f"\nDetected {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu['name']}")
            logger.info(f"    Memory: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB ({gpu['load']:.1f}% load)")
            if gpu.get('temperature'):
                logger.info(f"    Temperature: {gpu['temperature']}°C")
    else:
        logger.info("\nNo GPU detected or GPUtil not installed")
    
    # Рекомендованный провайдер
    logger.info(f"\nRecommended GPU Provider: {gpu_info['recommended_provider'].upper()}")
    logger.info("=" * 50)