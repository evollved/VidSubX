import logging
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

import psutil

import utilities.utils as utils

logger = logging.getLogger(__name__)


class NullPerformanceOptimiser:
    def record_perf(self) -> None:
        pass

    def optimise_performance(self) -> None:
        pass


class PerformanceOptimiser:

    def __init__(self, cpu_min: int = 80, cpu_max: int = 95, 
                 gpu_min: int = 70, gpu_max: int = 90) -> None:
        self.cpu_min, self.cpu_max = cpu_min, cpu_max
        self.gpu_min, self.gpu_max = gpu_min, gpu_max
        
        self.cpu_ocr_processes = utils.CONFIG.cpu_ocr_processes
        self.gpu_ocr_processes = utils.CONFIG.gpu_ocr_processes
        self.use_gpu = utils.CONFIG.ocr_opts["use_gpu"]
        self.gpu_provider = utils.CONFIG.gpu_provider.lower()
        
        if not self.use_gpu:
            psutil.cpu_percent()  # Prime cpu_percent (important!)
        self.cpu_percentages = []
        self.gpu_percentages = []

    def _get_cpu_usage(self) -> None:
        usage = psutil.cpu_percent()
        self.cpu_percentages.append(usage)

    def _get_gpu_usage(self, gpu_id: int = 0) -> None:
        if not HAS_GPUTIL:
            return
            
        try:
            gpus = GPUtil.getGPUs()
            if gpus and len(gpus) > gpu_id:
                gpu_usage = gpus[gpu_id].load * 100
                self.gpu_percentages.append(gpu_usage)
        except Exception as e:
            logger.debug(f"Could not get GPU usage: {e}")

    def record_perf(self) -> None:
        if self.use_gpu and self.gpu_provider in ["cuda", "directml"]:
            self._get_gpu_usage()
        else:
            self._get_cpu_usage()

    def _optimize_cpu_usage(self) -> None:
        logger.debug("Optimizing CPU usage...")
        if not self.cpu_percentages:
            return
            
        cpu_util = sum(self.cpu_percentages) / len(self.cpu_percentages)
        usage = f"Average CPU Usage: {cpu_util:.2f}%."
        
        if cpu_util < self.cpu_min and self.cpu_ocr_processes < utils.get_physical_cores():
            logger.info(f"{usage} Increasing CPU cores used!")
            self.cpu_ocr_processes += 1
        elif cpu_util > self.cpu_max and self.cpu_ocr_processes > 1:
            logger.warning(f"{usage} Decreasing CPU cores used!")
            self.cpu_ocr_processes -= 1
        else:
            logger.debug(usage)

    def _optimize_gpu_usage(self) -> None:
        logger.debug("Optimizing GPU usage...")
        if not self.gpu_percentages:
            return
            
        gpu_util = sum(self.gpu_percentages) / len(self.gpu_percentages)
        usage = f"Average GPU Usage: {gpu_util:.2f}%."
        
        if gpu_util < self.gpu_min and self.gpu_ocr_processes < utils.CONFIG.physical_cores:
            logger.info(f"{usage} Increasing GPU processes!")
            self.gpu_ocr_processes += 1
        elif gpu_util > self.gpu_max and self.gpu_ocr_processes > 1:
            logger.warning(f"{usage} Decreasing GPU processes!")
            self.gpu_ocr_processes -= 1
        else:
            logger.debug(usage)

    def _changes_made(self) -> bool:
        cpu_changed = utils.CONFIG.cpu_ocr_processes != self.cpu_ocr_processes
        gpu_changed = utils.CONFIG.gpu_ocr_processes != self.gpu_ocr_processes
        return cpu_changed or gpu_changed

    def _save_perf_optimisations(self) -> None:
        if not self._changes_made():
            return
            
        logger.debug("Saving performance optimisations!")
        updates = {}
        if utils.CONFIG.cpu_ocr_processes != self.cpu_ocr_processes:
            updates["cpu_ocr_processes"] = self.cpu_ocr_processes
        if utils.CONFIG.gpu_ocr_processes != self.gpu_ocr_processes:
            updates["gpu_ocr_processes"] = self.gpu_ocr_processes
            
        utils.CONFIG.set_config(**updates)

    def optimise_performance(self) -> None:
        if self.use_gpu and self.gpu_provider in ["cuda", "directml"]:
            self._optimize_gpu_usage()
        else:
            self._optimize_cpu_usage()
        self._save_perf_optimisations()
