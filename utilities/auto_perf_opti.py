import logging

import psutil

import utilities.utils as utils

logger = logging.getLogger(__name__)


class NullPerformanceOptimiser:
    def record_perf(self) -> None:
        pass

    def optimise_performance(self) -> None:
        pass


class PerformanceOptimiser:

    def __init__(self, cpu_min: int = 80, cpu_max: int = 95) -> None:
        self.cpu_min, self.cpu_max = cpu_min, cpu_max
        self.cpu_ocr_processes = utils.CONFIG.cpu_ocr_processes
        self.use_gpu = utils.CONFIG.ocr_opts["use_gpu"]
        if self.use_gpu:
            pass
        else:
            psutil.cpu_percent()  # Prime cpu_percent (important!)
        self.percentages = []

    def _get_cpu_usage(self) -> None:
        usage = psutil.cpu_percent()
        self.percentages.append(usage)

    def _get_gpu_usage(self, gpu_id: int = 0) -> None:
        pass

    def record_perf(self) -> None:
        if self.use_gpu:
            self._get_gpu_usage()
        else:
            self._get_cpu_usage()

    def _optimize_cpu_usage(self) -> None:
        logger.debug("Optimizing CPU usage...")
        self.percentages.pop()  # Remove the cpu percentage from the last batch
        cpu_util = sum(self.percentages) / len(self.percentages)
        usage = f"Average CPU Usage: {cpu_util:.2f}%."
        if cpu_util < self.cpu_min and self.cpu_ocr_processes < utils.get_physical_cores():
            logger.info(f"{usage} Increasing cores used!")
            self.cpu_ocr_processes += 1
        elif cpu_util > self.cpu_max:
            logger.warning(f"{usage} Decreasing cores used!")
            self.cpu_ocr_processes -= 1
        else:
            logger.debug(usage)

    def _optimize_gpu_usage(self) -> None:
        pass

    def _changes_made(self) -> bool:
        return utils.CONFIG.cpu_ocr_processes != self.cpu_ocr_processes

    def _save_perf_optimisations(self) -> None:
        if not self._changes_made():
            return
        logger.debug("Saving performance optimisations!")
        utils.CONFIG.set_config(cpu_ocr_processes=self.cpu_ocr_processes)

    def optimise_performance(self) -> None:
        if self.use_gpu:
            self._optimize_gpu_usage()
        else:
            self._optimize_cpu_usage()
        self._save_perf_optimisations()
