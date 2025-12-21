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

    def __init__(self) -> None:
        self.cpu_ocr_processes = None
        self.values = ("cpu_ocr_processes", "cpu_onnx_intra_threads", "gpu_ocr_processes", "gpu_onnx_intra_threads")
        for val in self.values:
            setattr(self, val, getattr(utils.CONFIG, val))

        psutil.cpu_percent()  # Prime cpu_percent (important!)
        self.cpu_percentages, self.gpu_percentages, self.gpu_mem_usages = [], [], []

    def _get_cpu_usage(self) -> None:
        usage = psutil.cpu_percent()
        self.cpu_percentages.append(usage)

    def _get_gpu_usage(self) -> None:
        pass

    def record_perf(self) -> None:
        self._get_cpu_usage()
        self._get_gpu_usage()

    def _optimize_cpu_usage(self) -> None:
        self.cpu_percentages.pop()  # Remove the cpu percentage from the last batch
        cpu_util = sum(self.cpu_percentages) / len(self.cpu_percentages)
        usage = f"Average CPU Usage: {cpu_util:.2f}%"
        logger.debug(f"{self.cpu_percentages=}")
        if cpu_util < 80 and self.cpu_ocr_processes < utils.get_physical_cores():
            logger.info(f"{usage}. Increasing cores used!")
            self.cpu_ocr_processes += 1
        elif cpu_util > 95:
            logger.warning(f"{usage}. Decreasing cores used!")
            self.cpu_ocr_processes -= 1
        else:
            logger.debug(usage)

    def _optimize_gpu_usage(self) -> None:
        pass

    def _changes_made(self) -> bool:
        return any(getattr(self, val) != getattr(utils.CONFIG, val) for val in self.values)

    def _save_perf_optimisations(self) -> None:
        if not self._changes_made():
            return
        logger.debug("Saving performance optimisations")
        utils.CONFIG.set_config(**{val: getattr(self, val) for val in self.values})

    def optimise_performance(self) -> None:
        self._optimize_cpu_usage()
        self._optimize_gpu_usage()
        self._save_perf_optimisations()
