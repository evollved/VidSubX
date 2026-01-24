import logging
import os
import platform
import sys
from configparser import ConfigParser
from functools import cache
from pathlib import Path

import psutil
import requests
from packaging.version import Version

logger = logging.getLogger(__name__)


class Process:
    interrupt_process = False

    @classmethod
    def start_process(cls) -> None:
        """
        Allows process to run.
        """
        cls.interrupt_process = False
        logger.debug(f"interrupt_process set to: {cls.interrupt_process}")

    @classmethod
    def stop_process(cls) -> None:
        """
        Stops process from running.
        """
        cls.interrupt_process = True
        logger.debug(f"interrupt_process set to: {cls.interrupt_process}")


def check_for_updates() -> None:
    """
    Checks GitHub for a new release.
    """
    try:
        response = requests.get(f"https://api.github.com/repos/voun7/{Config.program_name}/releases/latest")
        if response.status_code == 200:
            data = response.json()
            latest_version = Version(data["tag_name"])
            current_version = Version(Config.version_file.read_text())
            if latest_version > current_version:
                logger.info(f"Version {latest_version} is now available.\nLink: {data['html_url']}")
            else:
                logger.debug("No new updates available.")
    except Exception as error:
        logger.info("Failed to check for updates!")
        logger.debug(error)
        return


@cache
def get_physical_cores() -> int:
    return psutil.cpu_count(logical=False)


def get_config_dir() -> Path:
    config_dir = Path(__file__).parent.parent
    if getattr(sys, "frozen", False):  # config dir will be in working dir if not compiled
        operating_system = platform.system()
        if operating_system == "Windows":
            config_dir = Path(os.getenv("APPDATA"), Config.program_name)
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_log_dir() -> Path:
    log_dir = Path(__file__).parent.parent
    if getattr(sys, "frozen", False):  # log dir will be in working dir if not compiled
        operating_system = platform.system()
        if operating_system == "Windows":
            log_dir = Path(os.getenv("LOCALAPPDATA"), Config.program_name)
    log_dir.mkdir(exist_ok=True)
    return log_dir


class Config:
    program_name = "VidSubX"
    version_file = Path(__file__).parent.parent / "installer/version.txt"
    physical_cores = get_physical_cores()
    config_schema = {
        "Frame Extraction": {
            "frame_extraction_frequency": (int, 2),
            "frame_extraction_batch_size": (int, 250),
        },
        "Text Extraction": {
            "text_extraction_batch_size": (int, 100),
            "text_drop_score": (float, 0.6),
            "line_break": (bool, True),
        },
        "Subtitle Generator": {
            "text_similarity_threshold": (float, 0.85),
            "min_consecutive_sub_dur_ms": (float, 500.0),
            "max_consecutive_short_durs": (int, 4),
            "min_sub_duration_ms": (float, 120.0),
        },
        "Subtitle Detection": {
            "split_start": (float, 0.25),
            "split_stop": (float, 0.50),
            "no_of_frames": (int, 200),
            "sub_area_x_rel_padding": (float, 0.9),
            "sub_area_y_abs_padding": (int, 25),
            "bbox_drop_score": (float, 0.7),
            "use_search_area": (bool, True),
        },
        "Notification": {
            "win_notify_sound": (str, "Default"),
            "win_notify_loop_sound": (bool, True),
        },
        "OCR Engine": {
            "ocr_rec_language": (str, "ch"),
            "use_mobile_model": (bool, True),
            "use_text_ori": (bool, False),
        },
        "OCR Performance": {
            "cpu_ocr_processes": (int, physical_cores // 2),
            "cpu_onnx_intra_threads": (int, max(4, int(physical_cores / 2.5))),
            "gpu_ocr_processes": (int, max(2, physical_cores // 3)),
            "use_gpu": (bool, True),
            "auto_optimize_perf": (bool, True),
        },
    }

    def __init__(self) -> None:
        # Permanent values
        self.subarea_height_scaler = 0.75
        self.model_dir = Path(__file__).parent.parent / "models"
        self.ocr_opts = {"model_save_dir": str(self.model_dir)}

        self.config_file = get_config_dir() / "config.ini"
        self.config = ConfigParser()
        if not self.config_file.exists():
            self.create_default_config_file()

        self.config.read(self.config_file)
        self.load_config()

    def create_default_config_file(self) -> None:
        for section, data in self.config_schema.items():
            self.config[section] = {k: str(default) for k, (_, default) in data.items()}

        with open(self.config_file, "w") as f:
            self.config.write(f)

    @staticmethod
    def _convert(value: str, typ):
        if typ is bool:
            return value.lower() in ("1", "true", "yes", "on")
        return typ(value)

    def load_config(self) -> None:
        for section, data in self.config_schema.items():
            for key, (typ, default) in data.items():
                raw = self.config[section].get(key, str(default))
                setattr(self, key, self._convert(raw, typ))

    def set_config(self, **kwargs) -> None:
        for key, val in kwargs.items():
            for section, data in self.config_schema.items():
                if key in data:
                    setattr(self, key, val)
                    self.config[section][key] = str(val)
                    break

        with open(self.config_file, "w") as f:
            self.config.write(f)


def print_progress(iteration: int, total: int, prefix: str = '', suffix: str = 'Complete', decimals: int = 3,
                   bar_length: int = 25) -> None:
    """
    Call in a loop to create standard out progress bar.
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: a prefix string to be printed in progress bar
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    """
    if total == 0:
        return

    percent = f"{(iteration / total) * 100:.{decimals}f}"
    filled = int(bar_length * iteration / total)
    bar = '#' * filled + '-' * (bar_length - filled)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end='', flush=True)  # prints progress on the same line

    if iteration >= total:
        print()


CONFIG = Config()
