import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import utilities.utils as utils

os.environ["GLOG_minloglevel"] = "2"  # Supress InitGoogleLogging logs
warnings.filterwarnings("ignore", "No ccache found. Please be aware that recompiling", UserWarning)
warnings.filterwarnings("ignore", "Value do not have 'place' interface for pir graph mode", UserWarning)

import paddle
from .paddleocr import PaddleOCR, TextDetection

logging.getLogger("paddlex").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Supress logs when downloading models

logger = logging.getLogger(__name__)


def setup_ocr() -> None:
    utils.Config.ocr_opts["lang"] = utils.Config.ocr_rec_language

    setup_ocr_device()
    download_models()


def setup_ocr_device() -> None:
    if utils.Config.use_gpu and paddle.device.cuda.device_count() > 0:
        logger.debug("GPU is enabled.")
        utils.Config.ocr_opts.pop("device", None)

    else:
        logger.debug("GPU is disabled.")
        utils.Config.ocr_opts["device"] = "cpu"


def download_models() -> None:
    """
    Download models if dir does not exist.
    """
    logger.info("Checking for requested models...")
    _ = PaddleOCR(**utils.Config.ocr_opts)
    logger.info("")


def extract_bboxes(files: Path) -> list:
    """
    Returns the bounding boxes of detected texted in images.
    :param files: Directory with images for detection.
    """
    ocr_engine = TextDetection(box_thresh=utils.Config.bbox_drop_score)
    results = ocr_engine.predict_iter([str(file) for file in files.iterdir()])
    boxes = [box for result in results for box in result["dt_polys"]]
    return boxes


def extract_text(ocr_config: dict, text_output: Path, files: list, line_sep: str) -> None:
    """
    Extract text from a frame using ocr.
    :param ocr_config: OCR engine configuration.
    :param text_output: directory for extracted texts.
    :param files: files with text for extraction.
    :param line_sep: line seperator for the text.
    """
    ocr_engine = PaddleOCR(**ocr_config)
    for file in files:
        result = ocr_engine.predict(str(file))
        text = line_sep.join(result[0]["rec_texts"])
        with open(f"{text_output}/{file.stem}.txt", 'w', encoding="utf-8") as text_file:
            text_file.write(text)


def frames_to_text(frame_output: Path, text_output: Path) -> None:
    """
    Extracts the texts from frames using multiprocessing.
    :param frame_output: directory of the frames
    :param text_output: directory for extracted texts
    """
    batch_size = utils.Config.text_extraction_batch_size  # Size of files given to each processor.
    prefix, device = "Text Extraction", utils.Config.ocr_opts.get("device", "gpu").upper()
    max_processes = utils.Config.ocr_cpu_max_processes if device == "CPU" else utils.Config.ocr_gpu_max_processes
    line_sep = "\n" if utils.Config.line_break else " "

    if utils.Process.interrupt_process:  # Cancel if process has been cancelled by gui.
        logger.warning(f"{prefix} process interrupted!")
        return

    ocr_config = {"text_rec_score_thresh": utils.Config.text_drop_score} | utils.Config.ocr_opts
    files = list(frame_output.iterdir())
    file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    no_batches = len(file_batches)
    logger.info(f"Starting Multiprocess {prefix} from frames on {device}, Batches: {no_batches}.")
    with ProcessPoolExecutor(max_processes) as executor:
        futures = [executor.submit(extract_text, ocr_config, text_output, files, line_sep) for files in file_batches]
        for i, f in enumerate(as_completed(futures)):  # as each  process completes
            f.result()  # Prevents silent bugs. Exceptions raised will be displayed.
            utils.print_progress(i, no_batches - 1, prefix)
    logger.info(f"{prefix} done!")
