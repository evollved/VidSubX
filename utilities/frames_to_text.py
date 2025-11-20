import ctypes
import logging
import site
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import onnxruntime as ort

ctypes.CDLL(f"{site.getsitepackages()[1]}/nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll")
ort.preload_dlls()

from custom_ocr import CustomPaddleOCR, TextDetection

import utilities.utils as utils

logging.getLogger("custom_ocr").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def setup_ocr() -> None:
    utils.CONFIG.ocr_opts["lang"] = utils.CONFIG.ocr_rec_language
    utils.CONFIG.ocr_opts["ocr_version"] = utils.CONFIG.paddleocr_version
    utils.CONFIG.ocr_opts["use_mobile_model"] = utils.CONFIG.use_mobile_model

    download_models()
    setup_ocr_device()


def setup_ocr_device() -> None:
    sess_opt = ort.SessionOptions()
    if utils.CONFIG.use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        utils.CONFIG.ocr_opts["use_gpu"] = True
        sess_opt.intra_op_num_threads = 0
    else:
        utils.CONFIG.ocr_opts["use_gpu"] = False
        sess_opt.intra_op_num_threads = 0
    utils.CONFIG.ocr_opts["onnx_sess_options"] = sess_opt


def download_models() -> None:
    """
    Download models if dir does not exist.
    """
    logger.info("Checking for requested models...")
    _ = CustomPaddleOCR(**utils.CONFIG.ocr_opts)
    logger.info("")


def extract_bboxes(files: Path) -> list:
    """
    Returns the bounding boxes of detected texted in images.
    :param files: Directory with images for detection.
    """
    model_name = f"{utils.CONFIG.paddleocr_version}_{'mobile' if utils.CONFIG.use_mobile_model else 'server'}_det"
    det_config = {"model_save_dir": utils.CONFIG.ocr_opts["model_save_dir"], "model_name": model_name,
                  "box_thresh": utils.CONFIG.bbox_drop_score}
    ocr_engine = TextDetection(**det_config)
    results = ocr_engine.predict_iter(str(files))
    boxes = [box for result in results for box in result["dt_polys"]]
    return boxes


def extract_text(ocr_engine, text_output: Path, files: list, line_sep: str) -> None:
    """
    Extract text from a frame using ocr.
    :param ocr_engine: OCR engine.
    :param text_output: directory for extracted texts.
    :param files: files with text for extraction.
    :param line_sep: line seperator for the text.
    """
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
    batch_size = utils.CONFIG.text_extraction_batch_size  # Size of files given to each processor.
    prefix, device = "Text Extraction", "GPU" if utils.CONFIG.ocr_opts["use_gpu"] else "CPU"
    line_sep = "\n" if utils.CONFIG.line_break else " "

    if utils.Process.interrupt_process:  # Cancel if process has been cancelled by gui.
        logger.warning(f"{prefix} process interrupted!")
        return

    ocr_config = {"text_rec_score_thresh": utils.CONFIG.text_drop_score} | utils.CONFIG.ocr_opts
    ocr_engine = CustomPaddleOCR(**ocr_config)
    files = list(frame_output.iterdir())
    file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    no_batches = len(file_batches)
    logger.info(f"Starting Multiprocess {prefix} from frames on {device}, Batches: {no_batches}.")
    with ThreadPoolExecutor(utils.CONFIG.ocr_max_processes) as executor:
        futures = [executor.submit(extract_text, ocr_engine, text_output, files, line_sep) for files in file_batches]
        for i, f in enumerate(as_completed(futures)):  # as each  process completes
            f.result()  # Prevents silent bugs. Exceptions raised will be displayed.
            utils.print_progress(i, no_batches - 1, prefix)
    logger.info(f"{prefix} done!")
