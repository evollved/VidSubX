import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import onnxruntime as ort
from custom_ocr import CustomPaddleOCR, TextDetection

import utilities.utils as utils
from .auto_perf_opti import PerformanceOptimiser, NullPerformanceOptimiser

logging.getLogger("custom_ocr").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def setup_ocr() -> None:
    utils.CONFIG.ocr_opts["lang"] = utils.CONFIG.ocr_rec_language
    utils.CONFIG.ocr_opts["use_mobile_model"] = utils.CONFIG.use_mobile_model
    utils.CONFIG.ocr_opts["use_textline_orientation"] = utils.CONFIG.use_text_ori

    setup_ocr_device()
    download_models()


def setup_ocr_device() -> None:
    """
    Настройка устройств для OCR с поддержкой CUDA, DirectML и OpenVINO
    """
    available_providers = ort.get_available_providers()
    
    if not utils.CONFIG.use_gpu:
        # CPU режим
        utils.CONFIG.ocr_opts["use_gpu"] = False
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = utils.CONFIG.cpu_onnx_intra_threads
        utils.CONFIG.ocr_opts["onnx_sess_options"] = sess_opt
        return
    
    gpu_provider = utils.CONFIG.gpu_provider.lower()
    
    if gpu_provider == "cuda" and "CUDAExecutionProvider" in available_providers:
        utils.CONFIG.ocr_opts["use_gpu"] = True
        ort.preload_dlls()
        
    elif gpu_provider == "directml" and "DmlExecutionProvider" in available_providers:
        utils.CONFIG.ocr_opts["use_gpu"] = True
        # Настройка DirectML провайдера
        utils.CONFIG.ocr_opts["providers"] = ['DmlExecutionProvider']
        
    elif gpu_provider == "openvino" and "OpenVINOExecutionProvider" in available_providers:
        utils.CONFIG.ocr_opts["use_gpu"] = True
        # Настройка OpenVINO провайдера
        utils.CONFIG.ocr_opts["providers"] = ['OpenVINOExecutionProvider']
        
    else:
        # Fallback на CPU если выбранный GPU провайдер недоступен
        logger.warning(f"GPU provider '{gpu_provider}' not available. Falling back to CPU.")
        utils.CONFIG.ocr_opts["use_gpu"] = False
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = utils.CONFIG.cpu_onnx_intra_threads
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
    model_name = f"PP-OCRv5_{'mobile' if utils.CONFIG.use_mobile_model else 'server'}_det"
    det_config = {"model_save_dir": utils.CONFIG.ocr_opts["model_save_dir"], "model_name": model_name,
                  "box_thresh": utils.CONFIG.bbox_drop_score, "use_gpu": utils.CONFIG.ocr_opts["use_gpu"]}
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
    if device == "GPU":
        no_processes = utils.CONFIG.gpu_ocr_processes
        device = f"{device} ({utils.CONFIG.gpu_provider.upper()})"
    else:
        no_processes = utils.CONFIG.cpu_ocr_processes
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
    optimizer = PerformanceOptimiser() if utils.CONFIG.auto_optimize_perf else NullPerformanceOptimiser()
    with ThreadPoolExecutor(no_processes) as executor:
        futures = [executor.submit(extract_text, ocr_engine, text_output, files, line_sep) for files in file_batches]
        for i, f in enumerate(as_completed(futures)):  # as each  process completes
            f.result()  # Prevents silent bugs. Exceptions raised will be displayed.
            utils.print_progress(i, no_batches - 1, prefix)
            optimizer.record_perf()
    optimizer.optimise_performance()
    logger.info(f"{prefix} done!")
