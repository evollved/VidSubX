#!/usr/bin/env python3
"""
Command-line interface for VidSubX - Extract hardcoded subtitles from videos.

This module provides a CLI for the SubtitleExtractor class, allowing users to
extract subtitles from videos without launching the GUI. All configuration
options available in the GUI preferences are supported as command-line arguments.

Usage examples:
    python cmd.py video.mp4
    python cmd.py video.mp4 --sub-area 100 800 1800 1000
    python cmd.py video.mp4 --start-frame 500 --stop-frame 5000
    python cmd.py video.mp4 --lang en --use-gpu
    python cmd.py video.mp4 --output custom_output.log
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path if running script directly
if __name__ == "__main__":
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

from infra.logger_setup import setup_logging
from main import SubtitleExtractor, setup_ocr
from shared.config import CONFIG, physical_cores


def parse_sub_area(sub_area_str: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse subtitle area coordinates from string.
    
    Args:
        sub_area_str: String with 4 integers separated by spaces or commas
        
    Returns:
        Tuple of (x1, y1, x2, y2) or None if parsing fails
    """
    if not sub_area_str:
        return None
    
    # Replace commas with spaces and split
    cleaned = sub_area_str.replace(',', ' ').strip()
    parts = cleaned.split()
    
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Subtitle area must have 4 coordinates, got {len(parts)}"
        )
    
    try:
        coords = tuple(int(p) for p in parts)
        return coords  # type: ignore
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Subtitle area coordinates must be integers"
        )


def parse_range(value: str) -> int:
    """
    Parse integer range values.
    
    Args:
        value: String representation of integer
        
    Returns:
        Integer value
    """
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Value must be an integer, got {value}")


def parse_float_range(value: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Parse float values with range checking.
    
    Args:
        value: String representation of float
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Float value
    """
    try:
        val = float(value)
        if val < min_val or val > max_val:
            raise argparse.ArgumentTypeError(
                f"Value must be between {min_val} and {max_val}, got {val}"
            )
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Value must be a float, got {value}")


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with all available options.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Extract hardcoded subtitles from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4
  %(prog)s video.mp4 --sub-area 100 800 1800 1000
  %(prog)s video.mp4 --start-frame 500 --stop-frame 5000
  %(prog)s video.mp4 --lang en --use-gpu --output subtitles.log
        """
    )
    
    # Required arguments
    parser.add_argument(
        'video_file',
        type=str,
        help='Path to the video file to process'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path for logging (default: video_filename.log in same directory)'
    )
    
    # Subtitle area options
    parser.add_argument(
        '--sub-area',
        type=parse_sub_area,
        metavar='X1 Y1 X2 Y2',
        help='Subtitle area coordinates: x1 y1 x2 y2 (e.g., "100 800 1800 1000")'
    )
    
    # Frame range options
    parser.add_argument(
        '--start-frame',
        type=int,
        help='Starting frame number for extraction'
    )
    
    parser.add_argument(
        '--stop-frame',
        type=int,
        help='Stopping frame number for extraction'
    )
    
    # ===== Subtitle Detection Settings =====
    detection_group = parser.add_argument_group('Subtitle Detection Settings')
    
    detection_group.add_argument(
        '--split-start',
        type=lambda x: parse_float_range(x, 0.0, 0.5),
        metavar='FLOAT',
        help=f'Relative start position for detection (default: {CONFIG.split_start})'
    )
    
    detection_group.add_argument(
        '--split-stop',
        type=lambda x: parse_float_range(x, 0.5, 1.0),
        metavar='FLOAT',
        help=f'Relative stop position for detection (default: {CONFIG.split_stop})'
    )
    
    detection_group.add_argument(
        '--no-of-frames',
        type=int,
        metavar='INT',
        help=f'Number of frames to analyze (default: {CONFIG.no_of_frames})'
    )
    
    detection_group.add_argument(
        '--sub-area-x-padding',
        type=lambda x: parse_float_range(x, 0.5, 1.0),
        metavar='FLOAT',
        help=f'X-axis relative padding (default: {CONFIG.sub_area_x_rel_padding})'
    )
    
    detection_group.add_argument(
        '--sub-area-y-padding',
        type=int,
        metavar='INT',
        help=f'Y-axis absolute padding in pixels (default: {CONFIG.sub_area_y_abs_padding})'
    )
    
    detection_group.add_argument(
        '--bbox-drop-score',
        type=lambda x: parse_float_range(x, 0.0, 1.0),
        metavar='FLOAT',
        help=f'Bounding box drop score threshold (default: {CONFIG.bbox_drop_score})'
    )
    
    detection_group.add_argument(
        '--use-search-area',
        action='store_true',
        help='Use default search area for detection (default: enabled)'
    )
    
    detection_group.add_argument(
        '--no-use-search-area',
        action='store_false',
        dest='use_search_area',
        help='Disable default search area (use full frame)'
    )
    detection_group.set_defaults(use_search_area=CONFIG.use_search_area)
    
    # ===== Frame Extraction Settings =====
    frame_group = parser.add_argument_group('Frame Extraction Settings')
    
    frame_group.add_argument(
        '--frame-frequency',
        type=int,
        metavar='INT',
        help=f'Frame extraction frequency (every N frames) (default: {CONFIG.frame_extraction_frequency})'
    )
    
    frame_group.add_argument(
        '--frame-batch-size',
        type=int,
        metavar='INT',
        help=f'Frame extraction batch size (default: {CONFIG.frame_extraction_batch_size})'
    )
    
    # ===== Text Extraction Settings =====
    text_group = parser.add_argument_group('Text Extraction Settings')
    
    text_group.add_argument(
        '--text-batch-size',
        type=int,
        metavar='INT',
        help=f'Text extraction batch size (default: {CONFIG.text_extraction_batch_size})'
    )
    
    text_group.add_argument(
        '--text-drop-score',
        type=lambda x: parse_float_range(x, 0.0, 1.0),
        metavar='FLOAT',
        help=f'Text recognition drop score (default: {CONFIG.text_drop_score})'
    )
    
    text_group.add_argument(
        '--line-break',
        action='store_true',
        help='Use line breaks in extracted text (default: enabled)'
    )
    
    text_group.add_argument(
        '--no-line-break',
        action='store_false',
        dest='line_break',
        help='Disable line breaks (use spaces)'
    )
    text_group.set_defaults(line_break=CONFIG.line_break)
    
    # ===== Subtitle Generator Settings =====
    subtitle_group = parser.add_argument_group('Subtitle Generator Settings')
    
    subtitle_group.add_argument(
        '--similarity-threshold',
        type=lambda x: parse_float_range(x, 0.0, 1.0),
        metavar='FLOAT',
        help=f'Text similarity threshold (default: {CONFIG.text_similarity_threshold})'
    )
    
    subtitle_group.add_argument(
        '--min-consecutive-dur',
        type=float,
        metavar='FLOAT',
        help=f'Minimum consecutive subtitle duration in ms (default: {CONFIG.min_consecutive_sub_dur_ms})'
    )
    
    subtitle_group.add_argument(
        '--max-consecutive-short',
        type=int,
        metavar='INT',
        help=f'Maximum consecutive short durations (default: {CONFIG.max_consecutive_short_durs})'
    )
    
    subtitle_group.add_argument(
        '--min-sub-duration',
        type=float,
        metavar='FLOAT',
        help=f'Minimum subtitle duration in ms (default: {CONFIG.min_sub_duration_ms})'
    )
    
    # ===== OCR Engine Settings =====
    ocr_engine_group = parser.add_argument_group('OCR Engine Settings')
    
    ocr_engine_group.add_argument(
        '--lang',
        '--language',
        dest='ocr_rec_language',
        type=str,
        metavar='CODE',
        help=f'OCR recognition language code (default: {CONFIG.ocr_rec_language})'
    )
    
    ocr_engine_group.add_argument(
        '--use-mobile-model',
        action='store_true',
        help='Use mobile OCR model (faster, less accurate) (default: enabled)'
    )
    
    ocr_engine_group.add_argument(
        '--no-use-mobile-model',
        action='store_false',
        dest='use_mobile_model',
        help='Use server OCR model (slower, more accurate)'
    )
    ocr_engine_group.set_defaults(use_mobile_model=CONFIG.use_mobile_model)
    
    ocr_engine_group.add_argument(
        '--use-text-ori',
        action='store_true',
        help='Use text line orientation model (default: disabled)'
    )
    
    ocr_engine_group.add_argument(
        '--no-use-text-ori',
        action='store_false',
        dest='use_text_ori',
        help='Disable text line orientation model'
    )
    ocr_engine_group.set_defaults(use_text_ori=CONFIG.use_text_ori)
    
    # ===== OCR Performance Settings =====
    perf_group = parser.add_argument_group('OCR Performance Settings')
    
    perf_group.add_argument(
        '--cpu-processes',
        type=int,
        metavar='INT',
        help=f'Number of CPU OCR processes (default: {CONFIG.cpu_ocr_processes}, max: {physical_cores()})'
    )
    
    perf_group.add_argument(
        '--cpu-threads',
        type=int,
        metavar='INT',
        help=f'CPU ONNX intra threads (default: {CONFIG.cpu_onnx_intra_threads}, max: {physical_cores()})'
    )
    
    perf_group.add_argument(
        '--gpu-processes',
        type=int,
        metavar='INT',
        help=f'Number of GPU OCR processes (default: {CONFIG.gpu_ocr_processes})'
    )
    
    perf_group.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU if available (default: enabled)'
    )
    
    perf_group.add_argument(
        '--no-use-gpu',
        action='store_false',
        dest='use_gpu',
        help='Disable GPU usage (use CPU only)'
    )
    perf_group.set_defaults(use_gpu=CONFIG.use_gpu)
    
    perf_group.add_argument(
        '--auto-optimize',
        action='store_true',
        help='Auto-optimize performance (default: enabled)'
    )
    
    perf_group.add_argument(
        '--no-auto-optimize',
        action='store_false',
        dest='auto_optimize_perf',
        help='Disable auto-optimization'
    )
    perf_group.set_defaults(auto_optimize_perf=CONFIG.auto_optimize_perf)
    
    # ===== Notification Settings =====
    notify_group = parser.add_argument_group('Notification Settings (Windows only)')
    
    notify_group.add_argument(
        '--notify-sound',
        type=str,
        choices=['Default', 'IM', 'Mail', 'Reminder', 'SMS', 'Silent'],
        help=f'Notification sound (default: {CONFIG.win_notify_sound})'
    )
    
    notify_group.add_argument(
        '--loop-sound',
        action='store_true',
        help='Loop notification sound (default: enabled)'
    )
    
    notify_group.add_argument(
        '--no-loop-sound',
        action='store_false',
        dest='win_notify_loop_sound',
        help='Disable sound looping'
    )
    notify_group.set_defaults(win_notify_loop_sound=CONFIG.win_notify_loop_sound)
    
    # ===== Debug Options =====
    debug_group = parser.add_argument_group('Debug Options')
    
    debug_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    debug_group.add_argument(
        '--keep-cache',
        action='store_true',
        help='Keep cache files after extraction (for debugging)'
    )
    
    return parser


def setup_output_logging(output_path: Optional[str], verbose: bool) -> logging.FileHandler:
    """
    Setup file logging to the specified output path.
    
    Args:
        output_path: Path to output log file
        verbose: Enable debug logging
        
    Returns:
        Configured file handler
    """
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if output_path:
        log_file = Path(output_path)
    else:
        # Default to video filename with .log extension
        log_file = Path(args.video_file).with_suffix('.log')
    
    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(log_format)
    
    return file_handler


def update_config_from_args(args: argparse.Namespace) -> None:
    """
    Update CONFIG with values from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    config_updates = {}
    
    # Map command-line arguments to config keys
    arg_to_config = {
        # Subtitle Detection
        'split_start': 'split_start',
        'split_stop': 'split_stop',
        'no_of_frames': 'no_of_frames',
        'sub_area_x_rel_padding': 'sub_area_x_rel_padding',
        'sub_area_y_abs_padding': 'sub_area_y_abs_padding',
        'bbox_drop_score': 'bbox_drop_score',
        'use_search_area': 'use_search_area',
        
        # Frame Extraction
        'frame_frequency': 'frame_extraction_frequency',
        'frame_batch_size': 'frame_extraction_batch_size',
        
        # Text Extraction
        'text_batch_size': 'text_extraction_batch_size',
        'text_drop_score': 'text_drop_score',
        'line_break': 'line_break',
        
        # Subtitle Generator
        'similarity_threshold': 'text_similarity_threshold',
        'min_consecutive_dur': 'min_consecutive_sub_dur_ms',
        'max_consecutive_short': 'max_consecutive_short_durs',
        'min_sub_duration': 'min_sub_duration_ms',
        
        # OCR Engine
        'ocr_rec_language': 'ocr_rec_language',
        'use_mobile_model': 'use_mobile_model',
        'use_text_ori': 'use_text_ori',
        
        # OCR Performance
        'cpu_processes': 'cpu_ocr_processes',
        'cpu_threads': 'cpu_onnx_intra_threads',
        'gpu_processes': 'gpu_ocr_processes',
        'use_gpu': 'use_gpu',
        'auto_optimize_perf': 'auto_optimize_perf',
        
        # Notifications
        'notify_sound': 'win_notify_sound',
        'win_notify_loop_sound': 'win_notify_loop_sound',
    }
    
    for arg_name, config_name in arg_to_config.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            config_updates[config_name] = getattr(args, arg_name)
    
    if config_updates:
        logging.getLogger(__name__).debug(f"Updating config with: {config_updates}")
        CONFIG.set_config(**config_updates)


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Add file handler for output logging
    file_handler = setup_output_logging(args.output, args.verbose)
    logging.getLogger().addHandler(file_handler)
    
    start_time = time.time()
    
    try:
        # Verify video file exists
        video_path = Path(args.video_file)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output log: {file_handler.baseFilename}")
        
        # Update configuration with command-line arguments
        update_config_from_args(args)
        
        # Setup OCR
        logger.info("Setting up OCR...")
        setup_ocr()
        
        # Create extractor instance
        extractor = SubtitleExtractor()
        
        # Run extraction
        logger.info("Starting subtitle extraction...")
        result = extractor.run_extraction(
            video_path=str(video_path),
            sub_area=args.sub_area,
            start_frame=args.start_frame,
            stop_frame=args.stop_frame
        )
        
        # Log result
        elapsed = time.time() - start_time
        if result:
            logger.info(f"Subtitle file created: {result}")
            logger.info(f"Extraction completed in {elapsed:.2f} seconds")
            print(f"\n✅ Subtitle file created: {result}")
        else:
            logger.error("Subtitle extraction failed")
            print("\n❌ Subtitle extraction failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\n⚠️ Process interrupted")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    finally:
        # Clean up file handler
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    main()