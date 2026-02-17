import logging

import cv2 as cv
import requests
from packaging.version import Version

from infra.app_paths import AppPaths
from shared.config import Config

logger = logging.getLogger(__name__)


def frame_no_to_duration(frame_no: float | int, fps: float | int) -> str:
    """
    Covert frame number to milliseconds then to time code duration.
    """
    frame_no_to_ms = (frame_no / fps) * 1000
    duration = timecode(frame_no_to_ms).replace(",", ":")
    return duration


def timecode(frame_no_in_milliseconds: float) -> str:
    """
    Use to frame no in milliseconds to create timecode.
    """
    # Calculate the components of the timecode.
    total_seconds = frame_no_in_milliseconds // 1000  # Convert milliseconds to total seconds.
    milliseconds_remainder = frame_no_in_milliseconds % 1000  # Calculate the remaining milliseconds.
    seconds = total_seconds % 60  # Calculate the seconds component (remainder after removing minutes).
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600  # Calculate the number of hours in the total seconds.
    return "%02d:%02d:%02d,%03d" % (hours, minutes, seconds, milliseconds_remainder)


def video_details(video_path: str) -> tuple:
    """
    Get the video details of the video in path.
    :return: video details
    """
    capture = cv.VideoCapture(video_path)
    fps = capture.get(cv.CAP_PROP_FPS)
    frame_total = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return fps, frame_total, frame_width, frame_height


def default_sub_area(frame_width: int, frame_height: int) -> tuple:
    """
    Returns a default subtitle area that can be used if no subtitle is given.
    :return: Position of subtitle relative to the resolution of the video. x2 = width and y2 = height
    """
    x1, y1, x2, y2 = 0, int(frame_height * Config.subarea_height_scaler), frame_width, frame_height
    return x1, y1, x2, y2


def check_for_updates() -> None:
    """
    Checks GitHub for a new release.
    """
    try:
        response = requests.get(f"https://api.github.com/repos/voun7/{AppPaths.program_name}/releases/latest")
        if response.status_code == 200:
            data = response.json()
            latest_version = Version(data["tag_name"])
            current_version = Version(AppPaths.version_file.read_text())
            if latest_version > current_version:
                logger.info(f"Version {latest_version} is now available.\nLink: {data['html_url']}")
            else:
                logger.debug("No new updates available.")
    except Exception as error:
        logger.info("Failed to check for updates!")
        logger.debug(error)
        return


def cancel_futures(futures: list) -> None:
    logger.info("Cancelling all pending processes...")
    for future in futures:
        cancelled = future.cancel()
        logger.debug(f"Attempted to cancel {future}. Cancelled: {cancelled}")


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
