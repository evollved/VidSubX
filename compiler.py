import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from time import perf_counter


def run_command(command: list, use_shell: bool = False) -> None:
    subprocess.run(command, check=True, shell=use_shell)


def install_requirements(device: str) -> None:
    print(f"\nInstalling {device} requirements...")
    run_command(['pip', 'install', '-r', f'requirements-{device}.txt'])


def uninstall_requirements(device: str) -> None:
    print(f"\nUninstalling {device} requirements...")
    run_command(['pip', 'uninstall', '-r', f'requirements-{device}.txt', '-y'])


def install_package(name: str) -> None:
    print(f"\n...Installing package {name}...")
    run_command(["pip", "install", name])


def download_all_models() -> None:
    import utilities.utils as utils
    from custom_ocr import CustomPaddleOCR

    txt_line_ori_models = ["PP-LCNet_x0_25_textline_ori", "PP-LCNet_x1_0_textline_ori"]
    det_models = ['PP-OCRv5_mobile_det', 'PP-OCRv5_server_det']
    rec_models = ['PP-OCRv5_mobile_rec', 'PP-OCRv5_server_rec', 'arabic_PP-OCRv5_mobile_rec',
                  'cyrillic_PP-OCRv5_mobile_rec', 'devanagari_PP-OCRv5_mobile_rec', 'en_PP-OCRv5_mobile_rec',
                  'eslav_PP-OCRv5_mobile_rec', 'korean_PP-OCRv5_mobile_rec', 'latin_PP-OCRv5_mobile_rec',
                  'ta_PP-OCRv5_mobile_rec', 'te_PP-OCRv5_mobile_rec']
    utils.CONFIG.ocr_opts["use_gpu"] = False
    for model in txt_line_ori_models:
        print(f"\nChecking for {model} model...")
        _ = CustomPaddleOCR(textline_orientation_model_name=model, **utils.CONFIG.ocr_opts)
        print("-" * 150)
    print("-" * 200)
    for model in det_models:
        print(f"\nChecking for {model} model...")
        _ = CustomPaddleOCR(text_detection_model_name=model, **utils.CONFIG.ocr_opts)
        print("-" * 150)
    print("-" * 200)
    for model in rec_models:
        print(f"\nChecking for {model} model...")
        _ = CustomPaddleOCR(text_recognition_model_name=model, **utils.CONFIG.ocr_opts)
        print("-" * 150)


def remove_non_onnx_models() -> None:
    import utilities.utils as utils

    print("\nRemoving all non Onnx Models...")
    for file in utils.CONFIG.model_dir.rglob("*"):
        if file.is_file() and ".onnx" not in file.name and ".yml" not in file.name:
            print(f"Removing file: {file}")
            file.unlink()


def remove_compiler_leftovers() -> None:
    print("\nRemoving compiler leftovers...")
    Path("gui.spec").unlink()
    shutil.rmtree("build")


def compile_program(gpu_enabled: bool) -> None:
    cmd = [
        "pyinstaller",
        "--add-data=installer/vsx.ico.ico:installer",
        "--add-data=models:models",
        "--collect-data=custom_ocr",
        "--icon=installer/vsx.ico",
        "--noconsole",
        "--noconfirm",
        "--clean",
    ]
    if gpu_enabled:
        cmd.append("--collect-binaries=nvidia")
    cmd.append("gui.py")
    print(f"\nCompiling program with PyInstaller... \nCommand: {' '.join(cmd)}")
    run_command(cmd, True)


def build_dist(gpu_enabled: bool) -> None:
    start_time = perf_counter()
    if gpu_enabled:
        uninstall_requirements("cpu")
        install_requirements("gpu")
    else:
        uninstall_requirements("gpu")
        install_requirements("cpu")
    install_package("pyinstaller==6.18.0")

    download_all_models()
    remove_non_onnx_models()
    compile_program(gpu_enabled)

    remove_compiler_leftovers()

    print(f"\nCompilation Duration: {timedelta(seconds=round(perf_counter() - start_time))}")


if __name__ == '__main__':
    build_dist(gpu_enabled=True)
