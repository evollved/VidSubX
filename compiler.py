import platform
import shutil
import site
import subprocess
from datetime import timedelta
from pathlib import Path
from time import perf_counter


def run_command(command: list, use_shell: bool = False) -> None:
    subprocess.run(command, check=True, shell=use_shell)


def install_requirements(device: str) -> None:
    print("\nInstalling requirements...")
    run_command(['pip', 'install', '-r', f'requirements-{device}.txt'])


def install_package(name: str) -> None:
    print(f"\n...Installing package {name}...")
    run_command(["pip", "install", name])


def uninstall_package(name: str) -> None:
    temp_dir = Path(f"{site.getsitepackages()[1]}/~addle")
    if temp_dir.exists():
        print("\nRemoving undeleted temp directory...")
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n...Uninstalling package {name}...")
    run_command(["pip", "uninstall", "-y", name])


def download_all_models() -> None:
    import utilities.utils as utils
    from custom_ocr import CustomPaddleOCR

    txt_line_ori_models = ["PP-LCNet_x0_25_textline_ori", "PP-LCNet_x1_0_textline_ori"]
    det_models = ['PP-OCRv3_mobile_det', 'PP-OCRv3_server_det', 'PP-OCRv4_mobile_det', 'PP-OCRv4_server_det',
                  'PP-OCRv5_mobile_det', 'PP-OCRv5_server_det']
    rec_models = ['PP-OCRv3_mobile_rec', 'PP-OCRv4_mobile_rec', 'PP-OCRv4_server_rec', 'PP-OCRv5_mobile_rec',
                  'PP-OCRv5_server_rec', 'arabic_PP-OCRv3_mobile_rec', 'arabic_PP-OCRv5_mobile_rec',
                  'chinese_cht_PP-OCRv3_mobile_rec', 'cyrillic_PP-OCRv3_mobile_rec', 'cyrillic_PP-OCRv5_mobile_rec',
                  'devanagari_PP-OCRv3_mobile_rec', 'devanagari_PP-OCRv5_mobile_rec', 'en_PP-OCRv3_mobile_rec',
                  'en_PP-OCRv4_mobile_rec', 'en_PP-OCRv5_mobile_rec', 'eslav_PP-OCRv5_mobile_rec',
                  'japan_PP-OCRv3_mobile_rec', 'ka_PP-OCRv3_mobile_rec', 'korean_PP-OCRv3_mobile_rec',
                  'korean_PP-OCRv5_mobile_rec', 'latin_PP-OCRv3_mobile_rec', 'latin_PP-OCRv5_mobile_rec',
                  'ta_PP-OCRv3_mobile_rec', 'ta_PP-OCRv5_mobile_rec', 'te_PP-OCRv3_mobile_rec',
                  'te_PP-OCRv5_mobile_rec']
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


def compile_program() -> None:
    cmd = [
        "nuitka",
        "--standalone",
        "--enable-plugin=tk-inter",
        "--windows-console-mode=disable",
        "--include-data-dir=models=models",
        "--include-package-data=custom_ocr",
        "--include-data-files=docs/images/vsx.ico=docs/images/vsx.ico",
        "--windows-icon-from-ico=docs/images/vsx.ico",
        "--remove-output",
        "gui.py"
    ]
    print(f"\nCompiling program with Nuitka... \nCommand: {' '.join(cmd)}")
    run_command(cmd, True)


def rename_exe() -> None:
    print("\nRenaming exe file...")
    exe_file = Path("gui.dist/gui.exe")
    exe_file.rename("gui.dist/VSX.exe")


def get_gpu_files() -> None:
    print("\nCopying GPU files...")
    gpu_files_dir = Path(site.getsitepackages()[1], "nvidia")
    required_dirs = ["cudnn", "cufft", "cublas", "cuda_runtime"]
    for dir_name in required_dirs:
        shutil.copytree(gpu_files_dir / f"{dir_name}/bin", f"gui.dist/nvidia/{dir_name}/bin")


def zip_files(gpu_enabled: bool) -> None:
    print("\nZipping distribution files...")
    name = f"VSX-{platform.system()}-{'GPU' if gpu_enabled else 'CPU'}-v"
    shutil.make_archive(name, "zip", "gui.dist")


def delete_dist_dir() -> None:
    print("\nRemoving distribution directory...")
    shutil.rmtree("gui.dist")


def main(gpu_enabled: bool) -> None:
    start_time = perf_counter()

    if gpu_enabled:
        uninstall_package("onnxruntime")
        install_requirements("gpu")
    else:
        uninstall_package("onnxruntime-gpu")
        install_requirements("cpu")
    install_package("Nuitka==2.8.1")

    download_all_models()
    remove_non_onnx_models()
    compile_program()
    rename_exe()
    if gpu_enabled:
        get_gpu_files()
    zip_files(gpu_enabled)
    delete_dist_dir()

    print(f"\nCompilation Duration: {timedelta(seconds=round(perf_counter() - start_time))}")


if __name__ == '__main__':
    main(False)
    main(True)
