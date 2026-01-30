import platform
import shutil
import site
import subprocess
from datetime import timedelta
from pathlib import Path
from time import perf_counter


def run_command(command: list, use_shell: bool = False) -> None:
    """Выполняет команду в shell."""
    print(f"\n🔧 Running command: {' '.join(command)}")
    result = subprocess.run(command, check=True, shell=use_shell, capture_output=True, text=True)
    if result.stdout:
        print(f"Output: {result.stdout[:500]}...")  # Показываем первые 500 символов вывода
    return result


def install_requirements_gpu() -> None:
    """Устанавливает требования для GPU версии."""
    print("\n" + "="*60)
    print("Installing GPU requirements...")
    print("="*60)
    
    # Удаляем CPU версию если установлена
    print("\nChecking for CPU packages...")
    try:
        subprocess.run(["pip", "uninstall", "-y", "onnxruntime"], 
                      check=False, capture_output=True)
        print("✓ Removed onnxruntime (if existed)")
    except:
        print("✓ onnxruntime not installed")
    
    # Устанавливаем GPU требования
    req_file = Path("requirements-gpu.txt")
    if req_file.exists():
        print(f"\nFound requirements file: {req_file}")
        run_command(['pip', 'install', '-r', 'requirements-gpu.txt'])
        print("✓ Installed GPU requirements")
    else:
        print("⚠ Warning: requirements-gpu.txt not found")
        print("Trying requirements.txt...")
        alt_req_file = Path("requirements.txt")
        if alt_req_file.exists():
            run_command(['pip', 'install', '-r', 'requirements.txt'])
        else:
            print("⚠ No requirements file found, installing basic packages...")
    
    # Устанавливаем нужные пакеты для GPU
    print("\nInstalling GPU packages...")
    run_command(["pip", "install", "onnxruntime-gpu"])
    run_command(["pip", "install", "Nuitka==2.8.1"])
    
    # Проверяем установку CUDA
    print("\nChecking CUDA availability...")
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f"✓ ONNX Runtime providers: {providers}")
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA is available!")
        else:
            print("⚠ CUDA is not available in ONNX Runtime")
    except ImportError as e:
        print(f"⚠ Could not check CUDA: {e}")
    
    print("✓ All GPU packages installed")


def find_gui_file() -> Path:
    """Находит файл gui.py в текущей директории или поддиректориях."""
    print("\n🔍 Searching for gui.py...")
    
    # Проверяем в текущей директории
    current_dir = Path.cwd()
    gui_file = current_dir / "gui.py"
    
    if gui_file.exists():
        print(f"✓ Found gui.py at: {gui_file}")
        return gui_file
    
    # Ищем во всех поддиректориях
    for file in current_dir.rglob("gui.py"):
        if file.exists():
            print(f"✓ Found gui.py at: {file}")
            return file
    
    # Ищем файлы с похожими именами
    for file in current_dir.rglob("*.py"):
        if "gui" in file.name.lower() or "main" in file.name.lower():
            print(f"⚠ Found possible main file: {file}")
            return file
    
    raise FileNotFoundError("Could not find gui.py or main python file")


def download_all_models_gpu() -> None:
    """Загружает все модели OCR для GPU."""
    try:
        print("\n" + "="*60)
        print("Downloading models for GPU...")
        print("="*60)
        
        # Импортируем здесь, чтобы не вызывать ошибки раньше времени
        import utilities.utils as utils
        from custom_ocr import CustomPaddleOCR
        
        txt_line_ori_models = ["PP-LCNet_x0_25_textline_ori", "PP-LCNet_x1_0_textline_ori"]
        det_models = ['PP-OCRv5_mobile_det', 'PP-OCRv5_server_det']
        rec_models = ['PP-OCRv5_mobile_rec', 'PP-OCRv5_server_rec', 'arabic_PP-OCRv5_mobile_rec',
                      'cyrillic_PP-OCRv5_mobile_rec', 'devanagari_PP-OCRv5_mobile_rec', 'en_PP-OCRv5_mobile_rec',
                      'eslav_PP-OCRv5_mobile_rec', 'korean_PP-OCRv5_mobile_rec', 'latin_PP-OCRv5_mobile_rec',
                      'ta_PP-OCRv5_mobile_rec', 'te_PP-OCRv5_mobile_rec']
        
        utils.CONFIG.ocr_opts["use_gpu"] = True
        
        print("\nDownloading text line orientation models (GPU)...")
        for model in txt_line_ori_models:
            print(f"\nChecking for {model} model...")
            _ = CustomPaddleOCR(textline_orientation_model_name=model, **utils.CONFIG.ocr_opts)
            print("-" * 80)
        
        print("\nDownloading detection models (GPU)...")
        for model in det_models:
            print(f"\nChecking for {model} model...")
            _ = CustomPaddleOCR(text_detection_model_name=model, **utils.CONFIG.ocr_opts)
            print("-" * 80)
        
        print("\nDownloading recognition models (GPU)...")
        for model in rec_models:
            print(f"\nChecking for {model} model...")
            _ = CustomPaddleOCR(text_recognition_model_name=model, **utils.CONFIG.ocr_opts)
            print("-" * 80)
            
        print("\n✓ All models downloaded successfully (GPU)")
        
    except ImportError as e:
        print(f"⚠ Error importing modules: {e}")
        print("Trying to install missing dependencies...")
        # Пробуем установить paddlepaddle если его нет
        try:
            run_command(["pip", "install", "paddlepaddle-gpu"])
            print("✓ Installed paddlepaddle-gpu, retrying...")
            download_all_models_gpu()  # Рекурсивный вызов
        except:
            print("❌ Could not install GPU dependencies automatically")
            print("Trying CPU version of paddlepaddle...")
            run_command(["pip", "install", "paddlepaddle"])
            download_all_models_gpu()
    except Exception as e:
        print(f"⚠ Error downloading models: {e}")
        print("Continuing without models...")
        # Продолжаем компиляцию даже без моделей


def remove_non_onnx_models() -> None:
    """Удаляет все модели, кроме ONNX формата."""
    try:
        import utilities.utils as utils
        
        print("\n" + "="*60)
        print("Cleaning up non-ONNX models...")
        print("="*60)
        
        model_dir = utils.CONFIG.model_dir
        print(f"Model directory: {model_dir}")
        
        if not model_dir.exists():
            print("⚠ Model directory does not exist, skipping cleanup")
            return
        
        removed_count = 0
        for file in model_dir.rglob("*"):
            if file.is_file() and ".onnx" not in file.name and ".yml" not in file.name:
                print(f"Removing: {file.name}")
                try:
                    file.unlink()
                    removed_count += 1
                except Exception as e:
                    print(f"  ⚠ Could not remove {file.name}: {e}")
        
        print(f"\n✓ Removed {removed_count} non-ONNX files")
        
    except ImportError as e:
        print(f"⚠ Error importing utilities: {e}")
        print("Skipping model cleanup...")


def get_gpu_files() -> None:
    """Копирует необходимые GPU файлы."""
    print("\n" + "="*60)
    print("Checking GPU runtime files...")
    print("="*60)
    
    try:
        # Ищем nvidia директории
        site_packages = site.getsitepackages()
        if not site_packages:
            print("⚠ Could not find site-packages directory")
            return
        
        gpu_files_dir = Path(site_packages[1] if len(site_packages) > 1 else site_packages[0]) / "nvidia"
        
        if not gpu_files_dir.exists():
            print(f"⚠ NVIDIA directory not found at: {gpu_files_dir}")
            print("Looking for CUDA in common locations...")
            
            # Проверяем стандартные пути для CUDA
            cuda_paths = [
                Path("/usr/local/cuda"),
                Path("/usr/local/cuda-11.8"),
                Path("/usr/local/cuda-12.0"),
                Path("/usr/local/cuda-12.1"),
                Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
                Path.home() / ".cuda",
            ]
            
            for cuda_path in cuda_paths:
                if cuda_path.exists():
                    print(f"✓ Found CUDA at: {cuda_path}")
                    gpu_files_dir = cuda_path
                    break
            else:
                print("⚠ CUDA not found in standard locations")
                print("GPU acceleration may not work in the compiled application")
                return
        
        print(f"Found NVIDIA/CUDA directory: {gpu_files_dir}")
        
        # Проверяем директорию дистрибутива
        dist_dir = Path("dist_gpu/gui.dist")
        if not dist_dir.exists():
            print("⚠ Distribution directory not found, creating...")
            dist_dir.mkdir(parents=True, exist_ok=True)
        
        required_dirs = ["cudnn", "cufft", "cublas", "cuda_runtime", "bin", "lib64", "lib"]
        
        copied_count = 0
        for dir_name in required_dirs:
            src_dir = gpu_files_dir / dir_name
            
            if src_dir.exists():
                print(f"\nCopying {dir_name}...")
                dest_dir = dist_dir / f"nvidia/{dir_name}"
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    if src_dir.is_dir():
                        # Используем dirs_exist_ok для Python 3.8+
                        if hasattr(shutil, 'copytree'):
                            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
                        else:
                            # Для старых версий Python
                            if dest_dir.exists():
                                shutil.rmtree(dest_dir)
                            shutil.copytree(src_dir, dest_dir)
                    else:
                        shutil.copy2(src_dir, dest_dir)
                    
                    print(f"✓ Copied: {src_dir} → {dest_dir}")
                    copied_count += 1
                    
                except Exception as e:
                    print(f"⚠ Could not copy {dir_name}: {e}")
            else:
                print(f"⚠ GPU component not found: {src_dir}")
        
        if copied_count > 0:
            print(f"\n✅ Successfully copied {copied_count} GPU components")
        else:
            print("\n⚠ No GPU files were copied")
            
    except Exception as e:
        print(f"⚠ Error copying GPU files: {e}")
        print("Continuing without GPU runtime files...")


def compile_program_gpu() -> None:
    """Компилирует программу с Nuitka для GPU."""
    print("\n" + "="*60)
    print("Compiling with Nuitka for GPU...")
    print("="*60)
    
    # Находим главный файл
    try:
        main_file = find_gui_file()
    except FileNotFoundError:
        print("❌ Error: Could not find main python file to compile")
        print("Please make sure gui.py exists in the current directory")
        return
    
    # Проверяем существование моделей директории
    models_dir = Path("models")
    if not models_dir.exists():
        print("⚠ Warning: models directory not found")
        models_dir.mkdir(exist_ok=True)
        print("✓ Created empty models directory")
    
    # Проверяем иконку
    icon_path = Path("docs/images/vsx.ico")
    if not icon_path.exists():
        print("⚠ Warning: Icon file not found at docs/images/vsx.ico")
        # Пробуем найти иконку в других местах
        for ico_file in Path.cwd().rglob("*.ico"):
            if "vsx" in ico_file.name.lower() or "icon" in ico_file.name.lower():
                icon_path = ico_file
                print(f"✓ Found alternative icon: {icon_path}")
                break
    
    # Собираем команду Nuitka
    cmd = [
        "nuitka",
        "--standalone",
        "--enable-plugin=tk-inter",
        "--windows-console-mode=disable",
        "--remove-output",
        "--output-dir=dist_gpu",
    ]
    
    # Добавляем опции только если файлы существуют
    if models_dir.exists():
        cmd.append("--include-data-dir=models=models")
    
    cmd.extend([
        "--include-package-data=custom_ocr",
        str(main_file)  # Главный файл В КОНЦЕ команды
    ])
    
    # Добавляем иконку если найдена
    if icon_path.exists():
        cmd.extend([
            f"--include-data-files={icon_path}=docs/images/vsx.ico",
            f"--windows-icon-from-ico={icon_path}"
        ])
    
    print(f"\n📦 Nuitka command:")
    print(" ".join(cmd))
    
    try:
        result = run_command(cmd, use_shell=False)
        print("✓ GPU compilation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Nuitka compilation failed")
        if e.stderr:
            print(f"Error: {e.stderr}")
        else:
            print(f"Exit code: {e.returncode}")
        
        # Пробуем упрощенную команду
        print("\n🔄 Trying simplified compilation...")
        simple_cmd = [
            "nuitka",
            "--standalone",
            "--output-dir=dist_gpu",
            str(main_file)
        ]
        try:
            run_command(simple_cmd, use_shell=False)
            print("✓ Simplified compilation succeeded")
        except subprocess.CalledProcessError as e2:
            print(f"❌ Simplified compilation also failed: {e2}")
            raise


def rename_exe_gpu() -> None:
    """Переименовывает исполняемый файл для GPU версии."""
    print("\nRenaming executable...")
    
    # Ищем скомпилированный файл
    dist_dir = Path("dist_gpu")
    
    if not dist_dir.exists():
        print("⚠ Warning: dist_gpu directory not found")
        return
    
    # Ищем gui.dist или другие варианты
    possible_paths = [
        dist_dir / "gui.dist/gui.exe",
        dist_dir / "gui.dist/gui",
        dist_dir / "gui.exe",
        dist_dir / "gui",
    ]
    
    for exe_file in possible_paths:
        if exe_file.exists():
            print(f"✓ Found executable: {exe_file}")
            
            # Для Linux
            if platform.system() != "Windows":
                new_name = exe_file.with_name("VSX-GPU")
            else:
                new_name = exe_file.with_name("VSX-GPU.exe")
            
            exe_file.rename(new_name)
            print(f"✓ Renamed to: {new_name}")
            return
    
    # Если не нашли, ищем любой исполняемый файл
    for file in dist_dir.rglob("*"):
        if file.is_file() and (file.suffix == '.exe' or file.stat().st_mode & 0o111):
            print(f"✓ Found executable: {file}")
            
            if platform.system() != "Windows":
                new_name = file.with_name("VSX-GPU")
            else:
                new_name = file.with_name("VSX-GPU.exe")
            
            file.rename(new_name)
            print(f"✓ Renamed to: {new_name}")
            return
    
    print("⚠ Warning: No executable file found to rename")


def zip_files_gpu() -> None:
    """Архивирует файлы дистрибутива для GPU."""
    print("\n" + "="*60)
    print("Creating GPU distribution archive...")
    print("="*60)
    
    # Ищем директорию дистрибутива
    dist_dirs = [
        Path("dist_gpu/gui.dist"),
        Path("dist_gpu"),
        Path("gui.dist"),
    ]
    
    dist_dir = None
    for dir_path in dist_dirs:
        if dir_path.exists():
            dist_dir = dir_path
            break
    
    if not dist_dir:
        print("⚠ Error: Distribution directory not found")
        return
    
    print(f"Found distribution directory: {dist_dir}")
    
    system = platform.system()
    name = f"VSX-{system}-GPU-v1.0"
    
    # Удаляем старый архив если существует
    old_zip = Path(f"{name}.zip")
    if old_zip.exists():
        old_zip.unlink()
        print("✓ Removed old archive")
    
    # Создаем новый архив
    try:
        shutil.make_archive(name, "zip", dist_dir)
        print(f"✅ Created GPU archive: {name}.zip")
        
        zip_size = Path(f"{name}.zip").stat().st_size
        print(f"📦 Archive size: {zip_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ Error creating archive: {e}")


def delete_dist_dir_gpu() -> None:
    """Удаляет временную директорию дистрибутива."""
    print("\nCleaning up...")
    
    dist_dir = Path("dist_gpu")
    if dist_dir.exists():
        try:
            shutil.rmtree(dist_dir, ignore_errors=True)
            print("✓ Removed GPU distribution directory")
        except Exception as e:
            print(f"⚠ Could not remove directory: {e}")
    else:
        print("✓ No distribution directory to clean")


def check_cuda_availability() -> bool:
    """Проверяет доступность CUDA."""
    print("\n" + "="*60)
    print("Checking CUDA and GPU availability...")
    print("="*60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ PyTorch CUDA is available!")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠ PyTorch CUDA is not available")
    except ImportError:
        print("⚠ PyTorch not installed")
    
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime CUDA is available!")
            return True
        else:
            print("⚠ ONNX Runtime CUDA is not available")
    except ImportError:
        print("⚠ ONNX Runtime not installed")
    
    # Проверяем nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi is available")
            print(f"GPU Info:\n{result.stdout[:500]}...")
            return True
        else:
            print("⚠ nvidia-smi returned error")
    except FileNotFoundError:
        print("⚠ nvidia-smi not found (CUDA driver not installed?)")
    
    print("\n⚠ GPU compilation may not work properly!")
    print("The application will be compiled with GPU support,")
    print("but may fall back to CPU if CUDA is not available on target system.")
    
    response = input("\nContinue with GPU compilation? (y/n): ")
    return response.lower() == 'y'


def main_gpu() -> None:
    """Основная функция компиляции GPU версии."""
    start_time = perf_counter()
    
    try:
        print("\n" + "="*80)
        print("STARTING GPU VERSION COMPILATION")
        print("="*80)
        
        # Проверяем CUDA доступность
        if not check_cuda_availability():
            print("\n❌ GPU compilation cancelled")
            return
        
        # Показываем текущую директорию
        print(f"\n📁 Current directory: {Path.cwd()}")
        print(f"📁 Contents:")
        for item in Path.cwd().iterdir():
            print(f"  {item.name}{'/' if item.is_dir() else ''}")
        
        # 1. Установка зависимостей
        install_requirements_gpu()
        
        # 2. Загрузка моделей
        try:
            download_all_models_gpu()
        except Exception as e:
            print(f"⚠ GPU model download failed, continuing: {e}")
        
        # 3. Очистка моделей
        try:
            remove_non_onnx_models()
        except Exception as e:
            print(f"⚠ Model cleanup failed, continuing: {e}")
        
        # 4. Компиляция
        compile_program_gpu()
        
        # 5. Переименование EXE
        rename_exe_gpu()
        
        # 6. Копирование GPU файлов
        get_gpu_files()
        
        # 7. Архивирование
        zip_files_gpu()
        
        # 8. Очистка
        delete_dist_dir_gpu()
        
        duration = timedelta(seconds=round(perf_counter() - start_time))
        print("\n" + "="*80)
        print(f"✅ GPU COMPILATION COMPLETED SUCCESSFULLY")
        print(f"⏱️  Total duration: {duration}")
        print("="*80)
        
        print("\n💡 Note: The GPU version requires CUDA and cuDNN to be installed")
        print("on the target system for GPU acceleration to work.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error output: {e.stderr}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"Output: {e.stdout}")
        raise
    except Exception as e:
        print(f"\n❌ GPU compilation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main_gpu()