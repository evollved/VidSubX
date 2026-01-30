import platform
import shutil
import site
import subprocess
import sys
import os
from datetime import timedelta
from pathlib import Path
from time import perf_counter


def get_system_info() -> dict:
    """Возвращает информацию о системе."""
    system = platform.system()
    arch = platform.machine()
    
    if system == "Windows":
        # Для Windows определяем разрядность
        is_64bit = sys.maxsize > 2**32
        arch = "x64" if is_64bit else "x86"
    elif system == "Linux":
        # Для Linux определяем конкретную архитектуру
        if "arm" in arch.lower() or "aarch" in arch.lower():
            arch = "arm64" if "64" in arch else "arm"
        elif "x86" in arch.lower():
            is_64bit = sys.maxsize > 2**32
            arch = "x64" if is_64bit else "x86"
    
    return {
        "system": system,
        "arch": arch,
        "is_windows": system == "Windows",
        "is_linux": system == "Linux",
        "is_mac": system == "Darwin"
    }


def run_command(command: list, use_shell: bool = False) -> None:
    """Выполняет команду в shell."""
    print(f"\n🔧 Running command: {' '.join(command)}")
    result = subprocess.run(command, check=True, shell=use_shell, capture_output=True, text=True)
    if result.stdout:
        # Показываем первые 500 символов вывода
        output_preview = result.stdout[:500]
        if len(result.stdout) > 500:
            output_preview += "..."
        print(f"Output: {output_preview}")
    return result


def install_requirements_gpu() -> None:
    """Устанавливает требования для GPU версии."""
    sys_info = get_system_info()
    
    print("\n" + "="*60)
    print(f"Installing GPU requirements for {sys_info['system']} ({sys_info['arch']})...")
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
    
    # Устанавливаем нужные пакеты для GPU в зависимости от системы
    print("\nInstalling GPU packages...")
    
    if sys_info["is_windows"]:
        # Для Windows устанавливаем onnxruntime-gpu
        run_command(["pip", "install", "onnxruntime-gpu"])
    elif sys_info["is_linux"]:
        # Для Linux устанавливаем соответствующую версию
        if sys_info["arch"] in ["arm", "arm64"]:
            print("⚠ GPU acceleration is limited on ARM architecture")
            run_command(["pip", "install", "onnxruntime"])
        else:
            try:
                run_command(["pip", "install", "onnxruntime-gpu"])
            except:
                print("⚠ Could not install onnxruntime-gpu, trying standard version...")
                run_command(["pip", "install", "onnxruntime"])
    
    # Устанавливаем Nuitka
    print("\nInstalling Nuitka...")
    try:
        run_command(["pip", "install", "Nuitka==2.8.1"])
    except:
        print("⚠ Could not install specific Nuitka version, trying latest...")
        run_command(["pip", "install", "Nuitka"])
    
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
    sys_info = get_system_info()
    
    try:
        print("\n" + "="*60)
        print(f"Downloading models for GPU ({sys_info['system']})...")
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
            if sys_info["is_windows"]:
                run_command(["pip", "install", "paddlepaddle-gpu"])
            elif sys_info["is_linux"]:
                if sys_info["arch"] in ["arm", "arm64"]:
                    run_command(["pip", "install", "paddlepaddle==2.5.2"])
                else:
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
    """Копирует необходимые GPU файлы в зависимости от системы."""
    sys_info = get_system_info()
    
    print("\n" + "="*60)
    print(f"Checking GPU runtime files for {sys_info['system']}...")
    print("="*60)
    
    try:
        if sys_info["is_windows"]:
            get_gpu_files_windows()
        elif sys_info["is_linux"]:
            get_gpu_files_linux()
        else:
            print(f"⚠ GPU file copying not supported for {sys_info['system']}")
            
    except Exception as e:
        print(f"⚠ Error copying GPU files: {e}")
        print("Continuing without GPU runtime files...")


def get_gpu_files_windows() -> None:
    """Копирует GPU файлы для Windows."""
    print("Looking for CUDA on Windows...")
    
    # Стандартные пути для CUDA на Windows
    cuda_paths = [
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
        Path("C:/CUDA"),
        Path.home() / "CUDA",
    ]
    
    cuda_version = None
    cuda_root = None
    
    # Ищем установленную CUDA
    for cuda_path in cuda_paths:
        if cuda_path.exists():
            # Ищем версии CUDA
            for version_dir in cuda_path.iterdir():
                if version_dir.is_dir() and version_dir.name.startswith("v"):
                    cuda_version = version_dir.name[1:]  # Убираем 'v'
                    cuda_root = version_dir
                    print(f"✓ Found CUDA {cuda_version} at: {cuda_root}")
                    break
    
    if not cuda_root:
        print("⚠ CUDA not found in standard Windows locations")
        print("GPU acceleration may not work in the compiled application")
        return
    
    # Проверяем директорию дистрибутива
    dist_dir = Path("dist_gpu/gui.dist")
    if not dist_dir.exists():
        print("⚠ Distribution directory not found, creating...")
        dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Копируем необходимые библиотеки
    required_files = {
        "bin": ["cudnn*.dll", "cublas*.dll", "cufft*.dll", "curand*.dll"],
        "lib/x64": ["cudnn*.lib", "cublas*.lib", "cufft*.lib"],
    }
    
    copied_count = 0
    for subdir, patterns in required_files.items():
        src_dir = cuda_root / subdir
        dest_dir = dist_dir / "cuda" / subdir
        
        if src_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for pattern in patterns:
                for file in src_dir.glob(pattern):
                    try:
                        shutil.copy2(file, dest_dir)
                        print(f"✓ Copied: {file.name}")
                        copied_count += 1
                    except Exception as e:
                        print(f"⚠ Could not copy {file.name}: {e}")
    
    if copied_count > 0:
        print(f"\n✅ Successfully copied {copied_count} CUDA files for Windows")
    else:
        print("\n⚠ No CUDA files were copied for Windows")


def get_gpu_files_linux() -> None:
    """Копирует GPU файлы для Linux."""
    print("Looking for CUDA on Linux...")
    
    # Стандартные пути для CUDA на Linux
    cuda_paths = [
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda-11.8"),
        Path("/usr/local/cuda-12.0"),
        Path("/usr/local/cuda-12.1"),
        Path("/usr/local/cuda-12.2"),
        Path("/usr/lib/cuda"),
        Path.home() / ".cuda",
    ]
    
    cuda_root = None
    
    # Ищем установленную CUDA
    for cuda_path in cuda_paths:
        if cuda_path.exists():
            cuda_root = cuda_path
            print(f"✓ Found CUDA at: {cuda_root}")
            break
    
    if not cuda_root:
        print("⚠ CUDA not found in standard Linux locations")
        
        # Проверяем установлены ли пакеты через apt
        try:
            result = subprocess.run(['dpkg', '-l'], capture_output=True, text=True)
            if 'cuda' in result.stdout.lower():
                print("✓ CUDA packages are installed via apt")
                cuda_root = Path("/usr")
        except:
            pass
        
        if not cuda_root:
            print("GPU acceleration may not work in the compiled application")
            return
    
    # Проверяем директорию дистрибутива
    dist_dir = Path("dist_gpu/gui.dist")
    if not dist_dir.exists():
        print("⚠ Distribution directory not found, creating...")
        dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Копируем необходимые библиотеки
    required_libs = [
        "libcudnn*.so*",
        "libcublas*.so*", 
        "libcufft*.so*",
        "libcurand*.so*",
        "libcudart*.so*",
    ]
    
    lib_dirs = [
        cuda_root / "lib64",
        cuda_root / "lib",
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu"),  # Для ARM
    ]
    
    copied_count = 0
    for lib_dir in lib_dirs:
        if lib_dir.exists():
            dest_dir = dist_dir / "lib"
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for pattern in required_libs:
                for file in lib_dir.glob(pattern):
                    try:
                        shutil.copy2(file, dest_dir)
                        print(f"✓ Copied: {file.name}")
                        copied_count += 1
                        
                        # Создаем символические ссылки
                        if ".so." in file.name:
                            base_name = file.name.split(".so.")[0] + ".so"
                            symlink_path = dest_dir / base_name
                            if not symlink_path.exists():
                                try:
                                    symlink_path.symlink_to(file.name)
                                except:
                                    pass
                    except Exception as e:
                        print(f"⚠ Could not copy {file.name}: {e}")
    
    if copied_count > 0:
        print(f"\n✅ Successfully copied {copied_count} CUDA libraries for Linux")
    else:
        print("\n⚠ No CUDA libraries were copied for Linux")


def compile_program_gpu() -> None:
    """Компилирует программу с Nuitka для GPU."""
    sys_info = get_system_info()
    
    print("\n" + "="*60)
    print(f"Compiling with Nuitka for GPU ({sys_info['system']})...")
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
    
    # Определяем количество ядер процессора
    try:
        cpu_count = os.cpu_count() or 4
        jobs = max(1, cpu_count // 2)
        print(f"📊 CPU cores: {cpu_count}, using {jobs} jobs for compilation")
    except:
        jobs = 4  # Значение по умолчанию
        print(f"📊 Using default job count: {jobs}")
    
    # Собираем команду Nuitka
    cmd = [
        "nuitka",
        "--standalone",
        "--enable-plugin=tk-inter",
        "--remove-output",
        f"--jobs={jobs}",
        "--output-dir=dist_gpu",
    ]
    
    # Добавляем системно-зависимые опции
    if sys_info["is_windows"]:
        cmd.append("--windows-console-mode=disable")
        if sys_info["arch"] == "x64":
            cmd.append("--mingw64")
        else:
            cmd.append("--mingw32")
    elif sys_info["is_linux"]:
        # Проверяем наличие PNG иконки для Linux
        png_icon = Path("docs/images/vsx.png")
        if png_icon.exists():
            cmd.append(f"--linux-icon={png_icon}")
        if sys_info["arch"] in ["arm", "arm64"]:
            cmd.append("--lto=no")  # Отключаем LTO для ARM
    
    # Добавляем опции только если файлы существуют
    if models_dir.exists():
        cmd.append("--include-data-dir=models=models")
    
    cmd.extend([
        "--include-package-data=custom_ocr",
        str(main_file)  # Главный файл В КОНЦЕ команды
    ])
    
    # Добавляем иконку если найдена
    if sys_info["is_windows"] and icon_path.exists():
        cmd.extend([
            f"--include-data-files={icon_path}=docs/images/vsx.ico",
            f"--windows-icon-from-ico={icon_path}"
        ])
    elif sys_info["is_linux"]:
        # Для Linux ищем PNG иконку
        png_icon = icon_path.with_suffix('.png')
        if not png_icon.exists():
            # Ищем в других местах
            for png_file in Path.cwd().rglob("*.png"):
                if "vsx" in png_file.name.lower() or "icon" in png_file.name.lower():
                    png_icon = png_file
                    print(f"✓ Found PNG icon for Linux: {png_icon}")
                    break
        
        if png_icon.exists():
            cmd.extend([
                f"--include-data-files={png_icon}=docs/images/vsx.png"
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
    sys_info = get_system_info()
    
    print("\nRenaming executable...")
    
    # Ищем скомпилированный файл
    dist_dir = Path("dist_gpu")
    
    if not dist_dir.exists():
        print("⚠ Warning: dist_gpu directory not found")
        return
    
    # Определяем расширение для системы
    exe_ext = ".exe" if sys_info["is_windows"] else ""
    
    # Ищем gui.dist или другие варианты
    possible_paths = [
        dist_dir / f"gui.dist/gui{exe_ext}",
        dist_dir / f"gui{exe_ext}",
    ]
    
    for exe_file in possible_paths:
        if exe_file.exists():
            print(f"✓ Found executable: {exe_file}")
            
            new_name = exe_file.with_name(f"VSX-GPU-{sys_info['system']}-{sys_info['arch']}{exe_ext}")
            exe_file.rename(new_name)
            print(f"✓ Renamed to: {new_name}")
            return
    
    # Если не нашли, ищем любой исполняемый файл
    for file in dist_dir.rglob("*"):
        if file.is_file():
            # Проверяем по расширению или правам выполнения
            if (sys_info["is_windows"] and file.suffix == '.exe') or \
               (sys_info["is_linux"] and os.access(file, os.X_OK)):
                print(f"✓ Found executable: {file}")
                
                new_name = file.with_name(f"VSX-GPU-{sys_info['system']}-{sys_info['arch']}{exe_ext}")
                file.rename(new_name)
                print(f"✓ Renamed to: {new_name}")
                return
    
    print("⚠ Warning: No executable file found to rename")


def zip_files_gpu() -> None:
    """Архивирует файлы дистрибутива для GPU."""
    sys_info = get_system_info()
    
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
    
    # Создаем имя архива с информацией о системе
    system_name = sys_info['system'].lower()
    arch_name = sys_info['arch'].lower()
    name = f"VSX-GPU-{system_name}-{arch_name}-v1.0"
    
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
    sys_info = get_system_info()
    
    print("\n" + "="*60)
    print(f"Checking CUDA and GPU availability for {sys_info['system']}...")
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
        if sys_info["is_windows"]:
            result = subprocess.run(['nvidia-smi.exe'], capture_output=True, text=True, shell=True)
        else:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ nvidia-smi is available")
            print(f"GPU Info:\n{result.stdout[:500]}...")
            return True
        else:
            print("⚠ nvidia-smi returned error")
    except FileNotFoundError:
        print("⚠ nvidia-smi not found (CUDA driver not installed?)")
    
    print(f"\n⚠ GPU compilation may not work properly on {sys_info['system']}!")
    print("The application will be compiled with GPU support,")
    print("but may fall back to CPU if CUDA is not available on target system.")
    
    response = input("\nContinue with GPU compilation? (y/n): ")
    return response.lower() == 'y'


def main_gpu() -> None:
    """Основная функция компиляции GPU версии."""
    sys_info = get_system_info()
    
    start_time = perf_counter()
    
    try:
        print("\n" + "="*80)
        print(f"STARTING GPU VERSION COMPILATION FOR {sys_info['system'].upper()} ({sys_info['arch']})")
        print("="*80)
        
        # Проверяем CUDA доступность
        if not check_cuda_availability():
            print("\n❌ GPU compilation cancelled")
            return
        
        # Показываем текущую директорию
        print(f"\n📁 Current directory: {Path.cwd()}")
        print(f"📁 Contents:")
        for item in Path.cwd().iterdir():
            if item.is_dir():
                print(f"  {item.name}/")
            else:
                print(f"  {item.name}")
        
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
        
        print(f"\n💡 Note: The GPU version for {sys_info['system']} requires CUDA and cuDNN")
        print("to be installed on the target system for GPU acceleration to work.")
        
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
