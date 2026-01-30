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


def install_requirements_cpu() -> None:
    """Устанавливает требования для CPU версии."""
    print("\n" + "="*60)
    print("Installing CPU requirements...")
    print("="*60)
    
    # Удаляем GPU версию если установлена
    print("\nChecking for GPU packages...")
    try:
        subprocess.run(["pip", "uninstall", "-y", "onnxruntime-gpu"], 
                      check=False, capture_output=True)
        print("✓ Removed onnxruntime-gpu (if existed)")
    except:
        print("✓ onnxruntime-gpu not installed")
    
    # Устанавливаем CPU требования
    req_file = Path("requirements-cpu.txt")
    if req_file.exists():
        print(f"\nFound requirements file: {req_file}")
        run_command(['pip', 'install', '-r', 'requirements-cpu.txt'])
        print("✓ Installed CPU requirements")
    else:
        print("⚠ Warning: requirements-cpu.txt not found")
        print("Trying requirements.txt...")
        alt_req_file = Path("requirements.txt")
        if alt_req_file.exists():
            run_command(['pip', 'install', '-r', 'requirements.txt'])
    
    # Устанавливаем нужные пакеты
    print("\nInstalling required packages...")
    run_command(["pip", "install", "onnxruntime"])
    run_command(["pip", "install", "Nuitka==2.8.1"])
    print("✓ All packages installed")


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


def download_all_models() -> None:
    """Загружает все модели OCR."""
    try:
        print("\n" + "="*60)
        print("Downloading models...")
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
        
        utils.CONFIG.ocr_opts["use_gpu"] = False
        
        print("\nDownloading text line orientation models...")
        for model in txt_line_ori_models:
            print(f"\nChecking for {model} model...")
            _ = CustomPaddleOCR(textline_orientation_model_name=model, **utils.CONFIG.ocr_opts)
            print("-" * 80)
        
        print("\nDownloading detection models...")
        for model in det_models:
            print(f"\nChecking for {model} model...")
            _ = CustomPaddleOCR(text_detection_model_name=model, **utils.CONFIG.ocr_opts)
            print("-" * 80)
        
        print("\nDownloading recognition models...")
        for model in rec_models:
            print(f"\nChecking for {model} model...")
            _ = CustomPaddleOCR(text_recognition_model_name=model, **utils.CONFIG.ocr_opts)
            print("-" * 80)
            
        print("\n✓ All models downloaded successfully")
        
    except ImportError as e:
        print(f"⚠ Error importing modules: {e}")
        print("Trying to install missing dependencies...")
        # Пробуем установить paddlepaddle если его нет
        try:
            run_command(["pip", "install", "paddlepaddle"])
            print("✓ Installed paddlepaddle, retrying...")
            download_all_models()  # Рекурсивный вызов
        except:
            print("❌ Could not install dependencies automatically")
            raise
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


def compile_program_cpu() -> None:
    """Компилирует программу с Nuitka для CPU."""
    print("\n" + "="*60)
    print("Compiling with Nuitka for CPU...")
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
        "--jobs=10",
        "--output-dir=dist_cpu",
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
        print("✓ Compilation completed successfully")
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
            "--output-dir=dist_cpu",
            str(main_file)
        ]
        try:
            run_command(simple_cmd, use_shell=False)
            print("✓ Simplified compilation succeeded")
        except subprocess.CalledProcessError as e2:
            print(f"❌ Simplified compilation also failed: {e2}")
            raise


def rename_exe_cpu() -> None:
    """Переименовывает исполняемый файл для CPU версии."""
    print("\nRenaming executable...")
    
    # Ищем скомпилированный файл
    dist_dir = Path("dist_cpu")
    
    if not dist_dir.exists():
        print("⚠ Warning: dist_cpu directory not found")
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
                new_name = exe_file.with_name("VSX-CPU")
            else:
                new_name = exe_file.with_name("VSX-CPU.exe")
            
            exe_file.rename(new_name)
            print(f"✓ Renamed to: {new_name}")
            return
    
    # Если не нашли, ищем любой исполняемый файл
    for file in dist_dir.rglob("*"):
        if file.is_file() and (file.suffix == '.exe' or file.stat().st_mode & 0o111):
            print(f"✓ Found executable: {file}")
            
            if platform.system() != "Windows":
                new_name = file.with_name("VSX-CPU")
            else:
                new_name = file.with_name("VSX-CPU.exe")
            
            file.rename(new_name)
            print(f"✓ Renamed to: {new_name}")
            return
    
    print("⚠ Warning: No executable file found to rename")


def zip_files_cpu() -> None:
    """Архивирует файлы дистрибутива для CPU."""
    print("\n" + "="*60)
    print("Creating distribution archive...")
    print("="*60)
    
    # Ищем директорию дистрибутива
    dist_dirs = [
        Path("dist_cpu/gui.dist"),
        Path("dist_cpu"),
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
    name = f"VSX-{system}-CPU-v1.0"
    
    # Удаляем старый архив если существует
    old_zip = Path(f"{name}.zip")
    if old_zip.exists():
        old_zip.unlink()
        print("✓ Removed old archive")
    
    # Создаем новый архив
    try:
        shutil.make_archive(name, "zip", dist_dir)
        print(f"✓ Created archive: {name}.zip")
        
        zip_size = Path(f"{name}.zip").stat().st_size
        print(f"✓ Archive size: {zip_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ Error creating archive: {e}")


def delete_dist_dir_cpu() -> None:
    """Удаляет временную директорию дистрибутива."""
    print("\nCleaning up...")
    
    dist_dir = Path("dist_cpu")
    if dist_dir.exists():
        try:
            shutil.rmtree(dist_dir, ignore_errors=True)
            print("✓ Removed distribution directory")
        except Exception as e:
            print(f"⚠ Could not remove directory: {e}")
    else:
        print("✓ No distribution directory to clean")


def main_cpu() -> None:
    """Основная функция компиляции CPU версии."""
    start_time = perf_counter()
    
    try:
        print("\n" + "="*80)
        print("STARTING CPU VERSION COMPILATION")
        print("="*80)
        
        # Показываем текущую директорию
        print(f"\n📁 Current directory: {Path.cwd()}")
        print(f"📁 Contents:")
        for item in Path.cwd().iterdir():
            print(f"  {item.name}{'/' if item.is_dir() else ''}")
        
        # 1. Установка зависимостей
        install_requirements_cpu()
        
        # 2. Загрузка моделей
        try:
            download_all_models()
        except Exception as e:
            print(f"⚠ Model download failed, continuing: {e}")
        
        # 3. Очистка моделей
        try:
            remove_non_onnx_models()
        except Exception as e:
            print(f"⚠ Model cleanup failed, continuing: {e}")
        
        # 4. Компиляция
        compile_program_cpu()
        
        # 5. Переименование EXE
        rename_exe_cpu()
        
        # 6. Архивирование
        zip_files_cpu()
        
        # 7. Очистка
        delete_dist_dir_cpu()
        
        duration = timedelta(seconds=round(perf_counter() - start_time))
        print("\n" + "="*80)
        print(f"✅ CPU COMPILATION COMPLETED SUCCESSFULLY")
        print(f"⏱️  Total duration: {duration}")
        print("="*80)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error output: {e.stderr}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"Output: {e.stdout}")
        raise
    except Exception as e:
        print(f"\n❌ Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main_cpu()