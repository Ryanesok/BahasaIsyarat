"""
Dependency Installer for Sign Language Reader
Automatically installs all required Python packages
"""

import subprocess
import sys
import os

def print_header():
    print("=" * 60)
    print("  Sign Language Reader - Dependency Installer")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("[1/4] Checking Python version...")
    version = sys.version_info
    print(f"      Python {version.major}.{version.minor}.{version.micro} detected")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[ERROR] Python 3.8 or higher is required!")
        print("        Please upgrade Python and try again.")
        sys.exit(1)
    
    print("[OK] Python version is compatible")
    print()

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("[2/4] Upgrading pip...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[OK] pip upgraded successfully")
    except subprocess.CalledProcessError:
        print("[WARN] Could not upgrade pip, continuing anyway...")
    print()

def install_dependencies():
    """Install all required packages from requirements.txt"""
    print("[3/4] Installing dependencies...")
    
    # Check if requirements.txt exists
    req_file = os.path.join("built-in", "requirements.txt")
    if not os.path.exists(req_file):
        print(f"[ERROR] {req_file} not found!")
        sys.exit(1)
    
    # Read requirements
    with open(req_file, 'r') as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.strip().startswith('#')]
    
    print(f"      Found {len(requirements)} packages to install")
    print()
    
    # Install each package
    failed_packages = []
    for i, package in enumerate(requirements, 1):
        package_name = package.split('>=')[0].split('==')[0]
        print(f"      [{i}/{len(requirements)}] Installing {package_name}...", end=" ")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[OK]")
        except subprocess.CalledProcessError:
            print("[FAILED]")
            failed_packages.append(package_name)
    
    print()
    
    if failed_packages:
        print("[WARN] Some packages failed to install:")
        for pkg in failed_packages:
            print(f"        - {pkg}")
        print()
        print("      You may need to install them manually:")
        for pkg in failed_packages:
            print(f"        pip install {pkg}")
        print()
    else:
        print("[OK] All dependencies installed successfully")
    
    print()

def verify_installation():
    """Verify that key packages can be imported"""
    print("[4/4] Verifying installation...")
    
    packages_to_test = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'numpy': 'numpy',
        'paho.mqtt.client': 'paho-mqtt'
    }
    
    failed_imports = []
    
    for import_name, package_name in packages_to_test.items():
        try:
            __import__(import_name)
            print(f"      [OK] {package_name}")
        except ImportError:
            print(f"      [FAILED] {package_name}")
            failed_imports.append(package_name)
    
    print()
    
    if failed_imports:
        print("[ERROR] Some packages could not be imported:")
        for pkg in failed_imports:
            print(f"        - {pkg}")
        print()
        print("      Please install them manually and try again.")
        return False
    else:
        print("[OK] All packages verified successfully")
        return True

def print_summary(success):
    """Print installation summary"""
    print()
    print("=" * 60)
    if success:
        print("  Installation Complete!")
        print("=" * 60)
        print()
        print("  You can now run the application:")
        print("    python desktop_app.py")
        print()
        print("  Or test the sensor fusion:")
        print("    python test_sensor_fusion.py")
    else:
        print("  Installation Failed")
        print("=" * 60)
        print()
        print("  Please fix the errors above and run this script again.")
    print()

def main():
    try:
        print_header()
        check_python_version()
        upgrade_pip()
        install_dependencies()
        success = verify_installation()
        print_summary(success)
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print()
        print()
        print("[CANCELLED] Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
