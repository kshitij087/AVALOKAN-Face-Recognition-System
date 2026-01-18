"""
System Diagnostic Tool
Checks all dependencies, camera, Supabase connection, and model loading
Run this if you're having issues
"""

import sys
import os
from datetime import datetime

def print_header(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def print_check(name, status, details=""):
    symbols = {"pass": "✅", "fail": "❌", "warn": "⚠️", "info": "ℹ️"}
    symbol = symbols.get(status, "•")
    print(f"{symbol} {name}")
    if details:
        print(f"   {details}")

def main():
    print_header("🔍 FACE RECOGNITION SYSTEM DIAGNOSTIC")
    print(f"Timestamp: {datetime.now()}")
    print(f"Python: {sys.version}")
    
    # Check 1: Python Version
    print_header("1. Python Environment")
    
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 8:
        print_check("Python Version", "pass", f"Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    else:
        print_check("Python Version", "fail", f"Need Python 3.8+, got {version_info.major}.{version_info.minor}")
    
    # Check virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print_check("Virtual Environment", "pass", "Running in venv ✓")
    else:
        print_check("Virtual Environment", "warn", "Not in venv - might cause issues")
    
    # Check 2: Required Packages
    print_header("2. Required Packages")
    
    required_packages = {
        'cv2': 'opencv-python',
        'deepface': 'deepface',
        'supabase': 'supabase',
        'tensorflow': 'tensorflow',
        'dotenv': 'python-dotenv',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'dotenv':
                __import__('dotenv')
            else:
                __import__(module_name)
            
            # Get version if possible
            try:
                if module_name == 'cv2':
                    import cv2
                    version = cv2.__version__
                elif module_name == 'tensorflow':
                    import tensorflow as tf
                    version = tf.__version__
                else:
                    mod = __import__(module_name)
                    version = getattr(mod, '__version__', 'unknown')
                
                print_check(package_name, "pass", f"Version: {version}")
            except:
                print_check(package_name, "pass", "Installed")
                
        except ImportError:
            print_check(package_name, "fail", f"NOT INSTALLED - Run: pip install {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\n⚠️  INSTALL MISSING PACKAGES:")
        print(f"pip install {' '.join(missing_packages)}")
    
    # Check 3: DeepFace Models
    print_header("3. DeepFace Models")
    
    try:
        from deepface import DeepFace
        print_check("DeepFace Import", "pass")
        
        # Try to load model
        print("\n   Testing model loading (this may take a minute first time)...")
        try:
            import numpy as np
            test_img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            result = DeepFace.represent(
                img_path=test_img,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="skip"
            )
            print_check("Facenet512 Model", "pass", "Loaded successfully")
        except Exception as e:
            print_check("Facenet512 Model", "fail", f"Error: {str(e)[:100]}")
        
        # Try detector
        try:
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            faces = DeepFace.extract_faces(
                img_path=test_img,
                enforce_detection=False,
                detector_backend="retinaface"
            )
            print_check("RetinaFace Detector", "pass", "Working")
        except Exception as e:
            print_check("RetinaFace Detector", "fail", f"Error: {str(e)[:100]}")
            
    except ImportError:
        print_check("DeepFace", "fail", "Package not installed")
    except Exception as e:
        print_check("DeepFace", "fail", str(e))
    
    # Check 4: Camera
    print_header("4. Camera Access")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print_check("Camera Detection", "pass", f"Resolution: {w}x{h}")
                
                # Check frame rate
                fps = cap.get(cv2.CAP_PROP_FPS)
                print_check("Camera FPS", "info", f"{fps:.1f} FPS")
            else:
                print_check("Camera Frame", "fail", "Can't read frames")
            cap.release()
        else:
            print_check("Camera Detection", "fail", "No camera found")
            print("   • Check if camera is connected")
            print("   • Check if another app is using it")
            print("   • Try index 1: cv2.VideoCapture(1)")
            
    except Exception as e:
        print_check("Camera Test", "fail", str(e))
    
    # Check 5: Supabase Configuration
    print_header("5. Supabase Configuration")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or "YOUR_" in url:
            print_check(".env File", "fail", "SUPABASE_URL not configured")
            print("   • Create .env file with your Supabase credentials")
        elif not key or "YOUR_" in key:
            print_check(".env File", "fail", "SUPABASE_KEY not configured")
        else:
            print_check(".env File", "pass", "Credentials found")
            
            # Test connection
            try:
                from supabase import create_client
                client = create_client(url, key)
                
                # Test table access
                try:
                    result = client.table('detected_faces').select("*").limit(1).execute()
                    print_check("Supabase Connection", "pass", "Connected to database")
                    print_check("Table Access", "pass", f"'detected_faces' table accessible")
                except Exception as e:
                    error_str = str(e).lower()
                    if 'relation' in error_str or 'does not exist' in error_str:
                        print_check("Table Access", "fail", "Table 'detected_faces' not found")
                        print("   • Run the SQL setup script in Supabase SQL Editor")
                    else:
                        print_check("Table Access", "fail", str(e)[:100])
                
                # Test storage bucket
                try:
                    buckets = client.storage.list_buckets()
                    bucket_names = [b['name'] for b in buckets]
                    if 'face_references' in bucket_names:
                        print_check("Storage Bucket", "pass", "'face_references' bucket exists")
                    else:
                        print_check("Storage Bucket", "fail", "'face_references' bucket not found")
                        print("   • Create bucket named 'face_references' in Supabase Storage")
                        print("   • Make it public")
                except Exception as e:
                    print_check("Storage Bucket", "warn", f"Can't check: {str(e)[:100]}")
                    
            except Exception as e:
                print_check("Supabase Connection", "fail", str(e)[:200])
                
    except ImportError:
        print_check("python-dotenv", "fail", "Package not installed")
    except Exception as e:
        print_check("Supabase Test", "fail", str(e))
    
    # Check 6: File System
    print_header("6. File System")
    
    directories = ['db_faces', 'temp_faces']
    for dir_name in directories:
        if os.path.exists(dir_name):
            file_count = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
            print_check(f"{dir_name}/", "pass", f"{file_count} files")
        else:
            print_check(f"{dir_name}/", "info", "Will be created on first run")
    
    # Check main.py
    if os.path.exists('main.py'):
        print_check("main.py", "pass", "Found")
    else:
        print_check("main.py", "fail", "Not found in current directory")
    
    # Check 7: System Resources
    print_header("7. System Resources")
    
    try:
        import psutil
        
        cpu_count = psutil.cpu_count()
        print_check("CPU Cores", "info", f"{cpu_count} cores")
        
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print_check("RAM", "info", f"{memory_gb:.1f} GB total")
        
        if memory_gb < 4:
            print("   ⚠️  Low RAM - consider increasing FRAME_SKIP")
            
    except ImportError:
        print_check("Resource Check", "info", "Install psutil for resource info")
    
    # Check 8: GPU Availability
    print_header("8. GPU Support")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print_check("GPU Detection", "pass", f"Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"   • GPU {i}: {gpu.name}")
        else:
            print_check("GPU Detection", "info", "No GPU found (CPU mode)")
            print("   • System will work but may be slower")
            print("   • For better performance, install CUDA and cuDNN")
            
    except Exception as e:
        print_check("GPU Check", "info", "Unable to check GPU")
    
    # Final Summary
    print_header("📋 SUMMARY & RECOMMENDATIONS")
    
    print("\n✅ READY TO USE:")
    print("   1. All dependencies installed")
    print("   2. Camera accessible")
    print("   3. Supabase configured")
    print("   4. Models can load")
    print("\n🚀 Next Steps:")
    print("   • Run: python main.py")
    print("   • Test with tune.py: python tune.py image1.jpg image2.jpg")
    print("\n📖 Documentation:")
    print("   • See setup guide for configuration options")
    print("   • Use 'r' key to reload database while running")
    print("   • Use 'q' key to quit")
    
    print("\n" + "="*70)
    print("Diagnostic Complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Diagnostic failed with error:")
        print(f"{e}")
        import traceback
        traceback.print_exc()