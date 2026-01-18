# check_env.py (Version 2 - Modern and More Thorough)
import sys
import importlib.metadata
import os

print(f"--- Diagnostic for Face Recognition System ---")
print(f"Timestamp: {__import__('datetime').datetime.now(__import__('datetime').timezone.utc)}\n")

# --- 1. Python Executable Verification ---
print("--- 1. Python Executable ---")
py_executable = sys.executable
print(f"This script is being run by: {py_executable}")

if "venv" not in py_executable:
    print("!! CRITICAL WARNING: You are NOT running from the 'venv' virtual environment. !!")
    print("!! This is the most likely cause of your problem. Please use the VS Code interpreter selector. !!\n")
else:
    print("OK: You are running from a Python interpreter inside the 'venv' folder.\n")

# --- 2. Installed Package Listing ---
print("--- 2. Installed Packages in this Environment ---")
try:
    installed_packages = sorted([f"{dist.name}=={dist.version}" for dist in importlib.metadata.distributions()])
    for package in installed_packages:
        print(package)
except Exception as e:
    print(f"Could not list packages. Error: {e}")
print("\n--- 3. Core Library Test ---")

# --- 3. Actively Test Face Recognition and its Models ---
try:
    import face_recognition
    print("OK: 'face_recognition' library imported successfully.")

    # The ultimate test: try to use a function that REQUIRES the models.
    # The library ships with example images we can use for this test.
    print("Attempting to use a function that requires the model files...")
    examples_folder = os.path.join(os.path.dirname(face_recognition.__file__), "examples")
    sample_image_path = os.path.join(examples_folder, "biden.jpg")

    if not os.path.exists(sample_image_path):
         print("!! WARNING: Could not find the sample image for testing. This might indicate a broken installation. !!")
    else:
        known_image = face_recognition.load_image_file(sample_image_path)
        _ = face_recognition.face_encodings(known_image)[0]
        print("SUCCESS: The 'face_recognition' library successfully loaded and used its model files.")

except ImportError:
    print("!! ERROR: 'face_recognition' IS NOT INSTALLED in this environment. !!")
except FileNotFoundError:
    print("!! CRITICAL ERROR: Could not find the model files when needed. !!")
    print("!! This confirms 'face_recognition_models' is not installed or not found in this environment. !!")
except Exception as e:
    print(f"!! An unexpected error occurred during the library test: {e} !!")