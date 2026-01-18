"""
Face Similarity Tuning Tool
Use this to test face matching and find optimal threshold
Usage: python tune.py <image1_path> <image2_path>
"""

import sys
import cv2
import numpy as np
from deepface import DeepFace

# Must match main.py configuration
MODEL_NAME = "Facenet512"
DISTANCE_METRIC = "cosine"
DETECTOR_BACKEND = "retinaface"

def analyze_image(img_path):
    """Analyze a single image for quality"""
    print(f"\n📸 Analyzing: {img_path}")
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            print("   ❌ Cannot read image")
            return False
        
        # Detect faces
        faces = DeepFace.extract_faces(
            img_path=img,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND,
            align=True
        )
        
        if not faces:
            print("   ❌ No faces detected")
            return False
        
        face = faces[0]
        print(f"   ✅ Face detected")
        print(f"   • Confidence: {face.get('confidence', 0):.2%}")
        print(f"   • Size: {face['facial_area']['w']}x{face['facial_area']['h']} pixels")
        
        # Quality checks
        face_img = face['face']
        if len(face_img.shape) == 2:
            gray = face_img
        else:
            gray = cv2.cvtColor((face_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"   • Sharpness: {laplacian:.1f} {'✅' if laplacian > 100 else '⚠️ (low)'}")
        
        # Brightness
        brightness = np.mean(gray)
        print(f"   • Brightness: {brightness:.1f} {'✅' if 40 < brightness < 220 else '⚠️'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def compare_faces(img1_path, img2_path):
    """Compare two face images and calculate similarity"""
    
    print("\n" + "="*60)
    print("🔍 FACE SIMILARITY ANALYSIS")
    print("="*60)
    
    # Analyze both images first
    if not analyze_image(img1_path):
        return
    if not analyze_image(img2_path):
        return
    
    print("\n" + "-"*60)
    print("📊 COMPARING FACES")
    print("-"*60)
    
    try:
        # Method 1: Using DeepFace.verify
        print("\n🔄 Running comparison...")
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        
        distance = result['distance']
        verified = result['verified']
        threshold = result['threshold']
        
        print(f"\n{'='*60}")
        print("📈 RESULTS")
        print(f"{'='*60}")
        print(f"Distance Score: {distance:.4f}")
        print(f"DeepFace Threshold: {threshold:.4f}")
        print(f"DeepFace Match: {'✅ YES' if verified else '❌ NO'}")
        
        print(f"\n{'='*60}")
        print("💡 RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if distance < 0.30:
            print("🎯 Excellent match! These faces are very similar.")
            print(f"   Recommended threshold: 0.35 - 0.40")
        elif distance < 0.40:
            print("✅ Good match! These faces are similar.")
            print(f"   Recommended threshold: 0.40 - 0.45")
        elif distance < 0.50:
            print("⚠️  Borderline match. Could be same person in different conditions.")
            print(f"   Recommended threshold: 0.45 - 0.50 (may cause false positives)")
        else:
            print("❌ No match. These appear to be different people.")
            print(f"   Current distance ({distance:.4f}) is above recommended threshold.")
        
        print(f"\n{'='*60}")
        print("🔧 TUNING GUIDE")
        print(f"{'='*60}")
        print(f"• Lower threshold (0.30-0.35): Strict matching, fewer false positives")
        print(f"• Medium threshold (0.35-0.45): Balanced, recommended for most cases")
        print(f"• Higher threshold (0.45-0.55): Loose matching, more false positives")
        print(f"\n• Current system threshold: 0.40")
        print(f"• For these images: {'✅ Would match' if distance < 0.40 else '❌ Would not match'}")
        
        # Additional tests with different thresholds
        print(f"\n{'='*60}")
        print("🧪 THRESHOLD SIMULATION")
        print(f"{'='*60}")
        
        for test_threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
            match = "✅ MATCH" if distance < test_threshold else "❌ NO MATCH"
            current = " ← CURRENT" if test_threshold == 0.40 else ""
            print(f"Threshold {test_threshold:.2f}: {match}{current}")
        
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        print("\nPossible issues:")
        print("• One or both images don't contain a clear face")
        print("• Images are corrupted or unreadable")
        print("• Face detector couldn't locate faces")
        print("\nTry with different images or check image quality.")

def batch_compare(image_folder):
    """Compare all images in a folder against each other"""
    import os
    
    print(f"\n🔍 Scanning folder: {image_folder}")
    
    images = [f for f in os.listdir(image_folder) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) < 2:
        print("❌ Need at least 2 images in folder")
        return
    
    print(f"Found {len(images)} images")
    print("\n" + "="*60)
    print("📊 BATCH COMPARISON RESULTS")
    print("="*60)
    
    results = []
    
    for i, img1 in enumerate(images):
        for img2 in images[i+1:]:
            path1 = os.path.join(image_folder, img1)
            path2 = os.path.join(image_folder, img2)
            
            try:
                result = DeepFace.verify(
                    img1_path=path1,
                    img2_path=path2,
                    model_name=MODEL_NAME,
                    distance_metric=DISTANCE_METRIC,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=True
                )
                
                distance = result['distance']
                results.append({
                    'img1': img1,
                    'img2': img2,
                    'distance': distance
                })
                
                match = "✅" if distance < 0.40 else "❌"
                print(f"{match} {img1[:20]:20} ↔ {img2[:20]:20} : {distance:.4f}")
                
            except Exception as e:
                print(f"⚠️  {img1[:20]:20} ↔ {img2[:20]:20} : ERROR")
    
    if results:
        print(f"\n{'='*60}")
        print("📈 STATISTICS")
        print(f"{'='*60}")
        
        distances = [r['distance'] for r in results]
        print(f"Minimum distance: {min(distances):.4f}")
        print(f"Maximum distance: {max(distances):.4f}")
        print(f"Average distance: {sum(distances)/len(distances):.4f}")
        print(f"Median distance: {sorted(distances)[len(distances)//2]:.4f}")
        
        # Threshold analysis
        print(f"\n{'='*60}")
        print("🎯 THRESHOLD ANALYSIS")
        print(f"{'='*60}")
        
        for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
            matches = sum(1 for d in distances if d < threshold)
            percentage = (matches / len(distances)) * 100
            current = " ← CURRENT" if threshold == 0.40 else ""
            print(f"Threshold {threshold:.2f}: {matches}/{len(distances)} matches ({percentage:.1f}%){current}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("🔧 FACE SIMILARITY TUNING TOOL")
        print("="*60)
        print("\nUsage:")
        print("  Compare two images:")
        print("    python tune.py <image1.jpg> <image2.jpg>")
        print("\n  Batch compare folder:")
        print("    python tune.py --batch <folder_path>")
        print("\nExamples:")
        print("  python tune.py person1_photo1.jpg person1_photo2.jpg")
        print("  python tune.py --batch ./test_faces/")
        print()
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("❌ Please provide folder path")
            print("Usage: python tune.py --batch <folder_path>")
            sys.exit(1)
        batch_compare(sys.argv[2])
    else:
        if len(sys.argv) < 3:
            print("❌ Please provide two image paths")
            print("Usage: python tune.py <image1> <image2>")
            sys.exit(1)
        compare_faces(sys.argv[1], sys.argv[2])