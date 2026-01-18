"""
Optimized Face Recognition System with Robust Tracking
Features: Face quality checks, proper tracking, duplicate prevention, reliable cloud sync
"""

import cv2
import os
from dotenv import load_dotenv
import uuid
from supabase import create_client, Client
from datetime import datetime, timezone
from deepface import DeepFace
import time
import numpy as np
import traceback
from collections import defaultdict
import hashlib

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
DB_PATH = "db_faces"
TEMP_PATH = "temp_faces"
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)

# Model Configuration
MODEL_NAME = "Facenet512"  # Best balance of speed and accuracy
DISTANCE_METRIC = "cosine"
DETECTOR_BACKEND = "mtcnn"  # Most reliable detector
DISTANCE_THRESHOLD = 0.40  # Tuned for Facenet512

# Performance Settings
FRAME_SKIP = 5 # Process every 2nd frame
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PROCESS_WIDTH = 324  # Resize for processing

# Face Quality Thresholds
MIN_FACE_SIZE = 80  # Minimum pixels
MAX_FACE_SIZE = 800  # Maximum before resize
MIN_FACE_CONFIDENCE = 0.70  # High confidence only
MIN_SHARPNESS = 60  # Laplacian variance threshold
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

# Tracking Configuration
TRACKING_DISTANCE_THRESHOLD = 180  # Pixels to consider same face
CONFIRMATION_FRAMES = 5  # Frames needed to confirm new face
RECOGNITION_COOLDOWN = 7 # Seconds between DB updates
TRACKING_TIMEOUT = 2.0  # Seconds before losing track

# Memory Management
MAX_TRACKED_FACES = 5
MAX_EMBEDDINGS_PER_PERSON = 3  # Store multiple embeddings per person

# ============================================================================
# SUPABASE SETUP
# ============================================================================

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or "YOUR_" in SUPABASE_URL:
    print("❌ ERROR: Configure SUPABASE_URL and SUPABASE_KEY in .env file")
    exit(1)

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase connected")
except Exception as e:
    print(f"❌ Supabase connection failed: {e}")
    exit(1)

# ============================================================================
# FACE QUALITY ASSESSMENT
# ============================================================================

def assess_face_quality(face_img):
    """Check if face image meets quality standards"""
    if face_img is None or face_img.size == 0:
        return False, "Empty image"
    
    h, w = face_img.shape[:2]
    
    # Size check
    if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
        return False, f"Too small ({w}x{h})"
    
    # Aspect ratio check (faces should be roughly square-ish)
    aspect_ratio = w / h
    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
        return False, f"Bad aspect ratio ({aspect_ratio:.2f})"
    
    # Sharpness check (detect blur)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < MIN_SHARPNESS:
        return False, f"Too blurry ({laplacian_var:.1f})"
    
    # Brightness check
    brightness = np.mean(gray)
    if brightness < MIN_BRIGHTNESS:
        return False, f"Too dark ({brightness:.1f})"
    if brightness > MAX_BRIGHTNESS:
        return False, f"Too bright ({brightness:.1f})"
    
    return True, "OK"

# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def generate_embedding(face_img, retry=True):
    """Generate face embedding with error handling"""
    try:
        # Resize if too large
        h, w = face_img.shape[:2]
        if h > MAX_FACE_SIZE or w > MAX_FACE_SIZE:
            scale = MAX_FACE_SIZE / max(h, w)
            face_img = cv2.resize(face_img, None, fx=scale, fy=scale)
        
        # Enhance contrast
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        result = DeepFace.represent(
            img_path=enhanced,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="opencv"  # We already have the face
        )
        
        return np.array(result[0]["embedding"])
        
    except Exception as e:
        if retry:
            # Try with original image
            try:
                result = DeepFace.represent(
                    img_path=face_img,
                    model_name="Facenet",
                    enforce_detection=False,
                    detector_backend="skip"
                )
                return np.array(result[0]["embedding"])
            except:
                pass
        print(f"⚠️ Embedding failed: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity (higher = more similar)"""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product > 0 else 0

def cosine_distance(vec1, vec2):
    """Calculate cosine distance (lower = more similar)"""
    return 1 - cosine_similarity(vec1, vec2)

# ============================================================================
# FACE DATABASE
# ============================================================================

class FaceDatabase:
    def __init__(self):
        self.faces = {}  # unique_id -> list of embeddings
        self.metadata = {}  # unique_id -> metadata
        self.load_from_disk()
    
    def load_from_disk(self):
        """Load all faces from local storage"""
        print("📂 Loading face database...")
        count = 0
        
        for filename in os.listdir(DB_PATH):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            unique_id = os.path.splitext(filename)[0]
            img_path = os.path.join(DB_PATH, filename)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                embedding = generate_embedding(img, retry=False)
                if embedding is not None:
                    self.faces[unique_id] = [embedding]
                    self.metadata[unique_id] = {
                        'path': img_path,
                        'loaded_at': time.time()
                    }
                    count += 1
            except Exception as e:
                print(f"⚠️ Failed to load {filename}: {e}")
        
        print(f"✅ Loaded {count} faces")
        return count
    
    def find_match(self, embedding):
        """Find best matching face"""
        if not self.faces:
            return None, 1.0
        
        best_id = None
        best_distance = 1.0
        
        for unique_id, embeddings in self.faces.items():
            # Compare against all embeddings for this person
            for emb in embeddings:
                dist = cosine_distance(embedding, emb)
                if dist < best_distance:
                    best_distance = dist
                    best_id = unique_id
        
        return best_id, best_distance
    
    def add_face(self, unique_id, embedding, face_img):
        """Add new face to database"""
        # Initialize or append embedding
        if unique_id not in self.faces:
            self.faces[unique_id] = []
        
        self.faces[unique_id].append(embedding)
        
        # Keep only best N embeddings
        if len(self.faces[unique_id]) > MAX_EMBEDDINGS_PER_PERSON:
            self.faces[unique_id] = self.faces[unique_id][-MAX_EMBEDDINGS_PER_PERSON:]
        
        # Save to disk
        face_path = os.path.join(DB_PATH, f"{unique_id}.jpg")
        cv2.imwrite(face_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        self.metadata[unique_id] = {
            'path': face_path,
            'loaded_at': time.time()
        }

# ============================================================================
# FACE TRACKING
# ============================================================================

class TrackedFace:
    def __init__(self, bbox, face_img, embedding):
        
        self.bbox = bbox  # (x, y, w, h)
        self.face_img = face_img
        self.embedding = embedding
        self.unique_id = None
        self.confirmed = False
        self.confirmation_count = 0
        self.last_seen = time.time()
        self.last_db_update = 0
        
    def update_position(self, bbox, face_img):
        """Update tracking position"""
        self.bbox = bbox
        self.face_img = face_img
        self.last_seen = time.time()
        self.confirmation_count += 1
        
        if self.confirmation_count >= CONFIRMATION_FRAMES:
            self.confirmed = True
    
    def distance_to(self, other_bbox):
        """Calculate distance to another bounding box"""
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other_bbox
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def is_active(self):
        """Check if still being tracked"""
        return (time.time() - self.last_seen) < TRACKING_TIMEOUT
    
    def can_update_db(self):
        """Check if enough time passed for DB update"""
        return (time.time() - self.last_db_update) > RECOGNITION_COOLDOWN

# ============================================================================
# FACE ENROLLMENT
# ============================================================================

def enroll_face(face_img, db):
    """Enroll a new face to database and cloud"""
    unique_id = str(uuid.uuid4())
    
    # Generate embedding
    embedding = generate_embedding(face_img)
    if embedding is None:
        print("❌ Failed to generate embedding for enrollment")
        return None
    
    # Add to local database
    db.add_face(unique_id, embedding, face_img)
    
    print(f"🆕 Enrolling new face: {unique_id[:8]}")
    
    # Upload to Supabase
    try:
        # Prepare image
        face_path = os.path.join(DB_PATH, f"{unique_id}.jpg")
        with open(face_path, 'rb') as f:
            image_bytes = f.read()
        
        # Upload to storage
        file_path = f"{unique_id}.jpg"
        supabase.storage.from_('face_references').upload(
            file_path, 
            image_bytes, 
            {"content-type": "image/jpeg", "upsert": "false"}
        )
        
        image_url = supabase.storage.from_('face_references').get_public_url(file_path)
        
        # Insert to database
        utc_now = datetime.now(timezone.utc).isoformat()
        supabase.table('detected_faces').insert({
            'unique_id': unique_id,
            'first_seen': utc_now,
            'last_seen': utc_now,
            'appearance_count': 1,
            'image_url': image_url
        }).execute()
        
        print(f"   ✅ Uploaded to Supabase")
        return unique_id
        
    except Exception as e:
        print(f"   ⚠️ Cloud upload failed: {e}")
        return unique_id  # Still return ID, local enrollment succeeded

def update_face_count(unique_id):
    """Update face appearance count in Supabase"""
    try:
        # Get current count
        result = supabase.table('detected_faces').select(
            'appearance_count'
        ).eq('unique_id', unique_id).execute()
        
        if not result.data:
            print(f"   ⚠️ Face {unique_id[:8]} not in cloud DB")
            return False
        
        # Update count
        new_count = result.data[0]['appearance_count'] + 1
        supabase.table('detected_faces').update({
            'appearance_count': new_count,
            'last_seen': datetime.now(timezone.utc).isoformat()
        }).eq('unique_id', unique_id).execute()
        
        print(f"   📊 Updated {unique_id[:8]} - Count: {new_count}")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Update failed: {e}")
        return False

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🚀 FACE RECOGNITION SYSTEM - PRODUCTION VERSION")
    print("="*70)
    
    # Initialize database
    db = FaceDatabase()
    
    # Setup camera
    print("\n📹 Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    time.sleep(1.0)
    
    if not cap.isOpened():
        print("❌ Camera not available")
        return
    
    # Tracking state
    tracked_faces = []
    frame_count = 0
    fps_list = []
    last_time = time.time()
    
    print("\n" + "="*70)
    print("✅ SYSTEM ACTIVE - Press 'q' to quit, 'r' to reload database")
    print("="*70 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time) if current_time != last_time else 0
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            last_time = current_time
            
            display = frame.copy()
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % FRAME_SKIP != 0:
                # Draw existing tracked faces
                for tf in tracked_faces:
                    if tf.is_active():
                        x, y, w, h = tf.bbox
                        color = (0, 255, 0) if tf.unique_id else (0, 165, 255)
                        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                        label = tf.unique_id[:8] if tf.unique_id else f"Tracking ({tf.confirmation_count}/{CONFIRMATION_FRAMES})"
                        cv2.putText(display, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Status
                cv2.putText(display, f"FPS: {avg_fps:.1f} | Known: {len(db.faces)} | Tracked: {len(tracked_faces)}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Face Recognition', display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Resize for processing
            scale = PROCESS_WIDTH / frame.shape[1]
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            
            # Detect faces
            try:
                faces = DeepFace.extract_faces(
                    img_path=small,
                    enforce_detection=False,
                    detector_backend=DETECTOR_BACKEND,
                    align=True
                )
                
                current_detections = []
                
                for face_data in faces:
                    confidence = face_data.get("confidence", 0)
                    if confidence < MIN_FACE_CONFIDENCE:
                        continue
                    
                    # Scale back to original
                    fa = face_data['facial_area']
                    x = int(fa['x'] / scale)
                    y = int(fa['y'] / scale)
                    w = int(fa['w'] / scale)
                    h = int(fa['h'] / scale)
                    
                    # Extract face
                    face_img = frame[max(0,y):min(frame.shape[0],y+h), 
                                   max(0,x):min(frame.shape[1],x+w)]
                    
                    if face_img.size == 0:
                        continue
                    
                    # Quality check
                    is_good, reason = assess_face_quality(face_img)
                    if not is_good:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 1)
                        cv2.putText(display, reason, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        continue
                    
                    # Generate embedding
                    embedding = generate_embedding(face_img)
                    if embedding is None:
                        continue
                    
                    current_detections.append({
                        'bbox': (x, y, w, h),
                        'face_img': face_img,
                        'embedding': embedding
                    })
                
                # Update or create tracked faces
                used_detections = set()
                
                # Update existing tracks
                for tf in tracked_faces[:]:
                    if not tf.is_active():
                        tracked_faces.remove(tf)
                        continue
                    
                    # Find nearest detection
                    best_dist = float('inf')
                    best_idx = None
                    
                    for idx, det in enumerate(current_detections):
                        if idx in used_detections:
                            continue
                        dist = tf.distance_to(det['bbox'])
                        if dist < best_dist and dist < TRACKING_DISTANCE_THRESHOLD:
                            best_dist = dist
                            best_idx = idx
                    
                    if best_idx is not None:
                        det = current_detections[best_idx]
                        tf.update_position(det['bbox'], det['face_img'])
                        used_detections.add(best_idx)
                        
                        # Try recognition if confirmed but not identified
                        if tf.confirmed and tf.unique_id is None:
                            match_id, distance = db.find_match(det['embedding'])
                            
                            if distance < DISTANCE_THRESHOLD:
                                tf.unique_id = match_id
                                print(f"✅ Recognized: {match_id[:8]} (dist: {distance:.3f})")
                            else:
                                # Enroll new face
                                print(f"❓ Unknown face (nearest: {distance:.3f})")
                                new_id = enroll_face(tf.face_img, db)
                                if new_id:
                                    tf.unique_id = new_id
                                    tf.last_db_update = time.time()
                        
                        # Update database if needed
                        elif tf.unique_id and tf.can_update_db():
                            update_face_count(tf.unique_id)
                            tf.last_db_update = time.time()
                
                # Create new tracks for unmatched detections
                for idx, det in enumerate(current_detections):
                    if idx not in used_detections and len(tracked_faces) < MAX_TRACKED_FACES:
                        tracked_faces.append(TrackedFace(
                            det['bbox'], det['face_img'], det['embedding']
                        ))
                
                # Draw all tracked faces
                for tf in tracked_faces:
                    x, y, w, h = tf.bbox
                    
                    if tf.unique_id:
                        color = (0, 255, 0)
                        label = f"{tf.unique_id[:8]}"
                    elif tf.confirmed:
                        color = (0, 255, 255)
                        label = "Enrolling..."
                    else:
                        color = (0, 165, 255)
                        label = f"Tracking ({tf.confirmation_count}/{CONFIRMATION_FRAMES})"
                    
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(display, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            except Exception as e:
                print(f"⚠️ Processing error: {e}")
            
            # Display
            cv2.putText(display, f"FPS: {avg_fps:.1f} | Known: {len(db.faces)} | Tracked: {len(tracked_faces)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Face Recognition', display)
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                db.load_from_disk()
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Shutdown complete")

if __name__ == "__main__":
    main()