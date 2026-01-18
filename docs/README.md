# AVALOKAN — Face Recognition & Monitoring System

## 1. Introduction

AVALOKAN is a lightweight, practical system for real‑time face detection, recognition, appearance tracking, and monitoring via a web dashboard. It is designed for research, demo, and small‑scale deployment use‑cases such as attendance, premises monitoring, and controlled access experiments.

"Avalokan" (from Sanskrit/Hindi) loosely means "to observe", "to look upon", or "to inspect" — reflecting the system's focus on continuous visual monitoring and recognition.

Problem statement:
- Provide reliable, privacy‑conscious face detection and recognition with appearance logging.
- Offer an operator dashboard to monitor activity in near real‑time.
- Integrate with cloud storage and a database for persistent records and synchronization.

---

## 2. System Architecture

High‑level textual architecture diagram:

Camera Input -> Face Processing Engine -> Local Storage (db_faces/, temp_faces/) -> Cloud Storage & Database (Supabase, conceptual) <-> Dashboard (dashboard.html, realtime subscriptions)

Components:
- Camera input
  - Live camera feed or recorded video frames captured by the application.
  - Frames are passed to the face processing engine at a configurable capture rate.
- Face processing engine
  - Performs detection, alignment (optional), encoding (feature vector extraction), and matching against stored encodings.
  - Handles decision logic: new face vs known face vs update counts.
  - Produces metadata (timestamp, bounding box, confidence, encoding ID).
- Database (conceptual)
  - Stores face records, metadata, and appearance logs. In this project, Supabase is the conceptual cloud DB for persistent storage and realtime subscriptions.
- Storage
  - Stores face images and thumbnails (local folders for capture and temporary images; cloud storage for persistent archival).
- Dashboard
  - Web UI that lists detected faces, supports filters, displays thumbnails, and receives realtime updates via database subscriptions.

Realtime flow:
- When a face is detected/recognized, the system writes a record to the database and (optionally) uploads an image to cloud storage.
- The dashboard subscribes to database changes and updates the UI in near realtime.

---

## 3. Technology Stack

- Frontend
  - HTML, CSS, JavaScript (dashboard.html). Simple, responsive UI for listing records and viewing details.
- Backend
  - Python for the capture and processing pipeline. Core files: `main.py`, `tune.py`, `diagnose.py`, `check_env.py`.
- Database & Storage (conceptual)
  - Supabase (Postgres + storage + realtime) — used conceptually for record persistence and subscriptions. Use placeholders for keys and URLs when configuring.
- Realtime mechanisms
  - Database change subscriptions (e.g., Supabase realtime or Postgres LISTEN/NOTIFY) push updates to the dashboard so operators see new/updated face events without polling.

---

## 4. Folder Structure Explanation

Project root: d:\MyProjects\COLLEGE\face_recognition_system

- `db_faces/`
  - Purpose: Persistent local store of known face images and their encodings. Each file typically corresponds to a known identity or a stored sample used for matching.
  - Usage: Used by the recognition engine at startup to build an in‑memory gallery of encodings.

- `temp_faces/`
  - Purpose: Temporary working directory for newly captured face crops, intermediate images, thumbnails, or processing output before upload or archival.
  - Usage: Cleared or rotated according to retention policy. Useful for debugging and manual review.

- `docs/`
  - Purpose: Project documentation and guides. This README is located here.

- `dashboard.html`
  - Purpose: Frontend UI for monitoring face events and viewing face records. Implements filters, sorting, and a realtime update mechanism (via database subscriptions).

- `main.py`
  - Purpose: Primary application entry point. Orchestrates camera capture, face detection, encoding, matching, local storage, and calls to upload/synchronize with the cloud database/storage.

- `tune.py`
  - Purpose: Utilities and scripts to tune detection/recognition parameters (e.g., confidence thresholds, non‑max suppression, detection model selection). Used during development to optimize performance.

- `diagnose.py`
  - Purpose: Diagnostic tools for camera health checks, pipeline latency measurement, and logging utilities. Useful for troubleshooting system behavior on different machines.

- `check_env.py`
  - Purpose: Validates runtime environment and configuration (Python version, required packages available, required environment variables present). Should check that required placeholders exist but must not print secrets.

---

## 5. Dashboard Documentation

UI sections:
- Header: System status, last sync time, connection status to realtime service.
- Filters/controls: Search box, date range selector, sort options, status filters.
- Face records table/grid: Thumbnail, name/ID (if known), timestamp, status tag, confidence, occurrence count.
- Detail/preview pane: Larger image, metadata, and action buttons (e.g., mark as known, add label, download image).
- Logs/Activity stream (optional): Shows raw events coming from the processing engine.

Filters:
- Search: Text search over name, ID, or notes.
- Date range: Select a start and end time to restrict results.
- Sorting: By timestamp (newest/oldest), occurrence count, confidence.
- Status filter: New, Known, Updated.

Realtime updates:
- The dashboard subscribes to database change events. On insert/update events for the face records table, the UI receives updates and refreshes the relevant rows without full reload.
- Connection status indicator informs operators if realtime connectivity is lost.

Face records table:
- Columns commonly shown: Thumbnail, Face ID (or label), Timestamp, Status, Confidence, Appearances (count).
- Pagination or infinite scroll for large datasets.

Status meanings:
- New: Face detected for the first time (no existing matching record).
- Known: Face matched to an existing known identity in the database/gallery.
- Updated: A previously known record was updated (e.g., additional appearance logged, image re‑captured, or label changed).

Image handling & fallbacks:
- If a thumbnail or image fails to load from cloud storage, the UI should display a local fallback icon or the last known local thumbnail (from temp storage) and show an explanatory tooltip.
- Thumbnails should be small for fast listing; clicking opens full preview.

---

## 6. Face Recognition Workflow

Lifecycle (step‑by‑step):

1. Detection
   - Input: Video frame(s) from camera.
   - Process: Run face detector to identify bounding boxes and detection confidences.

2. Preprocessing / Alignment (optional)
   - Crop face, resize to model input, and apply normalization or facial alignment if implemented.

3. Encoding
   - Compute a feature vector (embedding) for each detected face using the chosen recognition model.

4. Matching
   - Compare embedding against the local gallery (encodings loaded from `db_faces/`) using a distance metric (e.g., cosine or Euclidean).
   - If distance below configured threshold → match = Known. Else → New.

5. Storage
   - For New faces: create a new local record (store image to `db_faces/` or `temp_faces/`), generate an ID, and optionally upload thumbnail to cloud storage. Insert a record into the database.
   - For Known faces: log the appearance (timestamp, camera id), optionally update the stored thumbnail if better quality or if configured to refresh images.

6. Update count
   - Maintain an appearances counter. Each recognized appearance increments the count in the DB and updates the last_seen timestamp.

7. Dashboard sync
   - The write to the database triggers realtime subscriptions; dashboard receives the change and updates UI.
   - Optionally send a lightweight message via WebSocket or Pub/Sub for immediate UI feedback.

Notes:
- Tuning thresholds in `tune.py` helps balance false positives and missed matches.
- Pipeline should be asynchronous where possible: detection/encoding independent of network/storage operations.

---

## 7. Environment Configuration

Use a .env file for local development to centralize configuration. Do NOT include secrets in repository or documentation. Example placeholder variables (do not fill with real keys):

- SUPABASE_URL=<your_supabase_url>
- SUPABASE_ANON_KEY=<your_supabase_anon_key>
- SUPABASE_SERVICE_ROLE_KEY=<your_service_role_key>  # use with caution
- STORAGE_BUCKET=<your_storage_bucket_name>
- CAMERA_DEVICE_INDEX=0
- DETECTION_THRESHOLD=0.5
- MATCH_DISTANCE_THRESHOLD=0.45

Important:
- The project must never commit .env files containing real credentials. Use environment specific configuration management in production.
- check_env.py can assert presence of required variables and provide user‑friendly messages when placeholders are not set.

---

## 8. Security Considerations

- Secrets exclusion
  - Never commit API keys, DB passwords, or service role keys to source control.
  - Use environment variables or a secure secrets manager for production.

- Storage access policies
  - Configure cloud storage with least privilege: frontend clients receive only public/thumbnail URLs or signed URLs with expiry.
  - Use service keys server‑side only and rotate them regularly.

- Privacy and compliance
  - Inform stakeholders about face data usage and retention policies.
  - Implement data retention and deletion workflows for images and encodings.

- Safe development practices
  - Mask or replace real identifiers in shared logs or screenshots.
  - Limit access to development and staging projects.
  - Log minimal personal data; prefer hashed or assigned IDs for identities.

---

## 9. Installation & Setup (Conceptual)

1. Python environment
   - Create a virtual environment (recommended).
     - Windows example: python -m venv .venv
     - Activate: .venv\Scripts\activate

2. Dependency setup
   - Install required packages (detection, recognition, web framework, Supabase client). Use a requirements file (not included here) and install via pip: pip install -r requirements.txt

3. Configuration
   - Create a local `.env` with placeholders replaced by developer values (do not include production secrets).
   - Verify environment with: python check_env.py

4. Running the system
   - Start the pipeline: python main.py
   - Open the dashboard: open dashboard.html in a browser (or run a simple HTTP server to serve it if required).
   - Use diagnose.py for troubleshooting and tune.py to adjust thresholds.

---

## 10. Usage Guide

Operator workflow:
- Open the dashboard to monitor incoming face events.
- Use filters to focus on specific time windows or statuses.
- When a New face appears:
  - Inspect thumbnail and metadata.
  - Optionally label and save a known identity (this populates db_faces/ and updates the DB).
- For Known faces:
  - View appearance history and counts.
  - Mark images as preferred or replace thumbnail if needed.

Interpreting data:
- Timestamp indicates when the face was detected.
- Confidence and distance metrics indicate match quality — lower distance implies stronger match.
- Appearance count helps identify frequent visitors vs transient detections.

Monitoring activity:
- Watch the connection status for realtime updates.
- Use logs or diagnose tools to investigate missed frames or degraded detection rates.

---

## 11. Limitations

- Model dependency: Recognition quality depends on chosen detection/embedding models and capture conditions (lighting, angle).
- Scale: Local gallery and single‑machine processing may not scale to very large galleries or high camera counts without architectural changes.
- Privacy/legal: Face recognition involves sensitive data; deployment must follow applicable laws and ethics.
- Environmental factors: Low light, occlusion, extreme poses reduce accuracy.

---

## 12. Future Enhancements

- Multi‑camera support with distributed processing and centralized DB.
- Improved alignment and augmentation to increase encoding robustness.
- Active learning workflow: human-in-the-loop labeling to improve the gallery.
- Access control & role‑based dashboard features.
- Rate‑limited signed URL generation for image access.
- Model hot‑swap and A/B testing for detectors/encoders.

---

## 13. Conclusion

AVALOKAN is a compact, modular face detection and recognition system suited for prototyping, demos, and small deployments. It emphasizes a clear pipeline (capture → process → store → monitor), integrates conceptually with cloud services such as Supabase for persistence and realtime updates, and provides a user‑friendly dashboard for operators. The repository structure separates responsibilities (db_faces, temp_faces, docs) to support maintainability and iterative development.
