# delivery-verification-ml

HTTP microservice for **driver document verification**: OpenCV preprocessing → **RapidOCR (ONNX)** for text, **InsightFace (buffalo_l)** for document-vs-selfie similarity. Includes a **forgery** placeholder endpoint.

Contracts match `delivery-backend` `MlVerificationService`:

| Endpoint | Request | Response |
|----------|---------|----------|
| `POST /ocr` | `{ "imageBase64": "<base64>" }` | `{ "text": "...", "confidence": 0.95 }` |
| `POST /face` | `{ "documentImageBase64": "...", "selfieImageBase64": "..." }` | `{ "similarity": 0.0..1.0 }` |
| `GET /health` | — | `{ "status": "ok", "version": "..." }` |

Optional auth: set `API_KEY`; send `Authorization: Bearer <API_KEY>` (same as Nest `OCR_WORKER_API_KEY` / `FACE_WORKER_API_KEY`).

## Requirements

- Python **3.12+** (local) or Docker (recommended).

## Run with Docker

```bash
cd delivery-verification-ml
docker compose up --build
```

### Build failed with `g++: No such file or directory` (insightface)

The `insightface` package may compile native extensions during `pip install`. The Dockerfile includes `build-essential` and `g++` for that. If you trimmed them, restore those packages or use an image that already has build tools.

### Exit code (Git Bash vs PowerShell)

- **PowerShell:** after a command, `$LASTEXITCODE` is `0` if success, non-zero if failure.
- **Git Bash:** use `echo $?` (same convention as Linux: `0` = success).

If Docker prints `exit code: 1` or `ERROR: ... failed`, the **build did not finish successfully** — fix the error and run `docker compose build` again.

Service: `http://localhost:8000` · OpenAPI: `http://localhost:8000/docs`

**First request** may be slow (InsightFace downloads `buffalo_l` into the volume).

## Run locally

```bash
cd delivery-verification-ml
py -3 -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
set PYTHONPATH=src
uvicorn verification_ml.main:app --reload --host 0.0.0.0 --port 8000
```

## Wire `delivery-backend`

In `.env` (or environment):

```env
ML_VERIFY_ENABLED=true
OCR_WORKER_URL=http://localhost:8000/ocr
FACE_WORKER_URL=http://localhost:8000/face
# Optional, if API_KEY is set in the ML service:
OCR_WORKER_API_KEY=your-secret
FACE_WORKER_API_KEY=your-secret
```

Use the **same** secret as `API_KEY` in this service.

## Limitations

- **Forgery** (`POST /v1/forgery`): stub only; real JPEG/resave heuristics can be added later.
- **OCR**: tuned for general Latin layout; Tunisian ID-specific tuning can refine preprocessing.
- **Face**: uses first/largest face per image; very small faces on documents may score low.

## Data retention

Retention of images/embeddings is enforced in **orchestration** (Nest + storage policy), not inside this service. See project steering docs.
