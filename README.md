# Registration Photo Validator

A web tool that checks registration photos against the supplied acceptance rules. It combines:

- local computer-vision checks for face count, resolution, pose, crop, eye openness, lighting, and background;
- a local CLIP vision-model review for coverings, sunglasses/glare, filters/editing, selfie style, and a second one-person check;
- fail-closed behaviour: a photo is never marked valid if the local visual review is unavailable.

The image is processed in memory and is not written to disk by this application.

## Requirements

- Python 3.11
- About 1 GB of free disk space for dependencies and the local CLIP model

No API key or separate AI server is required.

## Setup

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Run

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8080 --loop asyncio --http h11
```

Open <http://127.0.0.1:8080>.

Stop the server with `Ctrl+C`.

The first photo check downloads the CLIP model (about 600 MB). Later checks use the locally cached copy. Uploaded photos are processed in memory and are not sent to an external AI service.

## Configuration

Optional: set `LOCAL_VISION_MODEL` to another Hugging Face CLIP-compatible model, or adjust the conservative pass threshold with `LOCAL_VISION_THRESHOLD` (default `0.60`).

## Registration integration

`POST /api/validate` accepts a multipart field named `file`. A response has:

- `status`: `valid`, `invalid`, or `review_required`;
- `checks`: every rule with its pass/fail result and user-facing message;
- `notes`: optional review details.

Only `valid` should enable registration. The demo emits a browser event named `registration-photo-approved` from its Continue button.

## Production notes

- Add authentication, rate limiting, CSRF protection, request auditing without image bodies, and your normal retention/privacy notice.
- Automatic checks can make mistakes. Provide a human-review route for accessibility and disputed rejections.
- This verifies photo compliance, not identity, liveness, age, or whether the photo is recent. Those require separate controls.
