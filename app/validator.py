from __future__ import annotations

import asyncio
import io
import os
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


MAX_BYTES = 8 * 1024 * 1024
MIN_SIDE = 480
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@dataclass
class Check:
    key: str
    label: str
    passed: bool
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass
class NormalizedLandmark:
    x: float
    y: float


def _check(key: str, label: str, passed: bool, fail: str) -> Check:
    # OpenCV/NumPy comparisons can return numpy.bool_, which FastAPI cannot
    # serialize. Normalize every result at this boundary.
    normalized = bool(passed)
    return Check(key, label, normalized, "Passed" if normalized else fail)


def _point(landmarks: Any, index: int, width: int, height: int) -> np.ndarray:
    mark = landmarks[index]
    return np.array([mark.x * width, mark.y * height], dtype=np.float64)


def _eye_aspect_ratio(landmarks: Any, indices: list[int], width: int, height: int) -> float:
    p = [_point(landmarks, index, width, height) for index in indices]
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]) + 1e-6))


def _background_score(image: np.ndarray, bounds: tuple[int, int, int, int]) -> float:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bounds
    mask = np.zeros((height, width), dtype=np.uint8)
    pad_x, pad_y = int((x2 - x1) * 0.2), int((y2 - y1) * 0.1)
    # Sample only the wall around and above the head. Lower side areas often
    # contain arms or clothing in otherwise valid upper-body portraits.
    mask[: max(1, y1 - pad_y), :] = 1
    mask[max(0, y1 - pad_y) : min(height, y2 + pad_y), : max(0, x1 - pad_x)] = 1
    mask[max(0, y1 - pad_y) : min(height, y2 + pad_y), min(width, x2 + pad_x) :] = 1
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab[mask == 1]
    if len(pixels) < 100:
        return 999.0
    return float(np.mean(np.std(pixels, axis=0)))


def deterministic_checks(data: bytes) -> tuple[list[Check], np.ndarray | None]:
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        return [Check("file", "Readable image", False, "The uploaded file is not a readable image.")], None

    height, width = image.shape[:2]
    checks = [
        _check(
            "resolution",
            "Clear, sufficient resolution",
            min(width, height) >= MIN_SIDE,
            f"Image is {width}×{height}; the shortest side must be at least {MIN_SIDE}px.",
        )
    ]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.65,
    ) as mesh:
        result = mesh.process(rgb)

    faces = result.multi_face_landmarks or []
    landmarks = None
    if faces:
        checks.append(
            _check(
                "one_person",
                "Only one person",
                len(faces) == 1,
                "More than one face was found.",
            )
        )
        if len(faces) > 1:
            return checks, image
        landmarks = faces[0].landmark
    else:
        # Face Mesh needs enough pixels for eye and pose landmarks. Use the
        # long-range detector, then retry on an enlarged face crop.
        with mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.6,
        ) as detector:
            distant_faces = detector.process(rgb).detections or []
        if len(distant_faces) != 1:
            checks.append(
                _check(
                    "one_person",
                    "Only one person",
                    False,
                    "No face was found." if not distant_faces else "More than one face was found.",
                )
            )
            return checks, image

        checks.append(_check("one_person", "Only one person", True, ""))
        box = distant_faces[0].location_data.relative_bounding_box
        if box.width < 0.10 or box.height < 0.075:
            checks.append(
                Check(
                    "face_size",
                    "Face large enough to check",
                    False,
                    "Your face is too small in the photo. Upload a closer portrait.",
                )
            )
            return checks, image

        margin_x, margin_y = box.width * 0.75, box.height * 0.75
        crop_x1 = max(0, int((box.xmin - margin_x) * width))
        crop_y1 = max(0, int((box.ymin - margin_y) * height))
        crop_x2 = min(width, int((box.xmin + box.width + margin_x) * width))
        crop_y2 = min(height, int((box.ymin + box.height + margin_y) * height))
        crop = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_height, crop_width = crop.shape[:2]
        scale = max(1.0, 640 / max(1, min(crop_width, crop_height)))
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
        ) as crop_mesh:
            crop_result = crop_mesh.process(enlarged)
        crop_faces = crop_result.multi_face_landmarks or []
        if not crop_faces:
            checks.append(
                Check(
                    "face_size",
                    "Face details clear enough to check",
                    False,
                    "Move closer or upload a sharper portrait so facial details can be checked.",
                )
            )
            return checks, image

        landmarks = [
            NormalizedLandmark(
                x=(crop_x1 + mark.x * crop_width) / width,
                y=(crop_y1 + mark.y * crop_height) / height,
            )
            for mark in crop_faces[0].landmark
        ]

    assert landmarks is not None
    xs = np.array([mark.x for mark in landmarks])
    ys = np.array([mark.y for mark in landmarks])
    x1, y1 = int(xs.min() * width), int(ys.min() * height)
    x2, y2 = int(xs.max() * width), int(ys.max() * height)
    face_width = (x2 - x1) / width
    face_height = (y2 - y1) / height

    left_eye = _point(landmarks, 33, width, height)
    right_eye = _point(landmarks, 263, width, height)
    nose = _point(landmarks, 1, width, height)
    eye_midpoint = (left_eye + right_eye) / 2
    eye_distance = np.linalg.norm(right_eye - left_eye)
    roll = abs(np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])))
    yaw_proxy = abs((nose[0] - eye_midpoint[0]) / (eye_distance + 1e-6))

    checks.extend(
        [
            _check(
                "front_facing",
                "Front-facing, head straight",
                roll <= 10 and yaw_proxy <= 0.27,
                "Use a mostly front-facing photo without a strong side angle or head tilt.",
            ),
            _check(
                "full_face",
                "Full face visible",
                x1 > width * 0.025 and y1 > height * 0.015 and x2 < width * 0.975 and y2 < height * 0.96,
                "Part of the face is too close to or outside the photo edge.",
            ),
            _check(
                "framing",
                "Usable portrait framing",
                face_width <= 0.62 and face_height <= 0.72,
                "Move slightly farther away so the full face remains comfortably in frame.",
            ),
        ]
    )

    left_ear = _eye_aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144], width, height)
    right_ear = _eye_aspect_ratio(landmarks, [362, 385, 387, 263, 373, 380], width, height)
    checks.append(
        _check(
            "eyes_open",
            "Both eyes open and visible",
            left_ear >= 0.16 and right_ear >= 0.16,
            "Both eyes must be open and fully visible.",
        )
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_light = float(np.mean(gray))
    left_light = float(np.mean(gray[:, : width // 2]))
    right_light = float(np.mean(gray[:, width // 2 :]))
    lighting_ok = 65 <= mean_light <= 210 and abs(left_light - right_light) <= 45
    checks.append(
        _check(
            "lighting",
            "Good, even lighting",
            lighting_ok,
            "Use brighter, even lighting without heavy side shadows.",
        )
    )

    background_score = _background_score(image, (x1, y1, x2, y2))
    checks.append(
        _check(
            "background",
            "Plain, simple background",
            background_score <= 30,
            "Use a plain wall or other uncluttered background.",
        )
    )
    return checks, image


CLIP_CATEGORIES = [
    (
        "face_uncovered",
        "Face fully uncovered",
        "Remove any mask, niqab, hand, or object covering the face.",
        "an uncovered face with forehead, eyes, nose, mouth, and chin clearly visible, hijab allowed",
        "a face covered by a mask, niqab, hand, scarf, hair, or another object",
    ),
    (
        "no_sunglasses",
        "No sunglasses or glasses glare",
        "Remove sunglasses or glasses with visible glare.",
        "a face with clearly visible uncovered eyes, no sunglasses and no eyeglass glare",
        "a face wearing sunglasses, dark glasses, or glasses glare hiding the eyes",
    ),
    (
        "no_filter_or_editing",
        "No filters or beauty editing",
        "Upload an original, unedited photo.",
        "a natural unedited real camera photograph of a person",
        "a beauty-filtered, airbrushed, sticker-covered, illustrated, or heavily edited portrait",
    ),
    (
        "passport_style_not_selfie",
        "Not a selfie",
        "Upload a photo taken by another person, not an arm's-length or mirror selfie.",
        "a photo of a person taken by someone else, with no camera or phone visible",
        "an arm-length selfie, mirror selfie, close-up phone selfie, or a person holding the camera",
    ),
    (
        "one_person",
        "Only one person (visual review)",
        "Only one person may appear in the frame.",
        "a photograph containing exactly one person",
        "a photograph containing two people, a group, or multiple faces",
    ),
]

_clip_model: Any = None
_clip_processor: Any = None


def _load_clip() -> tuple[Any, Any]:
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        from transformers import CLIPModel, CLIPProcessor

        model_name = os.getenv("LOCAL_VISION_MODEL", "openai/clip-vit-base-patch32")
        # Public model: force anonymous download so stale machine-level HF tokens
        # cannot break first-run setup.
        _clip_processor = CLIPProcessor.from_pretrained(model_name, token=False)
        _clip_model = CLIPModel.from_pretrained(model_name, token=False)
        _clip_model.eval()
    return _clip_model, _clip_processor


def _run_clip_checks(data: bytes) -> tuple[list[Check], str]:
    import torch

    model, processor = _load_clip()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    # Coverings and eyewear must be judged from the face, not from unrelated
    # objects or highlights elsewhere in the scene.
    rgb = np.asarray(image)
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    ) as detector:
        detections = detector.process(rgb).detections or []
    face_image = image
    if len(detections) == 1:
        box = detections[0].location_data.relative_bounding_box
        width, height = image.size
        margin = 0.35
        face_image = image.crop(
            (
                max(0, int((box.xmin - box.width * margin) * width)),
                max(0, int((box.ymin - box.height * margin) * height)),
                min(width, int((box.xmin + box.width * (1 + margin)) * width)),
                min(height, int((box.ymin + box.height * (1 + margin)) * height)),
            )
        )

    def classify_pairs(categories: list[tuple[str, str, str, str, str]], source: Image.Image) -> list[float]:
        prompts = [prompt for category in categories for prompt in category[3:5]]
        inputs = processor(text=prompts, images=source, return_tensors="pt", padding=True)
        with torch.inference_mode():
            logits = model(**inputs).logits_per_image[0].reshape(len(categories), 2)
            return torch.softmax(logits, dim=1)[:, 0].cpu().tolist()

    probabilities = classify_pairs(CLIP_CATEGORIES[:2], face_image)
    probabilities.extend(classify_pairs(CLIP_CATEGORIES[2:], image))

    threshold = float(os.getenv("LOCAL_VISION_THRESHOLD", "0.60"))
    checks = [
        _check(key, label, probability >= threshold, fail)
        for (key, label, fail, _, _), probability in zip(CLIP_CATEGORIES, probabilities, strict=True)
    ]
    confidence_note = "Local vision confidence: " + ", ".join(
        f"{category[0]} {probability:.0%}"
        for category, probability in zip(CLIP_CATEGORIES, probabilities, strict=True)
    )
    return checks, confidence_note


async def local_vision_checks(data: bytes) -> tuple[list[Check], str | None, str | None]:
    try:
        checks, notes = await asyncio.to_thread(_run_clip_checks, data)
        return checks, None, notes
    except ImportError:
        return [], "Local vision dependencies are missing. Run: pip install -r requirements.txt", None
    except (OSError, ValueError, RuntimeError) as exc:
        return [], f"Local vision review failed: {type(exc).__name__}.", None


async def validate_photo(data: bytes) -> dict[str, Any]:
    checks, image = deterministic_checks(data)
    cannot_review_face = any(
        check.key in {"one_person", "face_size"} and not check.passed
        for check in checks
    )
    if image is None or cannot_review_face:
        return {"status": "invalid", "checks": [check.as_dict() for check in checks], "notes": None}

    semantic, semantic_error, semantic_notes = await local_vision_checks(data)
    checks.extend(semantic)
    if semantic_error:
        return {
            "status": "review_required",
            "checks": [check.as_dict() for check in checks],
            "notes": semantic_error,
        }

    return {
        "status": "valid" if all(check.passed for check in checks) else "invalid",
        "checks": [check.as_dict() for check in checks],
        "notes": semantic_notes,
    }
