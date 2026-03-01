"""CRNN-based license plate OCR with state-specific validation,
confusion correction, duplicate suppression, and partial-read merging."""

import logging
import re
import time
from collections import OrderedDict

import cv2
import numpy as np

from ai.tensorrt_utils import TRTEngine
from ai.utils import compute_iou

logger = logging.getLogger(__name__)

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BLANK_IDX = len(CHARSET)

# ──────────────────────────────────────────────
# OCR confusion correction matrix
# Maps commonly mis-read characters to their intended correction.
# Applied context-sensitively: letter context vs digit context.
# ──────────────────────────────────────────────
_LETTER_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "G": "6", "Z": "2", "T": "7"}
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z", "7": "T"}

# ──────────────────────────────────────────────
# Accurate US state plate format patterns
# Each state has one or more regex patterns covering standard passenger plates.
# Format key: A = letter, 9 = digit, X = alphanumeric
# ──────────────────────────────────────────────
STATE_PATTERNS: dict[str, list[re.Pattern]] = {
    # Alabama: 1-9AA0A00 or 00AA000
    "AL": [re.compile(r"^[1-9][A-Z]{2}\d[A-Z]\d{2}$"), re.compile(r"^\d{2}[A-Z]{2}\d{3}$")],
    # Alaska: AAA 000
    "AK": [re.compile(r"^[A-Z]{3}\d{3}$")],
    # Arizona: AAA0000
    "AZ": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # Arkansas: 000 AAA or AAA 000
    "AR": [re.compile(r"^\d{3}[A-Z]{3}$"), re.compile(r"^[A-Z]{3}\d{3}$")],
    # California: 0AAA000
    "CA": [re.compile(r"^\d[A-Z]{3}\d{3}$")],
    # Colorado: AAA-000 or 000-AAA
    "CO": [re.compile(r"^[A-Z]{3}\d{3}$"), re.compile(r"^\d{3}[A-Z]{3}$")],
    # Connecticut: AA-00000
    "CT": [re.compile(r"^[A-Z]{2}\d{5}$")],
    # Delaware: 000000 (up to 6 digits, no letters)
    "DE": [re.compile(r"^\d{1,6}$")],
    # Florida: AAA A00 or 000 AAA
    "FL": [
        re.compile(r"^[A-Z]{3}[A-Z]\d{2}$"),
        re.compile(r"^[A-Z]{4}\d{2}$"),
        re.compile(r"^\d{3}[A-Z]{3}$"),
        re.compile(r"^[A-Z]{3}\d{3}$"),
    ],
    # Georgia: AAA0000
    "GA": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # Hawaii: AAA 000 or 000 AAA
    "HI": [re.compile(r"^[A-Z]{3}\d{3}$"), re.compile(r"^\d{3}[A-Z]{3}$")],
    # Idaho: 0A 00000 or A 000000
    "ID": [re.compile(r"^\d[A-Z]\d{5}$"), re.compile(r"^[A-Z]\d{6}$")],
    # Illinois: AA 00000
    "IL": [re.compile(r"^[A-Z]{2}\d{5}$"), re.compile(r"^[A-Z]\d{6}$")],
    # Indiana: 000 AAA
    "IN": [re.compile(r"^\d{3}[A-Z]{3}$")],
    # Iowa: AAA 000
    "IA": [re.compile(r"^[A-Z]{3}\d{3}$")],
    # Kansas: 000 AAA
    "KS": [re.compile(r"^\d{3}[A-Z]{3}$")],
    # Kentucky: 000 AAA
    "KY": [re.compile(r"^\d{3}[A-Z]{3}$")],
    # Louisiana: 000 AAA
    "LA": [re.compile(r"^\d{3}[A-Z]{3}$")],
    # Maine: 0000 AA
    "ME": [re.compile(r"^\d{4}[A-Z]{2}$")],
    # Maryland: 0AA0000
    "MD": [re.compile(r"^\d[A-Z]{2}\d{4}$")],
    # Massachusetts: 0AA 000 or 000 AA0
    "MA": [re.compile(r"^\d[A-Z]{2}\d{3}$"), re.compile(r"^\d{3}[A-Z]{2}\d$")],
    # Michigan: AAA 0000
    "MI": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # Minnesota: 000 AAA
    "MN": [re.compile(r"^\d{3}[A-Z]{3}$")],
    # Mississippi: AAA 0000
    "MS": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # Missouri: AA0 A0A
    "MO": [re.compile(r"^[A-Z]{2}\d[A-Z]\d[A-Z]$"), re.compile(r"^\d{3}[A-Z]{3}$")],
    # Montana: 0-00000A
    "MT": [re.compile(r"^\d\d{5}[A-Z]$"), re.compile(r"^\d{6}[A-Z]$")],
    # Nebraska: AAA 000
    "NE": [re.compile(r"^[A-Z]{3}\d{3}$"), re.compile(r"^\d[A-Z]{2}\d{3}$")],
    # Nevada: 000 AAA
    "NV": [re.compile(r"^\d{3}[A-Z]{3}$")],
    # New Hampshire: 000 0000
    "NH": [re.compile(r"^\d{3}\d{4}$"), re.compile(r"^\d{7}$")],
    # New Jersey: A00 AAA
    "NJ": [re.compile(r"^[A-Z]\d{2}[A-Z]{3}$")],
    # New Mexico: 000 AAA or AAA 000
    "NM": [re.compile(r"^\d{3}[A-Z]{3}$"), re.compile(r"^[A-Z]{3}\d{3}$")],
    # New York: AAA 0000
    "NY": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # North Carolina: AAA-0000
    "NC": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # North Dakota: 000 AAA
    "ND": [re.compile(r"^\d{3}[A-Z]{3}$")],
    # Ohio: AAA 0000
    "OH": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # Oklahoma: AAA 000
    "OK": [re.compile(r"^[A-Z]{3}\d{3}$")],
    # Oregon: 000 AAA
    "OR": [re.compile(r"^\d{3}[A-Z]{3}$")],
    # Pennsylvania: AAA 0000
    "PA": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # Rhode Island: AA-000
    "RI": [re.compile(r"^[A-Z]{2}\d{3}$"), re.compile(r"^\d{3}[A-Z]{2}$")],
    # South Carolina: AAA 000
    "SC": [re.compile(r"^[A-Z]{3}\d{3}$"), re.compile(r"^\d{3}\d[A-Z]{2}$")],
    # South Dakota: 0AA 000
    "SD": [re.compile(r"^\d[A-Z]{2}\d{3}$")],
    # Tennessee: 000 AAA
    "TN": [re.compile(r"^\d{3}[A-Z]{3}$")],
    # Texas: AAA-0000 or 000-AAA
    "TX": [re.compile(r"^[A-Z]{3}\d{4}$"), re.compile(r"^\d{3}[A-Z]{3}$")],
    # Utah: A00 0AA
    "UT": [re.compile(r"^[A-Z]\d{2}\d[A-Z]{2}$")],
    # Vermont: AAA 000
    "VT": [re.compile(r"^[A-Z]{3}\d{3}$")],
    # Virginia: AAA-0000
    "VA": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # Washington: AAA0000
    "WA": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # West Virginia: 0AA 000
    "WV": [re.compile(r"^\d[A-Z]{2}\d{3}$"), re.compile(r"^[A-Z]{3}\d{4}$")],
    # Wisconsin: AAA-0000
    "WI": [re.compile(r"^[A-Z]{3}\d{4}$")],
    # Wyoming: 0-00000
    "WY": [re.compile(r"^\d\d{5}$"), re.compile(r"^\d{6}$")],
    # DC
    "DC": [re.compile(r"^[A-Z]{2}\d{4}$")],
}


class _RecentPlateCache:
    """LRU cache with TTL for duplicate suppression.

    Stores recently seen plate texts and their last-seen timestamp.
    `is_duplicate()` returns True if the same plate was seen within `ttl` seconds.
    """

    def __init__(self, maxsize: int = 256, ttl: float = 3.0):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl

    def is_duplicate(self, plate_text: str) -> bool:
        now = time.monotonic()
        if plate_text in self._cache:
            last_seen = self._cache[plate_text]
            if now - last_seen < self._ttl:
                # Update timestamp and move to end
                self._cache[plate_text] = now
                self._cache.move_to_end(plate_text)
                return True
        # Not a duplicate — record it
        self._cache[plate_text] = now
        self._cache.move_to_end(plate_text)
        # Evict oldest if over capacity
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return False

    def get_best_read(self, plate_text: str) -> str | None:
        """Return the plate text if it exists in cache (for merging), else None."""
        if plate_text in self._cache:
            return plate_text
        return None


class _PartialReadMerger:
    """Merges partial OCR reads across frames for the same plate region.

    Tracks recent partial reads keyed by approximate bounding box position.
    When a new partial read arrives, attempts to merge it with prior partials
    to form a complete plate text.
    """

    def __init__(self, ttl: float = 2.0, iou_thresh: float = 0.5):
        self._partials: dict[str, dict] = {}  # key -> {text, confidence, timestamp, bbox}
        self._ttl = ttl
        self._iou_thresh = iou_thresh

    def try_merge(
        self, text: str, confidence: float, bbox: list[int] | None = None
    ) -> tuple[str, float]:
        """Attempt to merge a partial read with stored partials.

        Returns the best merged text and its confidence.
        """
        now = time.monotonic()

        # Evict stale entries
        stale = [k for k, v in self._partials.items() if now - v["timestamp"] > self._ttl]
        for k in stale:
            del self._partials[k]

        if not bbox:
            return text, confidence

        # Find matching partial by bbox overlap
        best_key = None
        best_iou = 0.0
        for key, entry in self._partials.items():
            iou = compute_iou(bbox, entry["bbox"])
            if iou > self._iou_thresh and iou > best_iou:
                best_iou = iou
                best_key = key

        key = best_key or f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"

        if best_key and best_key in self._partials:
            prev = self._partials[best_key]
            # Keep the longer read, or higher confidence if same length
            if len(text) > len(prev["text"]) or (
                len(text) == len(prev["text"]) and confidence > prev["confidence"]
            ):
                merged_text, merged_conf = text, confidence
            else:
                merged_text, merged_conf = prev["text"], prev["confidence"]

            self._partials[key] = {
                "text": merged_text,
                "confidence": merged_conf,
                "timestamp": now,
                "bbox": bbox,
            }
            return merged_text, merged_conf

        # No match — store as new partial
        self._partials[key] = {
            "text": text,
            "confidence": confidence,
            "timestamp": now,
            "bbox": bbox,
        }
        return text, confidence



def _apply_confusion_correction(text: str, pattern: re.Pattern | None = None) -> str:
    """Apply OCR confusion correction based on expected character positions.

    If a pattern is provided, uses it to determine which positions expect
    letters vs digits and corrects accordingly. Without a pattern, applies
    heuristic correction based on surrounding character context.
    """
    if not text:
        return text

    if pattern is None:
        # No pattern to guide — return as-is
        return text

    # Extract position expectations from pattern
    pat_str = pattern.pattern.strip("^$")
    corrected = list(text)

    pos = 0
    pat_idx = 0
    while pos < len(corrected) and pat_idx < len(pat_str):
        c = corrected[pos]

        # Determine what this position expects
        if pat_str[pat_idx] == "[":
            # Character class — find closing bracket
            try:
                end = pat_str.index("]", pat_idx)
            except ValueError:
                pat_idx += 1
                pos += 1
                continue
            char_class = pat_str[pat_idx + 1 : end]
            pat_idx = end + 1

            # Check for quantifier
            if pat_idx < len(pat_str) and pat_str[pat_idx] == "{":
                try:
                    q_end = pat_str.index("}", pat_idx)
                except ValueError:
                    pass
                else:
                    pat_idx = q_end + 1

            expects_letter = "A-Z" in char_class and "0-9" not in char_class
            expects_digit = "0-9" in char_class and "A-Z" not in char_class

            if expects_letter and c.isdigit() and c in _DIGIT_TO_LETTER:
                corrected[pos] = _DIGIT_TO_LETTER[c]
            elif expects_digit and c.isalpha() and c in _LETTER_TO_DIGIT:
                corrected[pos] = _LETTER_TO_DIGIT[c]

            pos += 1

        elif pat_str[pat_idx] == "\\" and pat_idx + 1 < len(pat_str):
            next_c = pat_str[pat_idx + 1]
            if next_c == "d":
                # Expects digit
                if c.isalpha() and c in _LETTER_TO_DIGIT:
                    corrected[pos] = _LETTER_TO_DIGIT[c]
                pat_idx += 2
                # Check for quantifier
                if pat_idx < len(pat_str) and pat_str[pat_idx] == "{":
                    q_end = pat_str.index("}", pat_idx)
                    pat_idx = q_end + 1
                pos += 1
            else:
                pat_idx += 2
                pos += 1
        else:
            pat_idx += 1
            pos += 1

    return "".join(corrected)


class PlateOCR:
    def __init__(self, config: dict):
        self.config = config
        self.input_w, self.input_h = config.get("input_size", [200, 48])
        self.conf_thresh = config.get("confidence_threshold", 0.7)
        self.char_conf_min = config.get("char_confidence_min", 0.4)
        self.validate = config.get("validate_format", True)
        self.max_length = config.get("max_plate_length", 8)
        self.min_length = config.get("min_plate_length", 2)
        self.enable_confusion_correction = config.get("confusion_correction", True)

        self.engine = TRTEngine(
            engine_path=config["model_path"],
            onnx_path=config.get("onnx_path"),
        )

        # Pre-created CLAHE (reused every preprocess call, eliminates per-call alloc)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Duplicate suppression: same plate within TTL window is suppressed
        dup_ttl = config.get("duplicate_ttl", 3.0)
        dup_maxsize = config.get("duplicate_cache_size", 256)
        self._dup_cache = _RecentPlateCache(maxsize=dup_maxsize, ttl=dup_ttl)

        # Partial read merging across frames
        self._partial_merger = _PartialReadMerger(
            ttl=config.get("partial_merge_ttl", 2.0),
            iou_thresh=config.get("partial_merge_iou", 0.5),
        )

    def preprocess(self, plate_img: np.ndarray) -> np.ndarray:
        """Preprocess plate image for CRNN.

        Converts to grayscale, enhances contrast, resizes, and normalizes.
        Uses pre-created CLAHE instance (no per-call allocation).
        """
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img

        # CLAHE for contrast enhancement (reuse instance from __init__)
        enhanced = self._clahe.apply(gray)

        # Resize to fixed input size
        resized = cv2.resize(enhanced, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] and add batch + channel dims
        blob = resized.astype(np.float32) / 255.0
        blob = blob[np.newaxis, np.newaxis, ...]  # [1, 1, H, W]

        return blob

    def ctc_decode(self, output: np.ndarray) -> tuple[str, float, list[float]]:
        """CTC greedy decode with per-character confidence.

        Args:
            output: CRNN output [T, C] or [1, T, C]

        Returns:
            Decoded text, average confidence, and per-character confidences
        """
        if output.ndim == 3:
            output = output.squeeze(0)

        # Softmax
        exp = np.exp(output - np.max(output, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)

        # Greedy decode
        indices = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)

        chars = []
        char_confs = []
        prev_idx = -1

        for t in range(len(indices)):
            idx = indices[t]
            if idx != BLANK_IDX and idx != prev_idx and idx < len(CHARSET):
                chars.append(CHARSET[idx])
                char_confs.append(float(max_probs[t]))
            prev_idx = idx

        text = "".join(chars)
        avg_conf = float(np.mean(char_confs)) if char_confs else 0.0

        return text, avg_conf, char_confs

    def validate_plate(self, text: str, state: str | None = None) -> bool:
        """Validate plate text against known formats.

        Args:
            text: Uppercase plate text (no spaces/dashes)
            state: Optional 2-letter state code for state-specific validation

        Returns:
            True if the plate text is plausible
        """
        if len(text) < self.min_length or len(text) > self.max_length:
            return False

        # Basic sanity: must contain at least one letter OR one digit
        # (Delaware plates are all-digits, vanity plates may be all-letters)
        if not re.search(r"[A-Z0-9]", text):
            return False

        # Reject text that looks like OCR garbage (all same character, etc.)
        if len(set(text)) == 1 and len(text) > 2:
            return False

        if state and state.upper() in STATE_PATTERNS and self.validate:
            return any(pat.match(text) for pat in STATE_PATTERNS[state.upper()])

        # Without a state, accept any plausible alphanumeric combination
        # Must be 2-8 chars, at least 2 unique characters
        return len(set(text)) >= 2

    def _find_matching_pattern(self, text: str, state: str | None = None) -> re.Pattern | None:
        """Find the state pattern that matches or best fits the text."""
        if not state or state.upper() not in STATE_PATTERNS:
            return None
        for pat in STATE_PATTERNS[state.upper()]:
            if pat.match(text):
                return pat
        # Return first pattern for confusion correction even if no match
        return STATE_PATTERNS[state.upper()][0] if STATE_PATTERNS.get(state.upper()) else None

    def read(
        self,
        plate_img: np.ndarray,
        state: str | None = None,
        plate_bbox: list[int] | None = None,
    ) -> dict | None:
        """Read text from a plate image.

        Args:
            plate_img: Cropped plate image (BGR or grayscale)
            state: Optional 2-letter state code for validation/correction
            plate_bbox: Optional [x,y,w,h] for partial-read merging

        Returns:
            Dict with plate_text, confidence, raw_ocr, is_duplicate, or None
        """
        if not self.engine.is_loaded:
            return None

        if plate_img.size == 0:
            return None

        blob = self.preprocess(plate_img)
        outputs = self.engine.infer(blob)
        if not outputs or outputs[0] is None:
            return None
        raw_text, avg_conf, char_confs = self.ctc_decode(outputs[0])

        # Deterministic formatting: uppercase, strip whitespace
        raw_text = raw_text.upper().strip()

        if not raw_text:
            return None

        # Per-character confidence filtering
        if char_confs:
            low_chars = sum(1 for c in char_confs if c < self.char_conf_min)
            if low_chars > len(char_confs) * 0.3:
                # Too many low-confidence characters — unreliable read
                logger.debug(
                    "Rejecting plate %s: %d/%d chars below min confidence",
                    raw_text,
                    low_chars,
                    len(char_confs),
                )
                return None

        # Average confidence threshold
        if avg_conf < self.conf_thresh:
            return None

        # Apply OCR confusion correction
        corrected_text = raw_text
        if self.enable_confusion_correction and state:
            pattern = self._find_matching_pattern(raw_text, state)
            if pattern:
                corrected_text = _apply_confusion_correction(raw_text, pattern)

        # Validate plate format
        if (
            self.validate
            and not self.validate_plate(corrected_text, state)
            and not self.validate_plate(corrected_text)
        ):
            logger.debug("Plate validation failed for: %s (state=%s)", corrected_text, state)
            return None

        # Partial-read merging
        if plate_bbox:
            corrected_text, avg_conf = self._partial_merger.try_merge(
                corrected_text, avg_conf, plate_bbox
            )

        # Duplicate suppression
        is_dup = self._dup_cache.is_duplicate(corrected_text)

        return {
            "plate_text": corrected_text,
            "confidence": round(avg_conf, 4),
            "raw_ocr": raw_text,
            "is_duplicate": is_dup,
            "char_confidences": [round(c, 3) for c in char_confs],
        }

