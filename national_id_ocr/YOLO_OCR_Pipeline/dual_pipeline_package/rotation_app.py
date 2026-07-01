"""
ID Card Rotation Tester  –  Gradio app (v2)
- Background removal via rembg.remove()
- Green-strip similarity sampled from the top-center of the card
  (where جمهورية مصر العربية appears on the Egyptian national ID)

Run:  .venv\Scripts\python.exe national_id_ocr\rotation_app.py
Then open http://127.0.0.1:7860
"""

import cv2
import numpy as np
import json
from pathlib import Path
from io import BytesIO

import gradio as gr
from PIL import Image
from rembg import remove as rembg_remove, new_session

# ── rembg session (loaded once at startup) ────────────────────────────────────
print("Loading rembg model …")
REM_SESSION = new_session("u2net")
print("rembg ready.")

# ── Paths (optional – used only when user saves a custom reference) ──────────
DIMS_JSON     = Path(r"d:\ocr\national_id_ocr\id_reference_dims.json")
REF_CROP_PATH = Path(r"d:\ocr\national_id_ocr\ref_crop_cache.jpg")

# ── Hard-coded built-in reference (718×453 px Egyptian national ID) ───────────
# Card size : 718 x 453 px   Ratio : 1.585
# These values are used automatically when no custom reference file has been saved.
BUILTIN_REF_WIDTH  = 718
BUILTIN_REF_HEIGHT = 453


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1  Background removal via rembg → crop card tightly
# ─────────────────────────────────────────────────────────────────────────────
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def align_by_min_area_rect(image_bgr, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    rect_pts = order_points(box)
    (tl, tr, br, bl) = rect_pts
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect_pts, dst)
    warped = cv2.warpPerspective(image_bgr, M, (maxWidth, maxHeight))
    
    return warped, maxWidth, maxHeight, box

def remove_background_and_crop(image_bgr):
    """
    1. Remove background with rembg (returns BGRA image).
    2. Use the alpha channel to find the card mask.
    3. Use minAreaRect to extract the card, auto-straightening it.
    4. Return (cropped_bgr, width, height).
    """
    # Convert BGR → PIL for rembg
    pil_in  = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pil_out = rembg_remove(pil_in, session=REM_SESSION)   # RGBA PIL image

    # Back to numpy BGRA
    rgba   = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGBA2BGRA)
    alpha  = rgba[:, :, 3]

    # Threshold alpha to find the card mask
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    # Clean up small holes / noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find the largest connected component (the card)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = image_bgr.shape[:2]
        return image_bgr, w, h

    card_contour = max(contours, key=cv2.contourArea)
    cropped, w, h, _ = align_by_min_area_rect(image_bgr, card_contour)
    
    return cropped, w, h


def draw_detection_box(image_bgr):
    """
    Run rembg, find the card contour, draw a green bounding box on the
    original image, and return it as a PIL image.
    """
    pil_in  = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pil_out = rembg_remove(pil_in, session=REM_SESSION)
    rgba    = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGBA2BGRA)
    alpha   = rgba[:, :, 3]
    _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dbg = image_bgr.copy()
    if contours:
        card    = max(contours, key=cv2.contourArea)
        _, w, h, box = align_by_min_area_rect(image_bgr, card)
        cv2.drawContours(dbg, [np.int32(box)], 0, (0, 220, 0), 4)
        top_pt = tuple(np.int32(box[np.argmin(box[:, 1])]))
        cv2.putText(dbg, f"{w}x{h} px", (top_pt[0], max(top_pt[1]-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3)
    return bgr_to_pil(dbg)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2  Green-strip similarity – top-center region only
#  (where جمهورية مصر العربية is printed on the Egyptian national ID)
# ─────────────────────────────────────────────────────────────────────────────
# Region to sample: top GREEN_STRIP_H fraction of height,
#                   center GREEN_STRIP_CX ± GREEN_STRIP_CW/2 of width
GREEN_STRIP_H          = 0.22   # top 22 % of card height
GREEN_STRIP_CX         = 0.50   # horizontal centre (50 %)
GREEN_STRIP_CW         = 0.55   # width of strip = 55 % of card width (centred)
GREEN_STRIP_OFFSET_PCT = 0.15   # shift centre 15% to the right
GREEN_SIM_THRESH       = 0.60


def _top_center_strip(img_bgr):
    """Crop the top-centre region where the green جمهورية مصر العربية band is."""
    h, w = img_bgr.shape[:2]
    top  = max(int(h * GREEN_STRIP_H), 1)
    cx   = int(w * GREEN_STRIP_CX) + int(w * GREEN_STRIP_OFFSET_PCT)   # shifted right
    half = int(w * GREEN_STRIP_CW / 2)
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    return img_bgr[:top, x0:x1]


def _green_strip_hist(img_bgr):
    """Normalised HSV-Hue histogram of green pixels in the top-center strip."""
    strip = _top_center_strip(img_bgr)
    if strip.size == 0:
        return np.zeros((64, 1), np.float32)
    hsv  = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    # Egyptian ID green: OpenCV Hue ~36-90 (= 72°-180° in degrees)
    mask = cv2.inRange(hsv, (30, 20, 30), (95, 255, 255))  # wider range to catch faded/pale greens
    hist = cv2.calcHist([hsv], [0], mask, [64], [0, 180])
    cv2.normalize(hist, hist)
    return hist


def green_strip_similarity(img_bgr, ref_bgr):
    """Histogram intersection (0–1) between top-center green strips."""
    return float(cv2.compareHist(
        _green_strip_hist(img_bgr),
        _green_strip_hist(ref_bgr),
        cv2.HISTCMP_INTERSECT,
    ))


from ultralytics import YOLO
import os

YOLO_MODEL_PATH = r"D:\ocr\national_id_ocr\deployment repo\vso-ocr-backend-main\detect_odjects.pt"
ROTATION_YOLO = None
if os.path.exists(YOLO_MODEL_PATH):
    print("Loading YOLO model for rotation face-check...")
    ROTATION_YOLO = YOLO(YOLO_MODEL_PATH)

def fix_180_flip(img_bgr, ref_bgr):
    """
    Check orientation via YOLO face (photo) detection.
    If the photo is on the right, it's upside down.
    Falls back to green-strip similarity if no photo is detected.
    Returns (corrected_img, was_flipped, score).
    """
    if ROTATION_YOLO is not None:
        h, w = img_bgr.shape[:2]
        results = ROTATION_YOLO.predict(img_bgr, conf=0.25, verbose=False)
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = ROTATION_YOLO.names[class_id]
            if class_name == 'photo':
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                if cx > w / 2:
                    img180 = cv2.rotate(img_bgr, cv2.ROTATE_180)
                    return img180, True, 1.0 # 1.0 dummy score
                else:
                    return img_bgr, False, 1.0

    # Fallback to legacy green strip similarity
    sim_n  = green_strip_similarity(img_bgr, ref_bgr)
    img180 = cv2.rotate(img_bgr, cv2.ROTATE_180)
    sim180 = green_strip_similarity(img180, ref_bgr)
    
    if sim_n >= sim180:
        return img_bgr, False, sim_n
    return img180, True, sim180


def visualise_green_strip(img_bgr):
    """Draw the sampled region as a cyan overlay on the card image."""
    vis      = img_bgr.copy()
    h, w     = vis.shape[:2]
    top      = max(int(h * GREEN_STRIP_H), 1)
    cx       = int(w * GREEN_STRIP_CX) + int(w * GREEN_STRIP_OFFSET_PCT)   # shifted right
    half     = int(w * GREEN_STRIP_CW / 2)
    x0, x1  = max(cx - half, 0), min(cx + half, w)
    overlay  = vis.copy()
    cv2.rectangle(overlay, (x0, 0), (x1, top), (255, 200, 0), -1)
    vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)
    cv2.rectangle(vis, (x0, 0), (x1, top), (0, 200, 255), 3)
    cv2.putText(vis, "green sample region", (x0 + 5, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return bgr_to_pil(vis)


# ─────────────────────────────────────────────────────────────────────────────
#  Orientation helpers
# ─────────────────────────────────────────────────────────────────────────────
RATIO_TOL = 0.08

def rotate90(img, times):
    times = times % 4
    if times == 0: return img
    return cv2.rotate(img, {1: cv2.ROTATE_90_COUNTERCLOCKWISE,
                             2: cv2.ROTATE_180,
                             3: cv2.ROTATE_90_CLOCKWISE}[times])

def orientation_matches(w, h, rw, rh):
    same = (w >= h) == (rw >= rh)
    cr = max(w, h) / max(min(w, h), 1)
    rr = max(rw, rh) / max(min(rw, rh), 1)
    return same and abs(cr - rr) / max(rr, 1e-6) < RATIO_TOL

def bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def _make_builtin_ref_bgr():
    """
    Synthesise a reference image from the hard-coded dimensions.
    The top-centre strip is painted with the Egyptian ID golden-green so that
    the green-strip histogram comparison works without a real reference photo.
    Egyptian ID green ≈ HSV (55, 70, 130)  →  BGR ≈ (65, 130, 115)
    """
    h, w = BUILTIN_REF_HEIGHT, BUILTIN_REF_WIDTH
    img  = np.full((h, w, 3), (180, 180, 170), dtype=np.uint8)   # neutral card body
    top  = int(h * GREEN_STRIP_H)
    cx   = int(w * GREEN_STRIP_CX) + int(w * GREEN_STRIP_OFFSET_PCT)
    half = int(w * GREEN_STRIP_CW / 2)
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    img[:top, x0:x1] = (65, 130, 115)   # Egyptian ID green band
    return img


def _load_reference():
    """Return (ref_bgr, width, height) – from saved file or built-in fallback."""
    if DIMS_JSON.exists() and REF_CROP_PATH.exists():
        dims    = json.loads(DIMS_JSON.read_text())
        ref_bgr = cv2.imread(str(REF_CROP_PATH))
        if ref_bgr is not None:
            return ref_bgr, dims["width"], dims["height"]
    # ── Built-in reference (no file needed) ──────────────────────────────────
    return _make_builtin_ref_bgr(), BUILTIN_REF_WIDTH, BUILTIN_REF_HEIGHT


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio callbacks
# ─────────────────────────────────────────────────────────────────────────────

def set_reference(ref_pil):
    if ref_pil is None:
        return "No image uploaded.", None, None, None
    ref_bgr = cv2.cvtColor(np.array(ref_pil), cv2.COLOR_RGB2BGR)

    detect_vis = draw_detection_box(ref_bgr)     # draw card box on original
    cropped, rw, rh = remove_background_and_crop(ref_bgr)
    ratio = round(rw / rh, 4)
    DIMS_JSON.write_text(json.dumps({"width": rw, "height": rh, "aspect_ratio": ratio}, indent=2))
    cv2.imwrite(str(REF_CROP_PATH), cropped)

    green_vis = visualise_green_strip(cropped)   # show the sampled region

    msg = (f"Reference saved!\n"
           f"Card size : {rw} x {rh} px   Ratio : {ratio}\n"
           f"Green sample region (cyan box) = top {int(GREEN_STRIP_H*100)}%, "
           f"width {int(GREEN_STRIP_CW*100)}%, shifted right {int(GREEN_STRIP_OFFSET_PCT*100)}%")
    return msg, detect_vis, bgr_to_pil(cropped), green_vis


def test_card(card_pil):
    if card_pil is None:
        return "No image uploaded.", None, None, None, None, "—"
    ref_bgr, ref_w, ref_h = _load_reference()   # built-in fallback always available

    img_bgr    = cv2.cvtColor(np.array(card_pil), cv2.COLOR_RGB2BGR)
    detect_vis = draw_detection_box(img_bgr)

    # ── Step 1: rembg crop ────────────────────────────────────────────────────
    cropped, cw, ch = remove_background_and_crop(img_bgr)

    # ── Step 2: dimension-based rotation ─────────────────────────────────────
    if orientation_matches(cw, ch, ref_w, ref_h):
        rot, after_dim = 0, cropped
        dim_note = f"Dimension match at 0deg  (card {cw}x{ch}, ref {ref_w}x{ref_h})"
    else:
        rot = None
        for deg in (90, 180, 270):
            r = rotate90(cropped, deg // 90)
            rh_r, rw_r = r.shape[:2]
            if orientation_matches(rw_r, rh_r, ref_w, ref_h):
                rot, after_dim = deg, r
                dim_note = f"Dimension match at {deg}deg -> {rw_r}x{rh_r}"
                break
        if rot is None:
            after_dim, rot = cropped, 0
            dim_note = f"No dimension match (card {cw}x{ch}, ref {ref_w}x{ref_h}) - kept original"

    # ── Step 3: green top-center strip flip check ─────────────────────────────
    final, was_flipped, sim = fix_180_flip(after_dim, ref_bgr)
    if was_flipped:
        rot = (rot + 180) % 360
        flip_note = f"FACE-FLIP: rotated extra 180deg (score = {sim:.3f})"
    else:
        flip_note = f"FACE-OK  : no flip needed       (score = {sim:.3f})"

    green_vis = visualise_green_strip(final)   # show sampled region on final card

    summary = (
        f"Card detected : {cw} x {ch} px\n"
        f"Reference     : {ref_w} x {ref_h} px\n"
        f"Step 2 - {dim_note}\n"
        f"Step 3 - {flip_note}\n"
        f"FINAL rotation applied : {rot}deg"
    )
    return summary, detect_vis, bgr_to_pil(cropped), bgr_to_pil(final), green_vis, f"{rot}deg"


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="ID Card Rotation Tester") as demo:
    gr.Markdown(
        """
        # 🪪 ID Card Rotation Tester  (rembg + green-strip)
        **Tab 1** – Upload a correctly-oriented reference card.  
        **Tab 2** – Browse any ID card and see it auto-corrected.
        """
    )

    with gr.Tab("1️⃣  Set Reference Card"):
        gr.Markdown(
            "Upload a card already in **correct orientation** "
            "(landscape, green header with جمهورية مصر العربية at the **top**)."
        )
        ref_input = gr.Image(label="Reference card", type="pil", height=320)
        btn_ref   = gr.Button("💾  Save as Reference", variant="primary")
        ref_status = gr.Textbox(label="Status", lines=4, interactive=False)
        with gr.Row():
            ref_detect = gr.Image(label="Card detection", type="pil", height=240)
            ref_crop   = gr.Image(label="Cropped card (rembg)", type="pil", height=240)
            ref_green  = gr.Image(label="Green sample region (cyan)", type="pil", height=240)
        btn_ref.click(set_reference,
                      inputs=[ref_input],
                      outputs=[ref_status, ref_detect, ref_crop, ref_green])

    with gr.Tab("2️⃣  Test a Card"):
        gr.Markdown(
            "Upload any ID card (any orientation). "
            "The pipeline removes the background with **rembg**, "
            "fixes orientation via dimensions, "
            "then checks the **top-center green band** (جمهورية مصر العربية) "
            "to catch residual 180° flips."
        )
        card_input = gr.Image(label="Input card", type="pil", height=320)
        btn_test   = gr.Button("🔄  Detect & Rotate", variant="primary")
        out_badge  = gr.Textbox(label="Rotation applied", max_lines=1,
                                interactive=False, scale=0, min_width=160)
        result_txt = gr.Textbox(label="Step details", lines=6, interactive=False)
        with gr.Row():
            out_detect = gr.Image(label="Card detection box", type="pil", height=240)
            out_crop   = gr.Image(label="rembg crop", type="pil", height=240)
            out_final  = gr.Image(label="Final corrected", type="pil", height=240)
            out_green  = gr.Image(label="Green sample region", type="pil", height=240)
        btn_test.click(test_card,
                       inputs=[card_input],
                       outputs=[result_txt, out_detect, out_crop,
                                out_final, out_green, out_badge])

    gr.Markdown(
        "---\n"
        "**Tuning** – edit `rotation_app.py` to adjust:\n"
        "- `GREEN_STRIP_H` (default 0.22) – how tall the sample region is\n"
        "- `GREEN_STRIP_CW` (default 0.55) – how wide the centre sample is\n"
        "- `GREEN_STRIP_OFFSET_PCT` (default 0.15) – rightward shift %\n"
        "- `GREEN_SIM_THRESH` (default 0.60) – flip sensitivity"
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)

from PIL import ImageOps
def auto_rotate_id_from_path(file_path):
    # Load using PIL with EXIF transpose (like Gradio does)
    pil_img = Image.open(file_path)
    pil_img = ImageOps.exif_transpose(pil_img)
    
    # Convert to BGR
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    ref_bgr, ref_w, ref_h = _load_reference()

    cropped, cw, ch = remove_background_and_crop(img_bgr)

    if orientation_matches(cw, ch, ref_w, ref_h):
        rot, after_dim = 0, cropped
        dim_note = f"Dimension match at 0deg  (card {cw}x{ch}, ref {ref_w}x{ref_h})"
    else:
        rot = None
        for deg in (90, 180, 270):
            r = rotate90(cropped, deg // 90)
            rh_r, rw_r = r.shape[:2]
            if orientation_matches(rw_r, rh_r, ref_w, ref_h):
                rot, after_dim = deg, r
                dim_note = f"Dimension match at {deg}deg -> {rw_r}x{rh_r}"
                break
        if rot is None:
            after_dim, rot = cropped, 0
            dim_note = f"No dimension match (card {cw}x{ch}, ref {ref_w}x{ref_h}) - kept original"

    final, was_flipped, sim = fix_180_flip(after_dim, ref_bgr)
    if was_flipped:
        rot = (rot + 180) % 360
        flip_note = f"FACE-FLIP: rotated extra 180deg (score = {sim:.3f})"
    else:
        flip_note = f"FACE-OK  : no flip needed       (score = {sim:.3f})"

    note = dim_note + " | " + flip_note
    return final, rot, note
