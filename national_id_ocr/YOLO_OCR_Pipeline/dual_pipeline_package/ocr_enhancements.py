"""
OCR Enhancement Helpers
=======================
Fix #2: NID Validation (14-digit check)
Fix #3: Adaptive Lighting Enhancement
"""

import cv2
import numpy as np
import re


# =============================================================================
#  FIX #2: National ID Validation
# =============================================================================

def validate_national_id(raw_id):
    """
    Validate and fix Egyptian National ID to exactly 14 digits.
    
    Rules:
    - Must be exactly 14 digits
    - First digit must be 2 (1900s) or 3 (2000s)
    - Digits 4-5 (month) must be 01-12
    - Digits 6-7 (day) must be 01-31
    
    If 15 digits: tries dropping first or last digit.
    If 13 digits: tries common OCR fixes (duplicate detection).
    """
    digits = re.sub(r'\D', '', str(raw_id))
    
    if len(digits) == 14 and _is_valid_nid(digits):
        return digits
    
    # Try fix for 15 digits (extra digit picked up)
    if len(digits) == 15:
        # Try dropping first digit
        candidate1 = digits[1:]
        # Try dropping last digit
        candidate2 = digits[:-1]
        
        for candidate in [candidate1, candidate2]:
            if _is_valid_nid(candidate):
                return candidate
    
    # Try fix for 16 digits
    if len(digits) == 16:
        for start in range(3):
            candidate = digits[start:start+14]
            if _is_valid_nid(candidate):
                return candidate
    
    # Try fix for 13 digits (missing digit)
    if len(digits) == 13:
        # Try prepending likely first digits
        for prefix in ['2', '3']:
            candidate = prefix + digits
            if _is_valid_nid(candidate):
                return candidate
    
    # Return original if no fix works
    return digits


def _is_valid_nid(digits):
    """Check if a 14-digit string is a valid Egyptian National ID."""
    if len(digits) != 14:
        return False
    if digits[0] not in ('2', '3'):
        return False
    try:
        month = int(digits[3:5])
        day = int(digits[5:7])
        gov = digits[7:9]
        if not (1 <= month <= 12):
            return False
        if not (1 <= day <= 31):
            return False
        # Known governorate codes
        valid_govs = {'01','02','03','04','11','12','13','14','15','16','17',
                      '18','19','21','22','23','24','25','26','27','28','29',
                      '31','32','33','34','35','88'}
        if gov not in valid_govs:
            return False
        return True
    except (ValueError, IndexError):
        return False


# =============================================================================
#  FIX #3: Adaptive Lighting Enhancement
# =============================================================================

def assess_lighting(img_bgr):
    """
    Measure image lighting quality.
    
    Returns:
        dict with:
        - brightness: 0-255 (average luminance)
        - contrast: standard deviation of luminance
        - is_dark: True if image needs brightening
        - is_overexposed: True if image is too bright
        - is_low_contrast: True if contrast is poor
        - glare_ratio: percentage of very bright pixels (potential glare)
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    brightness = float(np.mean(l_channel))
    contrast = float(np.std(l_channel))
    
    # Detect glare: percentage of pixels with L > 230
    total_pixels = l_channel.size
    glare_pixels = np.sum(l_channel > 230)
    glare_ratio = float(glare_pixels / total_pixels) * 100
    
    # Detect uneven lighting: compare brightness of quadrants
    h, w = l_channel.shape
    quadrants = [
        l_channel[:h//2, :w//2],    # top-left
        l_channel[:h//2, w//2:],    # top-right
        l_channel[h//2:, :w//2],    # bottom-left
        l_channel[h//2:, w//2:],    # bottom-right
    ]
    quad_means = [float(np.mean(q)) for q in quadrants]
    unevenness = max(quad_means) - min(quad_means)
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'is_dark': brightness < 100,
        'is_overexposed': brightness > 200,
        'is_low_contrast': contrast < 30,
        'glare_ratio': glare_ratio,
        'unevenness': unevenness,
        'is_uneven': unevenness > 40,
        'quadrant_means': quad_means,
    }


def enhance_lighting(img_bgr, assessment=None):
    """
    Adaptively enhance image lighting based on measured conditions.
    
    Steps:
    1. Measure lighting (if not already provided)
    2. Shadow removal for uneven lighting
    3. Adaptive CLAHE based on brightness/contrast levels
    4. Glare reduction if significant glare detected
    5. Sharpening pass for better OCR
    
    Returns:
        enhanced_bgr: the enhanced image
        assessment: the lighting assessment dict
    """
    if assessment is None:
        assessment = assess_lighting(img_bgr)
    
    enhanced = img_bgr.copy()
    
    # Step 1: Shadow removal for uneven lighting
    if assessment['is_uneven'] or assessment['is_dark']:
        enhanced = _remove_shadows_adaptive(enhanced)
    
    # Step 2: Adaptive CLAHE
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    if assessment['is_dark']:
        # Dark image: aggressive CLAHE + brightness boost
        clip_limit = 3.5
        l_channel = cv2.add(l_channel, 30)  # Boost brightness
    elif assessment['is_overexposed']:
        # Overexposed: gentle CLAHE + reduce brightness
        clip_limit = 1.0
        l_channel = cv2.subtract(l_channel, 20)
    elif assessment['is_low_contrast']:
        # Low contrast: moderate CLAHE
        clip_limit = 2.5
    else:
        # Normal: light CLAHE
        clip_limit = 1.5
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    
    enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    
    # Step 3: Glare reduction
    if assessment['glare_ratio'] > 2.0:
        enhanced = _reduce_glare(enhanced)
    
    # Step 4: Light sharpening for OCR
    enhanced = _sharpen_for_ocr(enhanced)
    
    return enhanced, assessment


def _remove_shadows_adaptive(img_bgr):
    """Remove shadows using morphological background estimation."""
    result_channels = []
    h, w = img_bgr.shape[:2]
    kh = max(10, h // 4)
    kw = max(10, w // 4)
    kh = kh if kh % 2 == 1 else kh + 1
    kw = kw if kw % 2 == 1 else kw + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
    
    for ch in cv2.split(img_bgr):
        bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)
        norm = (ch.astype(np.float32) / (bg.astype(np.float32) + 1e-6)) * 200.0
        result_channels.append(np.clip(norm, 0, 255).astype(np.uint8))
    
    return cv2.merge(result_channels)


def _reduce_glare(img_bgr):
    """Reduce glare spots by clamping very bright pixels."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Find glare regions (very bright spots)
    glare_mask = l_channel > 230
    
    if np.any(glare_mask):
        # Inpaint the glare regions using surrounding pixels
        mask_uint8 = glare_mask.astype(np.uint8) * 255
        # Dilate mask slightly to cover edges of glare
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        # Inpaint
        img_bgr = cv2.inpaint(img_bgr, mask_uint8, 5, cv2.INPAINT_TELEA)
    
    return img_bgr


def _sharpen_for_ocr(img_bgr):
    """Light sharpening to improve OCR character recognition."""
    gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 1.5)
    sharpened = cv2.addWeighted(img_bgr, 1.3, gaussian, -0.3, 0)
    return sharpened


# =============================================================================
#  COMBINED: Apply all enhancements before OCR
# =============================================================================

def preprocess_for_ocr(img_bgr, verbose=True):
    """
    Full preprocessing pipeline: assess lighting, enhance, and prepare for OCR.
    
    Args:
        img_bgr: Input image (BGR format)
        verbose: If True, print diagnostic info
        
    Returns:
        enhanced_bgr: Enhanced image ready for OCR
        assessment: Lighting assessment results
    """
    assessment = assess_lighting(img_bgr)
    
    if verbose:
        print(f"  [Lighting] Brightness: {assessment['brightness']:.1f}/255")
        print(f"  [Lighting] Contrast:   {assessment['contrast']:.1f}")
        print(f"  [Lighting] Glare:      {assessment['glare_ratio']:.1f}%")
        print(f"  [Lighting] Unevenness: {assessment['unevenness']:.1f}")
        
        issues = []
        if assessment['is_dark']:       issues.append("DARK")
        if assessment['is_overexposed']:issues.append("OVEREXPOSED")
        if assessment['is_low_contrast']:issues.append("LOW CONTRAST")
        if assessment['glare_ratio']>2: issues.append("GLARE")
        if assessment['is_uneven']:     issues.append("UNEVEN LIGHT")
        
        if issues:
            print(f"  [Lighting] Issues:     {', '.join(issues)}")
            print(f"  [Lighting] Applying adaptive enhancement...")
        else:
            print(f"  [Lighting] Quality:    GOOD (minor enhancement applied)")
    
    enhanced, assessment = enhance_lighting(img_bgr, assessment)
    return enhanced, assessment


# =============================================================================
#  TEST (run this file directly to test)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_enhancements.py <image_path>")
        print("\nTests:")
        
        # Test NID validation
        test_cases = [
            ("27308220102404", "Valid 14-digit"),
            ("227308220102404", "15 digits - extra leading 2"),
            ("7308220102404", "13 digits - missing first"),
            ("2730822010240", "13 digits - missing last"),
        ]
        print("\n--- NID Validation Tests ---")
        for raw, desc in test_cases:
            fixed = validate_national_id(raw)
            status = "✅" if len(fixed) == 14 and _is_valid_nid(fixed) else "❌"
            print(f"  {status} {desc}: {raw} → {fixed}")
        
        sys.exit(0)
    
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Cannot read: {sys.argv[1]}")
        sys.exit(1)
    
    enhanced, assessment = preprocess_for_ocr(img, verbose=True)
    cv2.imwrite("enhanced_output.jpg", enhanced)
    print(f"\nSaved enhanced image to: enhanced_output.jpg")
