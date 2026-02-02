import re
import string
from datetime import datetime

# Constants
ARABIC_DIGITS = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
PUN = set(string.punctuation)
_ARABIC_TO_WESTERN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
_WESTERN_TO_ARABIC = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')

def to_western_digits(sval: str) -> str:
    return (sval or "").translate(_ARABIC_TO_WESTERN)

def to_arabic_digits(sval: str) -> str:
    return (sval or "").translate(_WESTERN_TO_ARABIC)

def count_arabic_letters(sval: str) -> int:
    return len(re.findall(r'[\u0600-\u06FF]', sval or ""))

def arabic_words(sval: str) -> list[str]:
    return re.findall(r'[\u0600-\u06FF]{2,}', sval or "")

def clean_name(sval: str) -> str:
    sval = re.sub(r'[^\u0600-\u06FF\s]', ' ', sval or "")
    return ' '.join(sval.split())

def best_text(text_a: str, text_b: str) -> str:
    a = text_a.strip() if text_a else ""
    b = text_b.strip() if text_b else ""
    if not a and not b:
        return ""
    if count_arabic_letters(a) != count_arabic_letters(b):
        return a if count_arabic_letters(a) > count_arabic_letters(b) else b
    return a if len(a) >= len(b) else b

def sanitize_addr(sval: str) -> str:
    """Keep Arabic letters, digits and common separators; drop OCR garbage like > ؟ etc."""
    sval = (sval or "").replace('؟', ' ').replace('?', ' ').replace('>', ' ').replace('<', ' ')
    sval = re.sub(r'[^\u0600-\u06FF0-9٠-٩\s\-ـ]', ' ', sval)
    sval = ' '.join(sval.split())
    return sval

def extract_locality_prefix(sval: str) -> str:
    """Return the part before numbers/markers (often the area name)."""
    sval = sanitize_addr(sval)
    m = re.search(r'[مق]|[0-9٠-٩]', sval)
    prefix = sval[:m.start()] if m else sval
    prefix = re.sub(r'[^\u0600-\u06FF\s]', ' ', prefix)
    prefix = ' '.join(prefix.split())
    return prefix

def extract_longest_arabic_phrase(sval: str) -> str:
    """Pick the longest phrase of Arabic words (ignoring markers/digits)."""
    sval = sanitize_addr(sval)
    if not sval:
        return ""
    tmp = re.sub(r'[0-9٠-٩]', ' ', sval)
    tmp = re.sub(r'\b[مق]\b', ' ', tmp)
    tmp = re.sub(r'[\-ـ]', ' ', tmp)
    tmp = ' '.join(tmp.split())
    phrases = re.findall(r'[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,}){0,3}', tmp)
    if not phrases:
        return ""
    phrases = [' '.join(p.split()) for p in phrases]
    phrases.sort(key=lambda p: (count_arabic_letters(p), len(p.split()), len(p)), reverse=True)
    return phrases[0]

def pick_locality(addr_t: str, addr_e: str) -> str:
    cands = []
    for s0 in [addr_t, addr_e]:
        cands.append(extract_locality_prefix(s0))
        cands.append(extract_longest_arabic_phrase(s0))
    cands = [c.strip() for c in cands if c and c.strip()]
    if not cands:
        return ""
    cands = list(dict.fromkeys(cands))
    cands.sort(key=lambda p: (count_arabic_letters(p), len(p.split()), len(p)), reverse=True)
    best = cands[0]
    if count_arabic_letters(best) < 3:
        raw = sanitize_addr(addr_t)
        raw2 = sanitize_addr(addr_e)
        best = raw if count_arabic_letters(raw) >= count_arabic_letters(raw2) else raw2
    return best.strip()

def extract_marker_number(sval: str, marker: str) -> str:
    sval = sanitize_addr(sval)
    m = re.search(rf'(?:^|[\s\-ـ]){marker}\s*[\-ـ:]?\s*([0-9٠-٩]{{1,3}})', sval)
    return m.group(1) if m else ""

def closest_number_after_marker(sval: str, marker: str) -> str:
    """Pick the nearest 2-3 digit group to the marker, if it exists."""
    sval = sanitize_addr(sval)
    marker_idx = -1
    for match in re.finditer(rf'(?:^|[\s\-ـ]){marker}(?:[\s\-ـ]|$)', sval):
        marker_idx = match.start() + (1 if match.group(0)[0] in ' \-ـ' else 0)
        break
    
    if marker_idx == -1:
        return ""
    best = None
    for m in re.finditer(r'[0-9٠-٩]{2,3}', sval):
        dist = abs(m.start() - marker_idx)
        cand = m.group(0)
        if best is None or dist < best[0]:
            best = (dist, cand)
    return best[1] if best else ""

def best_number(num_t: str, num_e: str, all_digits_t: list[str], all_digits_e: list[str]) -> str:
    """Prefer longer numbers (2-3 digits). If only 1 digit exists, try to build 2 digits from the other OCR digits."""
    candidates = [x for x in [num_t, num_e] if x]
    if not candidates:
        return ""
    candidates_sorted = sorted(candidates, key=len, reverse=True)
    best = candidates_sorted[0]
    if len(best) >= 2:
        return best
    singles = [d for d in (all_digits_t + all_digits_e) if len(d) == 1 and d != best]
    if singles:
        return best + singles[0]
    return best

def choose_address(addr_tesseract: str, addr_easyocr: str) -> str:
    """Build a clean address using BOTH OCR outputs with multiple marker types."""
    addr_t = sanitize_addr(addr_tesseract)
    addr_e = sanitize_addr(addr_easyocr)
    if not addr_t and not addr_e:
        return "0"
    locality = pick_locality(addr_t, addr_e) or addr_t or addr_e
    
    markers = {}
    possible_markers = ['م', 'ق', 'ك']
    
    for marker in possible_markers:
        m_t = extract_marker_number(addr_t, marker)
        m_e = extract_marker_number(addr_e, marker)
        
        if not m_t and not m_e:
            continue
            
        m2_t = closest_number_after_marker(addr_t, marker)
        m2_e = closest_number_after_marker(addr_e, marker)
        
        all_t = [to_western_digits(x) for x in re.findall(r'[0-9٠-٩]+', addr_t)]
        all_e = [to_western_digits(x) for x in re.findall(r'[0-9٠-٩]+', addr_e)]
        
        best = best_number(to_western_digits(m2_t or m_t), to_western_digits(m2_e or m_e), all_t, all_e)
        
        if len(best) == 1:
            twos = [d for d in (all_t + all_e) if len(d) == 2]
            if twos:
                best = twos[-1]
        
        if best:
            markers[marker] = to_arabic_digits(best)
    
    locality = ' '.join(locality.split())
    
    if len(markers) == 0:
        return locality.strip() or (addr_e or addr_t)
    elif len(markers) == 1:
        marker, number = list(markers.items())[0]
        return f"{locality} {marker} {number}".strip()
    elif len(markers) == 2:
        items = list(markers.items())
        return f"{locality} {items[0][0]} {items[0][1]} -{items[1][0]} {items[1][1]}".strip()
    else:
        marker_str = ' -'.join([f"{k} {v}" for k, v in markers.items()])
        return f"{locality} {marker_str}".strip()

def extract_birthdate_from_id(id_value: str) -> str:
    digits = re.sub(r'\D', '', to_western_digits(str(id_value)))
    if len(digits) < 7:
        return "0"
    if len(digits) >= 7 and digits[0] in ('2', '3') and len(digits) >= 7:
        century = 1900 if digits[0] == '2' else 2000
        yy = int(digits[1:3])
        mm = int(digits[3:5])
        dd = int(digits[5:7])
        try:
            dt = datetime(century + yy, mm, dd)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return "0"
    if len(digits) >= 6:
        yy = int(digits[0:2])
        mm = int(digits[2:4])
        dd = int(digits[4:6])
        century = 2000 if yy <= (datetime.now().year % 100) else 1900
        try:
            dt = datetime(century + yy, mm, dd)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return "0"
    return "0"

def clean_id(id_val: str) -> int:
    ar=['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
    pun = set(string.punctuation)
    s_val = str(id_val)
    for i in s_val:
        if i in ar or i in pun:
            s_val = s_val.replace(i, "")

    matches = re.findall(r'[٠-٩]+', s_val)
    matches.reverse()
    concatenated_string = ''.join(matches)
    
    if concatenated_string:
        return int(concatenated_string.translate(_ARABIC_TO_WESTERN))
    
    return int(re.sub(r'\D', '', s_val)) if re.sub(r'\D', '', s_val) else 0
