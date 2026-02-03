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
    # Remove English letters (a-z, A-Z)
    sval = re.sub(r'[a-zA-Z]', ' ', sval)
    # Remove common OCR artifacts and special characters, keep only Arabic, digits, spaces, and separators
    sval = re.sub(r'[^\u0600-\u06FF0-9٠-٩\s\-ـ]', ' ', sval)
    # Remove excessive spaces
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

def extract_all_locality_parts(sval: str) -> str:
    """Extract all Arabic locality parts (area, district, governorate) excluding standalone markers."""
    sval = sanitize_addr(sval)
    # Remove standalone markers (م, ق, ك) when they're followed by numbers
    cleaned = re.sub(r'\b[مقك]\s*[\-ـ:]?\s*[0-9٠-٩]+', ' ', sval)
    # Remove all remaining standalone numbers
    cleaned = re.sub(r'\b[0-9٠-٩]+\b', ' ', cleaned)
    # Remove standalone single letter markers that remain
    cleaned = re.sub(r'\b[مقك]\b', ' ', cleaned)
    # Clean up extra separators
    cleaned = re.sub(r'[\-ـ]+', ' ', cleaned)
    # Keep only Arabic letters and spaces
    cleaned = re.sub(r'[^\u0600-\u06FF\s]', ' ', cleaned)
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()

def extract_city_district(sval: str) -> str:
    """Extract city/district/governorate name from address."""
    known_cities = [
        'اكتوبر', '6 اكتوبر', 'القاهرة', 'الجيزة', 'الاسكندرية', 'الاسماعيلية',
        'بورسعيد', 'السويس', 'المنصورة', 'طنطا', 'الزقازيق', 'اسيوط', 'الفيوم',
        'بنها', 'دمياط', 'اسوان', 'الاقصر', 'قنا', 'سوهاج', 'المنيا', 'كفر الشيخ',
        'الدقهلية', 'الشرقية', 'الغربية', 'القليوبية', 'البحيرة', 'مطروح'
    ]
    sval = sanitize_addr(sval)
    words = sval.split()
    
    # Look for known city names
    for i, word in enumerate(words):
        for city in known_cities:
            if city in word or word in city:
                # Return from this position to end
                return ' '.join(words[i:])
    
    # If no known city found, return last 1-2 words
    if len(words) >= 2:
        return ' '.join(words[-2:])
    elif len(words) == 1:
        return words[0]
    return ""

def extract_area_name(sval: str, city: str) -> str:
    """Extract area name (first part before city/markers)."""
    sval = sanitize_addr(sval)
    # Remove city name
    if city:
        sval = sval.replace(city, ' ')
    # Remove markers and numbers
    sval = re.sub(r'\b[مقك]\s*[\-ـ:]?\s*[0-9٠-٩]+', ' ', sval)
    sval = re.sub(r'\b[0-9٠-٩]+\b', ' ', sval)
    sval = re.sub(r'\b[مقك]\b', ' ', sval)
    sval = re.sub(r'[\-ـ]+', ' ', sval)
    sval = ' '.join(sval.split())
    return sval.strip()

def pick_locality(addr_t: str, addr_e: str) -> str:
    cands = []
    for s0 in [addr_t, addr_e]:
        # Try to extract all locality parts (including governorate)
        full_locality = extract_all_locality_parts(s0)
        if full_locality:
            cands.append(full_locality)
        # Also try traditional methods as fallback
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
    
    # Extract city/district from both sources
    city_t = extract_city_district(addr_t)
    city_e = extract_city_district(addr_e)
    city = city_t if count_arabic_letters(city_t) >= count_arabic_letters(city_e) else city_e
    
    # Extract area name (without city)
    area_t = extract_area_name(addr_t, city)
    area_e = extract_area_name(addr_e, city)
    area = area_t if count_arabic_letters(area_t) >= count_arabic_letters(area_e) else area_e
    
    # Support multiple marker types: م (meem), ق (qaf), ك (kaf)
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
    
    result = ""
    if len(markers) == 0:
        result = f"{area} {city}".strip() or addr_e or addr_t
    elif len(markers) == 1:
        marker, number = list(markers.items())[0]
        result = f"{area} {marker} {number} {city}".strip()
    elif len(markers) == 2:
        items = list(markers.items())
        result = f"{area} {items[0][0]} {items[0][1]} -{items[1][0]} {items[1][1]} {city}".strip()
    else:
        marker_str = ' -'.join([f"{k} {v}" for k, v in markers.items()])
        result = f"{area} {marker_str} {city}".strip()
    
    # Final cleanup: remove any remaining English letters or special characters
    result = re.sub(r'[a-zA-Z]', '', result)
    result = re.sub(r'[^\u0600-\u06FF0-9٠-٩\s\-ـ]', ' ', result)
    result = ' '.join(result.split())
    return result

def remove_cross_line_duplicates(address: str) -> str:
    """Remove duplicate words that appear in multiple parts of the address.
    If a word appears in multiple parts, keep it only in the last occurrence."""
    if not address or address == "0":
        return address
    
    # Normalize: separate dashes and markers to ensure consistent tokenization
    # Replace "-ق" with "- ق", etc.
    normalized = address
    for marker in ['م', 'ق', 'ك']:
        normalized = normalized.replace(f'-{marker}', f'- {marker}')
        normalized = normalized.replace(f'ـ{marker}', f'ـ {marker}')
    
    # Split by spaces
    parts = re.split(r'\s+', normalized)
    if len(parts) <= 1:
        return address
    
    # Track words and marker+number combinations
    word_positions = {}
    i = 0
    while i < len(parts):
        word = parts[i]
        
        # Check if this token contains a marker+number combo (e.g., "م٢٦" or "ق٢٦")
        marker_num_match = re.match(r'^([مقك])([0-9٠-٩]+)$', word)
        if marker_num_match:
            marker = marker_num_match.group(1)
            number = marker_num_match.group(2)
            combo = f"{marker} {number}"
            if combo not in word_positions:
                word_positions[combo] = []
            word_positions[combo].append(i)
            i += 1
            continue
        
        # Check if this is a marker followed by a number in the next token
        if word in ['م', 'ق', 'ك'] and i + 1 < len(parts) and re.match(r'^[0-9٠-٩]+$', parts[i + 1]):
            combo = f"{word} {parts[i + 1]}"
            if combo not in word_positions:
                word_positions[combo] = []
            word_positions[combo].append((i, i + 1))
            i += 2
            continue
        
        # Skip standalone markers, numbers, and separators
        if word in ['م', 'ق', 'ك', '-', 'ـ'] or re.match(r'^[0-9٠-٩]+$', word):
            i += 1
            continue
        
        # Regular word
        if word not in word_positions:
            word_positions[word] = []
        word_positions[word].append(i)
        i += 1
    
    # Mark positions to remove (keep only the last occurrence of duplicates)
    positions_to_remove = set()
    for item, positions in word_positions.items():
        if len(positions) > 1:
            for pos in positions[:-1]:
                if isinstance(pos, tuple):
                    positions_to_remove.add(pos[0])
                    positions_to_remove.add(pos[1])
                else:
                    positions_to_remove.add(pos)
    
    # Rebuild address without removed positions
    cleaned_parts = [parts[i] for i in range(len(parts)) if i not in positions_to_remove]
    return ' '.join(cleaned_parts)

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
