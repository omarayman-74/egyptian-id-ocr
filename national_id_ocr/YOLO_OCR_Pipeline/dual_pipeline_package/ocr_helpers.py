import re
from datetime import datetime

_ARABIC_TO_WESTERN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
_WESTERN_TO_ARABIC = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')

def _to_western_digits(sval): return (sval or "").translate(_ARABIC_TO_WESTERN)
def _to_arabic_digits(sval):  return (sval or "").translate(_WESTERN_TO_ARABIC)
def _count_arabic_letters(sval): return len(re.findall(r'[\u0600-\u06FF]', sval or ""))
def _arabic_words(sval): return re.findall(r'[\u0600-\u06FF]{2,}', sval or "")
def _clean_name(sval):
    sval = re.sub(r'[^\u0600-\u06FF\s]', ' ', sval or "")
    return ' '.join(sval.split())

def _sanitize_addr(sval):
    sval = (sval or "").replace('؟',' ').replace('?',' ').replace('>',' ').replace('<',' ')
    sval = re.sub(r'[a-zA-Z]', ' ', sval)
    sval = re.sub(r'[^\u0600-\u06FF0-9٠-٩\s\-ـ]', ' ', sval)
    return ' '.join(sval.split())

def _extract_leading_number(sval):
    sval = _sanitize_addr(sval)
    m = re.match(r'^[0-9٠-٩]{1,6}', sval)
    return m.group(0) if m else ""

def _extract_locality_prefix(sval):
    sval = _sanitize_addr(sval)
    m = re.search(r'[م ق ك ش]|[0-9٠-٩]', sval)
    prefix = sval[:m.start()] if m else sval
    prefix = re.sub(r'[^\u0600-\u06FF\s]', ' ', prefix)
    return ' '.join(prefix.split())

def _extract_longest_arabic_phrase(sval):
    sval = _sanitize_addr(sval)
    if not sval: return ""
    tmp = re.sub(r'[0-9٠-٩]', ' ', sval)
    tmp = re.sub(r'\b[م ق ك ش]\b', ' ', tmp)
    tmp = re.sub(r'[\-ـ]', ' ', tmp)
    tmp = ' '.join(tmp.split())
    phrases = re.findall(r'[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,}){0,3}', tmp)
    if not phrases: return ""
    phrases.sort(key=lambda p: (_count_arabic_letters(p), len(p.split()), len(p)), reverse=True)
    return phrases[0]

def _extract_all_locality_parts(sval):
    sval = _sanitize_addr(sval)
    cleaned = re.sub(r'\b[م ق ك ش]\s*[\-ـ:]?\s*[0-9٠-٩]+', ' ', sval)
    cleaned = re.sub(r'\b[0-9٠-٩]+\b', ' ', cleaned)
    cleaned = re.sub(r'\b[مقكش]\b', ' ', cleaned)
    cleaned = re.sub(r'[\-ـ]+', ' ', cleaned)
    cleaned = re.sub(r'[^\u0600-\u06FF\s]', ' ', cleaned)
    return ' '.join(cleaned.split()).strip()

def _extract_marker_number(sval, marker):
    sval = _sanitize_addr(sval)
    m = re.search(rf'(?:^|[\s\-ـ]){marker}\s*[\-ـ:]?\s*([0-9٠-٩]{{1,3}})', sval)
    return m.group(1) if m else ""

def _closest_number_after_marker(sval, marker):
    sval = _sanitize_addr(sval)
    marker_idx = -1
    for match in re.finditer(rf'(?:^|[\s\-ـ]){marker}(?:[\s\-ـ]|$)', sval):
        marker_idx = match.start() + (1 if match.group(0)[0] in ' \-ـ' else 0)
        break
    if marker_idx == -1: return ""
    best = None
    for m in re.finditer(r'[0-9٠-٩]{2,3}', sval):
        dist = abs(m.start() - marker_idx)
        if best is None or dist < best[0]: best = (dist, m.group(0))
    return best[1] if best else ""

def _best_number(num_t, num_e, all_t, all_e):
    candidates = [x for x in [num_t, num_e] if x]
    if not candidates: return ""
    best = sorted(candidates, key=len, reverse=True)[0]
    if len(best) >= 2: return best
    singles = [d for d in (all_t+all_e) if len(d)==1 and d!=best]
    return best + singles[0] if singles else best

def _extract_city_district(sval):
    known_cities = ['اكتوبر','القاهرة','الجيزة','الاسكندرية','الاسماعيلية',
        'بورسعيد','السويس','المنصورة','طنطا','الزقازيق','اسيوط','الفيوم',
        'بنها','دمياط','اسوان','الاقصر','قنا','سوهاج','المنيا','كفر الشيخ',
        'الدقهلية','الشرقية','الغربية','القليوبية','البحيرة','مطروح']
    sval = _sanitize_addr(sval)
    words = sval.split()
    for i, word in enumerate(words):
        for city in known_cities:
            if city in word or word in city:
                return ' '.join(words[i:])
    if len(words) >= 2: return ' '.join(words[-2:])
    elif len(words) == 1: return words[0]
    return ""

def _extract_area_name(sval, city):
    sval = _sanitize_addr(sval)
    if city: sval = sval.replace(city, ' ')
    sval = re.sub(r'\b[مقكش]\s*[\-ـ:]?\s*[0-9٠-٩]+', ' ', sval)
    sval = re.sub(r'\b[0-9٠-٩]+\b', ' ', sval)
    sval = re.sub(r'\b[مقكش]\b', ' ', sval)
    sval = re.sub(r'[\-ـ]+', ' ', sval)
    return ' '.join(sval.split()).strip()

def choose_address(addr_tesseract, addr_easyocr):
    addr_t = _sanitize_addr(addr_tesseract)
    addr_e = _sanitize_addr(addr_easyocr)
    if not addr_t and not addr_e: return "0"
    city_t = _extract_city_district(addr_t); city_e = _extract_city_district(addr_e)
    city   = city_t if _count_arabic_letters(city_t) >= _count_arabic_letters(city_e) else city_e
    area_t = _extract_area_name(addr_t, city); area_e = _extract_area_name(addr_e, city)
    area   = area_t if _count_arabic_letters(area_t) >= _count_arabic_letters(area_e) else area_e
    markers = {}
    for marker in ['م','ق','ك','ش']:
        m_t = _extract_marker_number(addr_t, marker)
        m_e = _extract_marker_number(addr_e, marker)
        if not m_t and not m_e: continue
        m2_t = _closest_number_after_marker(addr_t, marker)
        m2_e = _closest_number_after_marker(addr_e, marker)
        all_t = [_to_western_digits(x) for x in re.findall(r'[0-9٠-٩]+', addr_t)]
        all_e = [_to_western_digits(x) for x in re.findall(r'[0-9٠-٩]+', addr_e)]
        best  = _best_number(_to_western_digits(m2_t or m_t), _to_western_digits(m2_e or m_e), all_t, all_e)
        if len(best)==1:
            twos = [d for d in (all_t+all_e) if len(d)==2]
            if twos: best = twos[-1]
        if best: markers[marker] = _to_arabic_digits(best)
    if len(markers)==0:   result = f"{area} {city}".strip() or addr_e or addr_t
    elif len(markers)==1:
        mk,num = list(markers.items())[0]
        result = f"{area} {mk} {num} {city}".strip()
    elif len(markers)==2:
        items  = list(markers.items())
        result = f"{area} {items[0][0]} {items[0][1]} -{items[1][0]} {items[1][1]} {city}".strip()
    else:
        result = f"{area} {' -'.join(f'{k} {v}' for k,v in markers.items())} {city}".strip()
    lead_t = _extract_leading_number(addr_t); lead_e = _extract_leading_number(addr_e)
    lead   = lead_t if len(lead_t)>=len(lead_e) else lead_e
    if lead:
        lead = _to_arabic_digits(_to_western_digits(lead))
        if not result.startswith(lead): result = f"{lead} {result}".strip()
    result = re.sub(r'[a-zA-Z]', '', result)
    result = re.sub(r'[^\u0600-\u06FF0-9٠-٩\s\-ـ]', ' ', result)
    return ' '.join(result.split())

def _extract_birthdate_from_id(id_value):
    digits = re.sub(r'\D', '', _to_western_digits(str(id_value)))
    if len(digits) < 7: return "0"
    if digits[0] in ('2','3') and len(digits) >= 7:
        century = 1900 if digits[0]=='2' else 2000
        try:
            dt = datetime(century+int(digits[1:3]), int(digits[3:5]), int(digits[5:7]))
            return dt.strftime('%Y-%m-%d')
        except Exception: return "0"
    if len(digits) >= 6:
        yy = int(digits[0:2]); mm = int(digits[2:4]); dd = int(digits[4:6])
        century = 2000 if yy <= (datetime.now().year % 100) else 1900
        try:
            dt = datetime(century+yy, mm, dd)
            return dt.strftime('%Y-%m-%d')
        except Exception: return "0"
    return "0"

def arabic_to_english_numbers(text):
    return text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789'))


