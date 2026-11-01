"""End-to-end pipeline test against a running deployment, using the real Sarvam APIs.

Voice cases: Bulbul speaks the answer -> the WAV goes to /api/answer -> Saaras -> extraction
-> validation -> transliteration/translation. Typed cases cover rejections and other languages.

Usage: python tests/e2e_test.py [base_url]
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE = (sys.argv[1] if len(sys.argv) > 1 else "https://sarvamaihackathon.vercel.app").rstrip("/")
http = httpx.Client(timeout=90)

eq = lambda v: (lambda got: got == v, v)
has = lambda s: (lambda got: s.lower() in (got or "").lower(), f"contains '{s}'")
starts = lambda *ss: (lambda got: (got or "").lower().startswith(tuple(ss)), "starts " + "/".join(ss))

# (form, field, spoken Hindi answer, expectation)
VOICE = [
    ("income", "name", "मेरा नाम रमेश कुमार यादव है", has("ramesh")),
    ("income", "father", "मेरे पिता का नाम सुरेश यादव है", has("suresh")),
    ("income", "dob", "मेरी जन्म तिथि पंद्रह अगस्त उन्नीस सौ पचानवे है", eq("1995-08-15")),
    ("income", "gender", "मैं महिला हूँ", eq("female")),
    ("income", "mobile", "मेरा नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है", eq("9876543210")),
    ("income", "aadhaar", "मेरा आधार नंबर दो तीन चार पाँच छह सात आठ नौ शून्य एक दो तीन है", eq("234567890123")),
    ("income", "district", "मैं लखनऊ ज़िले में रहता हूँ", starts("lakh", "luck")),
    ("income", "pincode", "मेरा पिन कोड दो दो छह शून्य शून्य एक है", eq("226001")),
    ("income", "occupation", "मैं दर्ज़ी का काम करता हूँ", eq("Tailor")),
    ("income", "income", "हमारी सालाना आमदनी डेढ़ लाख रुपये है", eq("150000")),
    ("income", "purpose", "बेटी के कॉलेज में दाखिले के लिए", eq("admission")),
    ("ration", "name", "परिवार के मुखिया का नाम सुनीता देवी है", has("sunita")),
    ("ration", "mobile", "सात शून्य एक दो तीन चार पाँच छह सात आठ", eq("7012345678")),
    ("ration", "aadhaar", "आठ सात छह पाँच चार तीन दो एक शून्य नौ आठ सात", eq("876543210987")),
    ("ration", "family_size", "हमारे परिवार में पाँच लोग हैं", eq("5")),
    ("ration", "address", "हम रामपुर गाँव में रहते हैं", eq("Rampur")),
    ("ration", "district", "बरेली ज़िला", (lambda g: (g or "").lower().startswith("bare") and " " not in (g or ""), "Bareilly, no generic suffix")),
    ("ration", "pincode", "दो चार तीन शून्य शून्य एक", eq("243001")),
    ("ration", "income", "अस्सी हज़ार रुपये साल के", eq("80000")),
    ("ration", "category", "हम बीपीएल परिवार हैं", eq("bpl")),
    ("kisan", "name", "मेरा नाम हरप्रीत सिंह है", has("singh")),
    ("kisan", "aadhaar", "तीन चार पाँच छह सात आठ नौ शून्य एक दो तीन चार", eq("345678901234")),
    ("kisan", "mobile", "आठ आठ आठ सात सात सात छह छह छह पाँच", eq("8887776665")),
    ("kisan", "state", "उत्तर प्रदेश", has("uttar")),
    ("kisan", "district", "मेरा ज़िला मेरठ है", starts("mer", "meer")),
    ("kisan", "village", "मेरे गाँव का नाम सरधना है", starts("sar")),
    ("kisan", "land", "मेरे पास ढाई एकड़ ज़मीन है", eq("2.5")),
    ("kisan", "crop", "मैं गेहूँ उगाता हूँ", eq("Wheat")),
]

# (form, field, lang, typed answer, expect_ok, expectation or None)
TYPED = [
    ("income", "aadhaar", "hi", "1234 5678 9012", False, None),  # Aadhaar can't start with 1
    ("income", "aadhaar", "hi", "23456789", False, None),  # too short
    ("income", "mobile", "hi", "5876543210", False, None),  # mobiles start 6-9
    ("income", "pincode", "hi", "40001", False, None),  # 5 digits
    ("income", "dob", "hi", "15 अगस्त 2090", False, None),  # future date
    ("income", "dob", "hi", "पिछले साल", False, None),  # incomplete date, must not guess
    ("ration", "family_size", "hi", "500", False, None),  # out of range
    ("ration", "category", "hi", "पता नहीं", False, None),  # no valid choice
    ("income", "name", "ta", "என் பெயர் முருகன் செல்வம்", True, has("murugan")),
    ("income", "occupation", "mr", "मी शेतकरी आहे", True, has("farmer")),
    ("income", "name", "bn", "আমার নাম অমিত দাস", True, has("amit")),
    ("kisan", "crop", "te", "నేను వరి పండిస్తాను", True, (lambda g: any(w in (g or "").lower() for w in ("paddy", "rice")), "paddy/rice")),
    ("income", "name", "en", "Priya Sharma", True, eq("Priya Sharma")),
]


def voice_case(c):
    form, field, said, (check, want) = c
    t0 = time.time()
    tts = http.post(f"{BASE}/api/tts", json={"text": said, "lang": "hi"})
    tts.raise_for_status()
    import base64

    wav = base64.b64decode(tts.json()["audio"])
    r = http.post(
        f"{BASE}/api/answer",
        data={"form_id": form, "field_id": field, "lang": "hi"},
        files={"audio": ("a.wav", wav, "audio/wav")},
    ).json()
    got = r.get("value")
    ok = bool(r.get("ok")) and check(got)
    return ok, f"{form}.{field}", r.get("transcript", ""), got, r.get("status", r.get("message_en", "")), want, time.time() - t0


def typed_case(c):
    form, field, lang, text, expect_ok, exp = c
    t0 = time.time()
    r = http.post(f"{BASE}/api/answer", data={"form_id": form, "field_id": field, "lang": lang, "text": text}).json()
    got = r.get("value")
    if expect_ok:
        check, want = exp
        ok = bool(r.get("ok")) and check(got)
    else:
        ok, want = (r.get("ok") is False), "rejected + re-ask"
    shown = got if r.get("ok") else "REJECTED: " + r.get("message_en", "")
    return ok, f"{form}.{field} [{lang}]", text, shown, r.get("status", ""), want, time.time() - t0


def main():
    health = http.get(f"{BASE}/api/health").json()
    print(f"Target: {BASE}  health={health}\n")
    with ThreadPoolExecutor(max_workers=6) as pool:
        v = list(pool.map(voice_case, VOICE))
        t = list(pool.map(typed_case, TYPED))
    passed = 0
    for title, rows in (("VOICE: Bulbul -> Saaras -> Sarvam-105B -> validate -> English", v), ("TYPED: validation + other languages", t)):
        print(f"== {title}")
        for ok, name, heard, got, status, want, secs in rows:
            passed += ok
            print(f"{'PASS' if ok else 'FAIL'}  {name:22} {secs:4.1f}s  heard: {heard[:48]:48}  -> {str(got)[:60]}  [{status}]" + ("" if ok else f"  expected {want}"))
        print()
    total = len(v) + len(t)
    print(f"{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
