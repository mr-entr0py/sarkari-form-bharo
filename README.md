# Sarkari Form Bharo

**Live:** https://sarvamaihackathon.vercel.app · *Just speak. We'll handle the paperwork.*

Voice-first government form filling in Hindi and English, built on Sarvam APIs.
Pick a form, listen to each question, answer by speaking, and get a validated, print-ready English form.

**Pipeline:** Saaras (speech to text) → deterministic parsing, with Sarvam-105B for extraction → strict validators
(Aadhaar, mobile, PIN, dates, choices) → Sarvam Transliterate/Translate for English spelling → Bulbul speaks questions and the final read-back.

Names are phonetic transliterations, so every one is flagged "Confirm" until the applicant approves or edits it.
Demo only: not an official government portal; nothing is stored.

## Run locally
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt uvicorn
echo "SARVAM_API_KEY=your_key" > .env
.venv/bin/uvicorn api.index:app --reload
```
Open http://localhost:8000.

## Deploy (Vercel)
Import this repo in Vercel and add the environment variable `SARVAM_API_KEY`. `public/` is served statically and `api/index.py` runs as a Python function.

## Forms and languages
Income Certificate (11 fields), Ration Card (9), PM-Kisan registration (8). Speak in Hindi, English, Marathi,
Bengali, Tamil, Telugu, Gujarati, Kannada or Punjabi; the printed form is always in English.

## End-to-end tests
`tests/e2e_test.py` runs 41 checks against a deployment with the real Sarvam APIs: every field of all three forms
is spoken by Bulbul, transcribed by Saaras and validated, plus rejection cases (bad Aadhaar, mobile, PIN, future or
incomplete dates, out-of-range numbers, invalid choices) and answers in Tamil, Marathi, Bengali, Telugu and English.
```bash
.venv/bin/pip install httpx
.venv/bin/python tests/e2e_test.py https://sarvamaihackathon.vercel.app
```
