# Sarkari Form Bharo

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
