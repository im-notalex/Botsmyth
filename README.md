# Botsmyth

![Botsmyth Hero](assets/hero.svg)

[![Python](https://img.shields.io/badge/Python-3.10%2B-0f766e?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-111827?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Requests](https://img.shields.io/badge/Requests-HTTP%20Client-334155?logo=python&logoColor=white)](https://requests.readthedocs.io/)
[![Pillow](https://img.shields.io/badge/Pillow-Image%20Processing-1f2937?logo=python&logoColor=white)](https://python-pillow.org/)
[![Local First](https://img.shields.io/badge/Local--First-Data%20stays%20on%20your%20machine-0ea5e9?logo=lock&logoColor=white)](#data-and-storage)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-64748b?logo=windows&logoColor=white)](#quick-start)
[![UI](https://img.shields.io/badge/UI-Single%20File%20App-10b981?logo=html5&logoColor=white)](#project-layout)
[![Vision](https://img.shields.io/badge/Vision-Image--Aware%20Generation-14b8a6?logo=opencollective&logoColor=white)](#image-workflow)
[![Exports](https://img.shields.io/badge/Exports-Card%20v2%20%7C%20Janitor%20%7C%20Risu%20%7C%20PNG-6366f1?logo=files&logoColor=white)](#exports)
[![License](https://img.shields.io/badge/License-MIT-22c55e?logo=opensourceinitiative&logoColor=white)](LICENSE)

Botsmyth is a local-first RP bot builder with a fast Simple flow and full Advanced control. Create polished character packs (Description, First Messages, Scenario, Example Dialogues), test vision, and export to popular card formats.

Main QoL start: `quickstart.bat` auto-installs requirements and launches the app on Windows.

![App Preview](assets/preview.svg)

## Table of contents
- About
- Features
- Quick start
- Configuration
- Usage
- Image workflow
- Output length targets
- Exports
- Project layout
- Data and storage
- Troubleshooting
- FAQ
- License

## About
Botsmyth keeps your data on your machine while giving you a modern, image-aware workflow. It is a single-file Flask app with embedded HTML/CSS/JS for quick edits and easy sharing.

## Features
- Simple and Advanced modes with a guided onboarding tour
- Compile workflow that generates the full 4-section pack in one pass
- Image uploads with vision-aware generation and an in-app image viewer
- Library cards with stacked image previews for quick visual scanning
- Token target presets per section for predictable output length
- Local autosave and version history
- Test chat with Assistant or In-Character modes
- Exports to Card v2, Janitor, Risu, Prompt TXT, and PNG
- One-click `quickstart.bat` launcher for Windows
- Model-agnostic provider support with OpenAI-compatible endpoints

## Quick start
### Requirements
- Python 3.10+
- Dependencies: Flask, Requests, Pillow (installed via `requirements.txt`)

### Quickstart.bat (main QoL)
1) Double-click `quickstart.bat`
2) The script installs requirements if needed and launches the app

### Manual
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python botmaker.py
```

Open `http://localhost:8000` in your browser.

## Configuration
Set these in the **AI Settings** panel:
- Provider, model, base URL, API key
- Max tokens and temperature
- Use images (vision)

Environment variables:
- `PORT` (default: `8000`)

For custom endpoints, pick **OpenAI Compatible** and set your base URL and model.

## Usage
### Simple flow
1) Add Original Input (and optional name/age/species)
2) Generate Simple to get a starter kit
3) Compile for a clean export-ready pack
4) Apply Outputs to push results into the main fields

### Advanced flow
1) Enable Advanced mode
2) Generate All or per-section outputs
3) Tune any field, lists, and toggles
4) Compile and Export when ready

## Image workflow
- Upload up to the configured max images
- Click any image to open the Image Viewer
- Library cards show a stacked carousel preview of bot images

## Output length targets
In Compile, set minimum token targets for:
- Description
- First Messages (total)
- Scenario
- Example Dialogues (total)

Pick Auto, Low, Medium, High, Very High, or Extreme based on how long you want each section.

## Exports
- Card v2 (Chub/Tavern)
- Janitor JSON
- Risu JSON
- Prompt TXT
- PNG with optional embedded card data

## Project layout
- `botmaker.py` - single-file app (backend + UI)
- `quickstart.bat` - Windows installer/launcher
- `requirements.txt` - Python dependencies
- `data/` - local storage for bots, images, and exports
- `assets/` - README visuals

## Data and storage
All data is stored locally:
- `data/bots.json` for bot profiles and lists
- `data/images/` for uploaded images
- `data/exports/` for exported files
- `data/settings.json` for app settings

Autosave history is also stored in your browser local storage.

## Troubleshooting
**Outputs are too short**  
Increase Max Tokens, enable High Quality, and set Output Length targets.

**Compile returns empty output**  
Check your API key, model, and max tokens. Try a smaller model or lower temperature.

**Images do not appear**  
Ensure the image upload completed and that you are under the max image limit.

## FAQ
**Where is the UI code?**  
The UI (HTML/CSS/JS) lives inside `botmaker.py`.

**Can I use my own model endpoint?**  
Yes. Use **OpenAI Compatible** and set your base URL and model.

**How do I improve quality?**  
Increase Max Tokens, enable High Quality, and set Output Length targets.

**What does Botsmyth store and where?**  
Bots, images, exports, and settings live in the local `data/` folder.

## License
MIT License. See `LICENSE`.

## Why AGPL?
This project uses AGPL-3.0 to ensure that if anyone hosts a modified 
version as a public service, users have access to that modified source 
code. Since these tools handle personal creative content, transparency 
is important. Run it locally and modify freely - the license only 
requires sharing if you make it a public web service. Keep it local, or keep it open.
