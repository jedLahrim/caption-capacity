---
title: Tools
sdk: gradio
app_file: app.py
license: mit
short_description: Local Whisper video captioning tool
---

# Captacity Caption Studio

Local single-page Gradio app for uploading a video, generating animated word-by-word Captacity captions with local Whisper, and previewing or downloading the rendered output video.

## Prerequisites

- Python 3.10 or newer is recommended.
- ffmpeg must be installed separately as a system dependency.

Install ffmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows with Chocolatey
choco install ffmpeg
```

On Windows, you can also download ffmpeg manually from https://ffmpeg.org/download.html and add it to your PATH.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser when Gradio starts.

The first local Whisper run may download and cache a model depending on your `openai-whisper` cache state. No API key is required; the app calls Captacity with `use_local_whisper=True`.

The rendered output keeps the original input audio by muxing the Captacity-rendered video stream with the uploaded video's audio stream using ffmpeg.

Use `Top Padding` and `Bottom Padding` to move captions vertically from the default centered position. Increasing `Top Padding` pushes captions down by that many pixels; increasing `Bottom Padding` pushes captions up by that many pixels.

To fix caption text, click `Transcribe for Editing`, edit words in the transcript box, then click `Generate Captions`. If you only replace words and keep the same word count, the original word timings are preserved. If you add or remove words, the app redistributes word timings across the original transcript duration.

Use `Font Family` in Caption Style to choose between `Bangers`, `Knewave`, `Poetsen One`, `Urbanist`, and `Coplette`.

Use `Caption Type` to switch between the standard color-highlight style and the pink rounded-box highlight style shown in the preview.
