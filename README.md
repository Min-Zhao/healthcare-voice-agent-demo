# Healthcare Voice Agent Demo

A small prototype voice agent pipeline for spoken healthcare Q&A. It accepts an
audio question, transcribes it with open-source Whisper, sends the transcript
through an LLM reasoning step, and returns a spoken answer through local TTS.
The demo defaults to Whisper for easy local setup and also includes an optional
Mega-ASR backend for noisy, reverberant, or otherwise degraded audio. The goal
is to demonstrate the core building blocks of a voice healthcare assistant in a
minimal, inspectable Python project.

```text
Audio file -> ASR backend -> LLM reasoning step -> pyttsx3 TTS
```

This is a demo pipeline, not medical advice software. The assistant prompt is
intentionally conservative: it avoids diagnosis, encourages clinician follow-up,
and escalates red-flag symptoms.

## Features

- Open-source Whisper ASR wrapper for audio transcription
- Optional Mega-ASR wrapper for robust in-the-wild audio transcription
- LLM reasoning step with Ollama, OpenAI, or OpenAI-compatible APIs
- Local text-to-speech output with pyttsx3
- Small dependency-injected pipeline with fast unit tests

## ASR backend comparison

| ASR option | Best fit | Strengths | Trade-offs |
| --- | --- | --- | --- |
| Whisper | Default local demo backend | Mature, easy to install, strong multilingual baseline, runs locally | Can degrade on severe noise, reverberation, clipping, or far-field audio |
| Mega-ASR | Robust audio backend for messy real-world recordings | Built on Qwen3-ASR-1.7B with an audio-quality router and robustness tuning for noisy/degraded audio | Newer project, heavier setup, requires separate codebase and checkpoints |
| Qwen3-ASR | Open-source multilingual ASR family | 0.6B/1.7B sizes, 52 languages/dialects, streaming/offline paths | Heavier than Whisper; integration depends on Qwen tooling or API |
| Seed-ASR | Research-grade context-aware ASR | LLM-based ASR with strong reported Chinese/English benchmark gains | Mostly research/reference material; less plug-and-play for this demo |
| Gemini audio models | Multimodal audio understanding | Good for audio transcription plus summarization, timestamps, emotion, and richer prompting | Gemini API docs say it is not for real-time transcription; dedicated ASR services are better for live ASR |

For this demo, use Whisper when you want the simplest runnable local path. Use
Mega-ASR when the audio is clinically realistic: far-field, noisy, compressed,
echoey, or recorded through low-quality microphones.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Whisper also needs `ffmpeg` available on your machine.

## Run with Ollama

```bash
ollama pull llama3.1
ollama serve
python3 -m voice_demo.run_voice_demo path/to/question.wav \
  --llm-provider ollama \
  --llm-model llama3.1 \
  --tts-output outputs/answer.wav
```

## Run with Mega-ASR

Mega-ASR is optional and is not included in `requirements.txt`. Install the
official Mega-ASR codebase, download the `zhifeixie/Mega-ASR` checkpoint from
Hugging Face, and place it at `ckpt/Mega-ASR` as described by the Mega-ASR model
card.

```bash
git clone https://github.com/xzf-thu/Mega-ASR.git external/Mega-ASR
cd external/Mega-ASR
python3 -m pip install -r requirements.txt
cd ../..

# Download/copy the checkpoint so this folder exists:
# ckpt/Mega-ASR/Qwen3-ASR-1.7B
# ckpt/Mega-ASR/audio_quality_router/best_acc_model.pt

python3 -m voice_demo.run_voice_demo path/to/question.wav \
  --asr-provider mega-asr \
  --mega-asr-code-path external/Mega-ASR \
  --mega-asr-ckpt-dir ckpt/Mega-ASR \
  --llm-provider ollama \
  --llm-model llama3.1 \
  --tts-output outputs/answer.wav
```

## Run with OpenAI

```bash
export OPENAI_API_KEY=...
python3 -m voice_demo.run_voice_demo path/to/question.wav \
  --asr-provider whisper \
  --llm-provider openai \
  --llm-model gpt-4.1-mini \
  --tts-output outputs/answer.wav
```

## Test

```bash
python3 -m unittest discover -s tests
python3 -m py_compile voice_demo/*.py
```

## Resume bullet

> Built a prototype voice agent pipeline integrating Whisper (ASR), an LLM
> reasoning step, and TTS for spoken healthcare Q&A.
