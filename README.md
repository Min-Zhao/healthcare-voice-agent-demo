# Healthcare Voice Agent Demo

A small prototype voice agent pipeline for spoken healthcare Q&A. It accepts an
audio question, transcribes it with open-source Whisper, sends the transcript
through an LLM reasoning step, and returns a spoken answer through local TTS.
The goal is to demonstrate the core building blocks of a voice healthcare
assistant in a minimal, inspectable Python project.

```text
Audio file -> Whisper ASR -> LLM reasoning step -> pyttsx3 TTS
```

This is a demo pipeline, not medical advice software. The assistant prompt is
intentionally conservative: it avoids diagnosis, encourages clinician follow-up,
and escalates red-flag symptoms.

## Features

- Open-source Whisper ASR wrapper for audio transcription
- LLM reasoning step with Ollama, OpenAI, or OpenAI-compatible APIs
- Local text-to-speech output with pyttsx3
- Small dependency-injected pipeline with fast unit tests

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

## Run with OpenAI

```bash
export OPENAI_API_KEY=...
python3 -m voice_demo.run_voice_demo path/to/question.wav \
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
