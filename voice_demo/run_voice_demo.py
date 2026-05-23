#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    HealthcareVoicePipeline,
    MegaASRTranscriber,
    OllamaReasoner,
    OpenAICompatibleReasoner,
    Pyttsx3Speaker,
    WhisperTranscriber,
)


def build_reasoner(args):
    if args.llm_provider == "ollama":
        return OllamaReasoner(model=args.llm_model, base_url=args.base_url)
    if args.llm_provider == "openai":
        return OpenAICompatibleReasoner(model=args.llm_model)
    if args.llm_provider == "openai-compatible":
        return OpenAICompatibleReasoner(
            model=args.llm_model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
    raise ValueError(f"Unsupported LLM provider: {args.llm_provider}")


def build_transcriber(args):
    if args.asr_provider == "whisper":
        return WhisperTranscriber(model_name=args.whisper_model)
    if args.asr_provider == "mega-asr":
        return MegaASRTranscriber(
            ckpt_dir=args.mega_asr_ckpt_dir,
            code_path=args.mega_asr_code_path,
            routing_enabled=args.mega_asr_routing,
        )
    raise ValueError(f"Unsupported ASR provider: {args.asr_provider}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an ASR -> LLM -> pyttsx3 voice healthcare Q&A demo."
    )
    parser.add_argument("audio", type=Path, help="Input audio file for ASR.")
    parser.add_argument(
        "--asr-provider",
        choices=["whisper", "mega-asr"],
        default="whisper",
    )
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument(
        "--mega-asr-ckpt-dir",
        type=Path,
        default=Path("ckpt/Mega-ASR"),
        help="Mega-ASR checkpoint directory containing Qwen3-ASR-1.7B and router files.",
    )
    parser.add_argument(
        "--mega-asr-code-path",
        type=Path,
        help="Optional path to the cloned Mega-ASR codebase if not installed.",
    )
    parser.add_argument(
        "--mega-asr-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Mega-ASR's audio quality router when available.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["ollama", "openai", "openai-compatible"],
        default="ollama",
    )
    parser.add_argument("--llm-model", default="llama3.1")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL or OpenAI-compatible API base URL.",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--tts-output", type=Path, help="Optional path to write TTS audio.")
    parser.add_argument("--tts-rate", type=int)
    args = parser.parse_args()

    pipeline = HealthcareVoicePipeline(
        transcriber=build_transcriber(args),
        reasoner=build_reasoner(args),
        speaker=Pyttsx3Speaker(rate=args.tts_rate),
    )
    result = pipeline.run(args.audio, args.tts_output)

    print("Transcript:")
    print(result.transcript)
    print()
    print("Answer:")
    print(result.answer)
    if result.spoken_output:
        print()
        print(f"TTS audio saved to {result.spoken_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
