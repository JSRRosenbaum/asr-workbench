# Handoff — 2026-09-28

Pause here. Do not start another model sweep unless asked.

## Current

ASH is the work host. This repo is the record. Rowan `inference-lab-benchmarks` was the schema checkpoint only.

One model was scored: Moonshine Streaming Tiny Q8, CPU and GPU1, on official LibriSpeech test-clean and test-other. That is 10.75 hours of unique audio, 5,559 utterances. It is ASR word error only. It is not diarization, and it is not the earlier meeting comparison.

CPU was a few milliseconds per clip ahead of GPU1. Both are about 0.005× realtime. The model is 58 MB and the clips are short, so that gap is overhead, not a platform verdict.

Public report: https://asr.jezzick.net/

## Later

Research other local ASR runtimes against this same LibriSpeech set before naming a winner. The meeting-page candidates were not run here. Speaker diarizers are out of scope for WER. Do not put private meeting audio back on the public site.
