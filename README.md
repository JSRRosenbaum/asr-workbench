# asr-workbench

ASH-local speech recognition bench. The Rowan `inference-lab-benchmarks` harness was the starting checkpoint for the result shape. The runs, audio, and scored report are on ASH.

## What this scores

Moonshine Streaming Tiny Q8 (`private-moonshine-asr`) on the official LibriSpeech test sets:

| Backend | Set | Corpus WER | Edits / words | RTF |
|---|---|---:|---:|---:|
| CPU `127.0.0.1:18197` | test-clean | 4.84% | 2,546 / 52,576 | 0.0067 |
| GPU1 `127.0.0.1:18198` | test-clean | 4.87% | 2,560 / 52,576 | 0.0071 |
| CPU | test-other | 12.94% | 6,773 / 52,343 | 0.0070 |
| GPU1 | test-other | 13.12% | 6,870 / 52,343 | 0.0076 |

Corpus WER is total word edits divided by total reference words. Tokenizer: lowercase letters, digits, apostrophes. This is not the official SCTK scorer.

The public page is `reports/index.html`, served at asr.jezzick.net. It does not include LibriSpeech audio or any private recording.

## Layout

- `scripts/librispeech_eval.py` — transcribe a LibriSpeech subset and append jsonl
- `scripts/summarize_librispeech.py` — corpus WER summary
- `reports/` — scored jsonl, summary JSON, and the public HTML
- `config/cpu-asr.json` — the CPU audio.cpp server config used for the run

LibriSpeech itself stays on ASH at `/mnt/models/librispeech-20260928` and is not in this repo.
