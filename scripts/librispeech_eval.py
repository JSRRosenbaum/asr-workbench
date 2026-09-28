#!/usr/bin/env python3
"""Evaluate a local OpenAI-compatible ASR endpoint on LibriSpeech test sets."""
import argparse, json, os, re, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

WORD_RE = re.compile(r"[a-z0-9']+")

def words(text):
    return WORD_RE.findall(text.lower().replace("'", "'"))

def levenshtein(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, aw in enumerate(a, 1):
        curr = [i]
        for j, bw in enumerate(b, 1):
            cost = 0 if aw == bw else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]

def load_refs(subset_dir):
    refs = {}
    for path in Path(subset_dir).rglob("*.trans.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt, text = line.split(" ", 1)
            refs[utt] = text.strip()
    return refs

def flac_to_wav(path):
    # A pipe WAV has an unknown RIFF size and a LIST chunk before data.
    # audio.cpp rejects that with "failed to read WAV data chunk".
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1", f"/tmp/librispeech-utt-{os.getpid()}.wav"],
        check=True, capture_output=True,
    )
    return Path(f"/tmp/librispeech-utt-{os.getpid()}.wav").read_bytes()

def transcribe(endpoint, model, wav, timeout):
    boundary = "----librispeech"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="model"\r\n\r\n'
    body += model.encode() + b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="language"\r\n\r\nen\r\n'
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="file"; filename="utt.wav"\r\n'
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += wav + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("text", ""), time.perf_counter() - start, None, payload.get("timing")
    except Exception as exc:
        return "", time.perf_counter() - start, str(exc), None

def wav_duration(wav):
    # PCM16 mono 16kHz WAV: 44-byte header is not always exact, use ffprobe-free estimate
    if len(wav) <= 44:
        return 0.0
    return max(0.0, (len(wav) - 44) / (16000 * 2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset", required=True, choices=["test-clean", "test-other"])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    subset_dir = Path(args.root) / args.subset
    refs = load_refs(subset_dir)
    files = sorted(subset_dir.rglob("*.flac"))
    if args.limit:
        files = files[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    print(f"subset={args.subset} files={len(files)} already={len(done)}", file=sys.stderr, flush=True)
    with out.open("a") as fh:
        for idx, flac in enumerate(files, 1):
            utt = flac.stem
            if utt in done:
                continue
            ref = refs.get(utt)
            if ref is None:
                rec = {"id": utt, "error": "missing_reference"}
            else:
                try:
                    wav = flac_to_wav(flac)
                    hyp, wall, err, timing = transcribe(args.endpoint, args.model, wav, args.timeout)
                except Exception as exc:
                    hyp, wall, err, timing, wav = "", 0.0, str(exc), None, b""
                ref_w = words(ref)
                hyp_w = words(hyp)
                edits = levenshtein(ref_w, hyp_w) if not err else None
                dur = wav_duration(wav) if wav else 0.0
                rec = {
                    "id": utt,
                    "subset": args.subset,
                    "backend": args.backend,
                    "model": args.model,
                    "audio_path": str(flac),
                    "reference": ref,
                    "transcript": hyp,
                    "ground_truth_words": len(ref_w),
                    "edits": edits,
                    "wer": (edits / len(ref_w)) if edits is not None and ref_w else None,
                    "wall_s": round(wall, 4),
                    "audio_dur_s": round(dur, 4),
                    "rtf": round(wall / dur, 6) if dur else None,
                    "server_timing": timing,
                    "error": err,
                }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if idx % 25 == 0 or idx == len(files):
                print(f"{args.backend} {args.subset} {idx}/{len(files)} {utt}", file=sys.stderr, flush=True)
    print("done", out, file=sys.stderr)

if __name__ == "__main__":
    main()
