#!/usr/bin/env python3
import json, sys
from datetime import datetime, timezone
from pathlib import Path

def load(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def agg(rows):
    ok=[r for r in rows if not r.get("error")]
    edits=sum(r.get("edits") or 0 for r in ok)
    words=sum(r.get("ground_truth_words") or 0 for r in ok)
    wall=sum(r.get("wall_s") or 0 for r in rows)
    dur=sum(r.get("audio_dur_s") or 0 for r in ok)
    return {
        "utterances": len(rows),
        "ok": len(ok),
        "errors": len(rows)-len(ok),
        "ref_words": words,
        "edits": edits,
        "corpus_wer": round(edits/words, 4) if words else None,
        "word_accuracy_pct": round((1-edits/words)*100, 2) if words else None,
        "wall_s": round(wall, 2),
        "audio_s": round(dur, 2),
        "rtf": round(wall/dur, 4) if dur else None,
    }

def main():
    runs=[]
    for path in sys.argv[1:]:
        rows=load(path)
        if not rows:
            continue
        runs.append({"file": path, "backend": rows[0].get("backend"), "subset": rows[0].get("subset"), "model": rows[0].get("model"), **agg(rows)})
    stamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out=Path("/mnt/models/librispeech-20260928/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out/"stt-librispeech-v2.json").write_text(json.dumps({"generated": stamp, "runs": runs}, indent=2))
    def cell(v):
        return "—" if v is None else str(v)
    rows_html="".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            r["backend"], r["subset"], r["ok"], r["errors"], cell(r["corpus_wer"]), cell(r["word_accuracy_pct"]), cell(r["rtf"]), r["wall_s"]
        ) for r in runs)
    html=f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LibriSpeech ASR v2</title>
<style>
body{{margin:0;background:#f6f7f9;color:#15202b;font:16px/1.45 system-ui,sans-serif}}
main{{width:min(860px,calc(100% - 24px));margin:auto;padding:28px 0 48px}}
h1{{font-size:32px;letter-spacing:-.03em;margin:0}}
.sub{{color:#5e6a75}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #d9e0e6;border-radius:12px;overflow:hidden}}
th,td{{padding:10px 12px;border-bottom:1px solid #d9e0e6;text-align:left}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#43505a;background:#f0f3f5}}
.callout{{background:#daf1ec;border-left:5px solid #087e8b;padding:12px 14px;border-radius:8px}}
code{{font:13px ui-monospace,monospace}}
</style></head><body><main>
<p class="sub">ASH local ASR · LibriSpeech official transcripts · {stamp}</p>
<h1>LibriSpeech ASR v2</h1>
<p>Corpus WER is total word edits divided by total reference words. This is not an average of per-utterance rates. Tokenizer matches the lab harness: lowercase letters, digits, and apostrophes.</p>
<div class="callout"><strong>What ran:</strong> Moonshine Streaming Tiny Q8 through audio.cpp, CPU on 127.0.0.1:18197 and GPU1 on 127.0.0.1:18198. No private meeting audio. No OpenAI audio was submitted.</div>
<table style="margin-top:16px"><thead><tr><th>Backend</th><th>Set</th><th>OK</th><th>Errors</th><th>Corpus WER</th><th>Word acc</th><th>RTF</th><th>Wall s</th></tr></thead><tbody>
{rows_html}
</tbody></table>
<p class="sub">A transport error is not scored as a word error. Utterances under about 15 seconds, so the CPU server's 120-second ggml abort does not apply. GPU1 shares its server lock with Ali TTS; requests are serialized, not concurrent.</p>
</main></body></html>
"""
    (out/"stt-librispeech-v2.html").write_text(html)
    print(out/"stt-librispeech-v2.html")
    for r in runs:
        print(r["backend"], r["subset"], "wer", r["corpus_wer"], "rtf", r["rtf"], "err", r["errors"])

if __name__=="__main__":
    main()
