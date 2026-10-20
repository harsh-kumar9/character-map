"""Smoke test of the three persona-vector stages on two traits, 2 samples per question, output tag _smoke.

    python -m cm.pv.smoke
"""
import subprocess
import sys

TRAITS = "evil,proportionate_accountability"
for stage, extra in [("gen", ["--n_per_question", "2"]), ("judge", []), ("extract", [])]:
    cmd = [sys.executable, "-u", "-m", f"cm.pv.{stage}", "--traits", TRAITS, "--tag", "_smoke", *extra]
    print("[smoke]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
print("[smoke] done", flush=True)
