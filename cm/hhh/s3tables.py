"""S3 tables for the HHH line, written to results/hhh/s3 so the honesty tables do not move (ruling of 2026-10-24).

The global tables are rebuilt from every concept's results/s3/concepts file: summary.json carries BH q-values over all
cells (a new concept changes them), frames_table.json the within and transfer AUROCs, claims.json the D-033 classes,
which need those q-values. This runs S3's own code for each (cm.s3 table, cm.s3_report, cm.s3_claims), unchanged,
against a view directory whose concepts and control folders are links to the real ones, then copies the three
tables to results/hhh/s3. Same criteria, same code; only where the output lands differs.

    python -m cm.hhh.s3tables          CPU, seconds
"""
from __future__ import annotations

import shutil

from .. import cfg, s3, s3_claims, s3_report
from .blocks import out_dir

VIEW = cfg.outputs_dir() / "hhh" / "s3view"


def link(name, target):
    p = VIEW / name
    if p.is_symlink() or p.exists():
        p.unlink()
    p.symlink_to(target, target_is_directory=True)


def main():
    real = cfg.results_dir()
    (VIEW / "s3").mkdir(parents=True, exist_ok=True)
    link("s3/concepts", real / "s3" / "concepts")
    for ctrl in ("s3_ctrl_verdict", "s3_ctrl_negation"):
        if (real / ctrl).exists():
            link(ctrl, real / ctrl)
    s3.OUT = VIEW / "s3"
    s3.cmd_table()
    s3_report.S3 = VIEW / "s3"
    s3_report.main()
    s3_claims.RESULTS = VIEW
    s3_claims.main()
    dst = out_dir() / "s3"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("summary.json", "frames_table.json", "claims.json"):
        shutil.copy(VIEW / "s3" / f, dst / f)
    print(f"[hhh s3tables] wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
