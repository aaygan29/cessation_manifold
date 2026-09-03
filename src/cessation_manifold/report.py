"""Writes results/report.html summarizing the four kill-criteria gates."""
from __future__ import annotations

import json
from pathlib import Path


def render_report(results: dict, out_path: str = "results/report.html") -> None:
    gate1 = results.get("gate1", {})
    gate2 = results.get("gate2", {})
    gate3 = results.get("gate3", {})
    gate4 = results.get("gate4", {})

    def status(passed):
        if passed is True:
            return '<span style="color:green;font-weight:bold">PASS</span>'
        if passed is False:
            return '<span style="color:red;font-weight:bold">FAIL</span>'
        return '<span style="color:orange;font-weight:bold">PARTIAL / N-A</span>'

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>cessation_manifold report</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2em auto;}}
table{{border-collapse:collapse;width:100%;}} td,th{{border:1px solid #ccc;padding:6px 10px;text-align:left;}}
pre{{background:#f5f5f5;padding:1em;overflow-x:auto;}}</style></head>
<body>
<h1>cessation_manifold: kill-criteria gate report</h1>
<p>Apparatus validated on synthetic data. See README for what is and is not
wired to real EEG in this version.</p>
<table>
<tr><th>Gate</th><th>Description</th><th>Status</th></tr>
<tr><td>1</td><td>Within-subject reproducibility (synthetic dense-sampling)</td><td>{status(gate1.get('gate1_pass'))}</td></tr>
<tr><td>2</td><td>Non-meditator controls further from manifold than meditator baseline</td><td>{status(gate2.get('gate2_pass'))}</td></tr>
<tr><td>3</td><td>IAAFT surrogate EEG breaks the score</td><td>{status(gate3.get('gate3_pass'))}</td></tr>
<tr><td>4</td><td>Split-conformal coverage holds nominal rate</td><td>{status(gate4.get('gate4_pass'))}</td></tr>
</table>
<h2>Raw results</h2>
<pre>{json.dumps(results, indent=2, default=str)}</pre>
</body></html>
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
