"""The honesty layer: any number this pipeline reports must be gated.

Discipline (matches the neuroai-honesty-layer pattern used across this
program's other instruments): a claim is a `Finding`, produced only via
`gate()`, which raises `UnvalidatedClaimError` instead of returning a number
when the coverage/off-manifold checks fail. There is no code path that lets
a raw, unchecked score reach the caller labeled as a validated result.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class UnvalidatedClaimError(Exception):
    """Raised when a readout cannot be reported as a validated number."""


@dataclass
class ProvenanceStamp:
    dataset_id: str
    git_commit: str
    config_hash: str
    timestamp_utc: str

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class Finding:
    value: float
    lower: float
    upper: float
    coverage_target: float
    coverage_achieved: float
    provenance: ProvenanceStamp
    status: str = "VALIDATED"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "value": self.value,
            "interval": [self.lower, self.upper],
            "coverage_target": self.coverage_target,
            "coverage_achieved": self.coverage_achieved,
            "status": self.status,
            "provenance": self.provenance.to_dict(),
        }
        d.update(self.extra)
        return d


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def make_provenance(dataset_id: str, config: dict) -> ProvenanceStamp:
    return ProvenanceStamp(
        dataset_id=dataset_id,
        git_commit=_git_commit(),
        config_hash=config_hash(config),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def gate(
    value: float,
    lower: float,
    upper: float,
    coverage_target: float,
    coverage_achieved: float,
    dataset_id: str,
    config: dict,
    coverage_tolerance: float = 0.05,
    off_manifold: bool = False,
    extra: dict | None = None,
) -> Finding:
    """Return a validated Finding, or raise UnvalidatedClaimError.

    Abstains when:
      - achieved conformal coverage is more than `coverage_tolerance` below
        the nominal target (the interval cannot be trusted at the stated
        confidence level), or
      - the input epoch was flagged off-manifold by the caller (e.g. its
        features fall outside the range the conformal calibration set
        covered), or
      - the interval is non-finite.
    """
    provenance = make_provenance(dataset_id, config)

    reasons = []
    if coverage_achieved < coverage_target - coverage_tolerance:
        reasons.append(
            f"conformal coverage {coverage_achieved:.3f} below target "
            f"{coverage_target:.3f} - tolerance {coverage_tolerance:.3f}"
        )
    if off_manifold:
        reasons.append("input flagged off-manifold relative to calibration set")
    if not all(map(_is_finite, (value, lower, upper))):
        reasons.append("non-finite value or interval bound")

    if reasons:
        raise UnvalidatedClaimError(
            "UNVALIDATED: " + "; ".join(reasons) + f" (provenance: {provenance.to_dict()})"
        )

    return Finding(
        value=value,
        lower=lower,
        upper=upper,
        coverage_target=coverage_target,
        coverage_achieved=coverage_achieved,
        provenance=provenance,
        status="VALIDATED",
        extra=extra or {},
    )


def _is_finite(x: Any) -> bool:
    try:
        return x == x and abs(x) != float("inf")
    except Exception:
        return False
