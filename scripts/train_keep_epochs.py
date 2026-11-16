#!/usr/bin/env python3
"""Train a model under the score-aware objective, keeping every epoch.

``scripts/train.py`` writes one file, ``last.pt``, and rewrites it at the end
of every epoch.  A run that is still training therefore has no stable
checkpoint to point at: the file under a fixed name changes underneath the
reader.  This wrapper keeps a copy per epoch so the score can be read off a
known epoch instead of off a moment.

It patches ``torch.save`` rather than the training loop, so the training
itself is exactly the code the matrix ran.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as train_module  # noqa: E402

_original_save = torch.save
_epoch = {"n": 0}


def _patched_save(obj, path, *args, **kwargs):
    _original_save(obj, path, *args, **kwargs)
    text = str(path)
    if text.endswith("last.pt"):
        directory = os.path.dirname(text)
        keep = os.path.join(directory, f"epoch_{_epoch['n']:02d}.pt")
        _original_save(obj, keep)
        _epoch["n"] += 1
        print(f"kept {keep}", flush=True)


torch.save = _patched_save

if __name__ == "__main__":
    raise SystemExit(train_module.main())
