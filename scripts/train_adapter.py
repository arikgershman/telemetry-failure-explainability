#!/usr/bin/env python
"""Check reviewed training exports or explicitly launch CUDA LoRA/DPO."""

from telemetry_reasoning.models.training import main

if __name__ == "__main__":
    raise SystemExit(main())
