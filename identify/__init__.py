"""identify - v2 item-identification core for the Tarkov Stash Helper.

Pipeline (see :func:`identify.pipeline.scan`):

1. ``grid``     - line-structure based panel/grid detection (pitch, origin, all rows/cols)
2. ``segment``  - item footprints from the drawn borders (multi-cell, rotated, stacks)
3. ``catalog``  - template catalog of *every* item (tarkov.dev base images + EFT icon cache)
4. ``match``    - stage 1 masked pixel MAD, stage 2 DINOv2 re-rank, stage 3 OCR authority
5. ``pipeline`` - glue + calibrated confidence + ``Detection`` records
"""
from .config import EngineSettings  # noqa: F401
