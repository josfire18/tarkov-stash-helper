"""Hands-free stash scanning: find the game, capture passively, detect an inventory, scan once.

See README "Auto-scan".  Public surface: :class:`AutoScanner` and :func:`make_blueprint`.
"""
from .service import AutoScanner, make_blueprint  # noqa: F401
