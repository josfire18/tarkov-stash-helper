"""Synthetic full-screen frames for the autoscan tests (nothing personal, no game assets)."""
import cv2
import numpy as np

from synth import render_panel

ITEMS = [(0, 0, 2, 2), (3, 0, 1, 3), (4, 1, 3, 1), (0, 3, 1, 1), (2, 3, 1, 1), (5, 3, 2, 2)]


def menu_bar(img):
    """The lobby's bottom bar: black hairlines at its top and bottom edge, lit UI between."""
    h, w = img.shape[:2]
    bar = max(8, int(round(0.0287 * h)))
    img[h - bar - 3:h - bar] = 0
    img[h - bar:h - 3] = (28, 26, 24)
    for x in range(40, w - 40, 160):
        cv2.rectangle(img, (x, h - bar + 4), (x + 90, h - 7), (150, 150, 150), -1)
    img[h - 3:] = 0
    return img


def inventory_frame(w=2560, h=1440, pitch=None, chrome=True, cols=10, rows=8, origin=None, seed=3):
    pitch = pitch or 63.0 * h / 1080.0
    img = np.full((h, w, 3), (26, 28, 28), np.uint8)
    panel, _, _, _ = render_panel(pitch, cols, rows, ITEMS, origin=(10, 10), chrome=False)
    ox, oy = origin or (w // 3, int(0.1 * h))
    ph, pw = panel.shape[:2]
    pw, ph = min(pw, w - ox), min(ph, h - oy - 1)
    img[oy:oy + ph, ox:ox + pw] = panel[:ph, :pw]
    return menu_bar(img) if chrome else img


def raid_frame(w=2560, h=1440, seed=1, dark=False):
    """Textured outdoor-ish scene: gradients, noise, grey boxes, thin lines, crosshair."""
    rng = np.random.default_rng(seed)
    sky = np.linspace((200, 170, 120), (60, 70, 60), h)[:, None, :].repeat(w, 1).astype(np.uint8)
    img = np.clip(sky.astype(np.int16) + rng.integers(-3, 4, (h, w, 3)), 0, 255).astype(np.uint8)
    for _ in range(60):                                           # grey boxes in the border colour range
        x, y = int(rng.integers(0, w - 200)), int(rng.integers(0, h - 200))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(20, 400)), y + int(rng.integers(20, 300))),
                      (84 + int(rng.integers(-6, 6)), 81, 73), -1 if rng.random() < .5 else 1)
    for _ in range(25):                                           # long thin lines at random places
        x = int(rng.integers(0, w))
        cv2.line(img, (x, 0), (x + int(rng.integers(-30, 30)), h), (90, 85, 76), 1)
    cv2.line(img, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (20, 255, 20), 2)
    if dark:
        img = (img * 0.05).astype(np.uint8)
    return img
