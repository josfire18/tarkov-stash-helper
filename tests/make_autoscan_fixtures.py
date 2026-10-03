"""Regenerates tests/fixtures/autoscan_*.png (deterministic, synthetic, 720p)."""
import os

import cv2

from autoscan_frames import inventory_frame, raid_frame

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(os.path.join(OUT, 'autoscan_inventory_720p.png'), inventory_frame(1280, 720))
    cv2.imwrite(os.path.join(OUT, 'autoscan_raid_720p.png'), raid_frame(1280, 720))
