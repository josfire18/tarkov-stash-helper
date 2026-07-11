"""Generates the app/tray icon in-memory so no binary asset needs to be
hand-authored or committed as an opaque blob. Run this file directly to
also bake out assets/icon.ico for the PyInstaller build.
"""
from PIL import Image, ImageDraw


def _draw(size):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = max(1, size // 16)
    bg = (30, 29, 28, 255)      # matches EFT_BG_TINTS['grey']
    accent = (196, 149, 60, 255)  # amber, Tarkov-ish rarity gold
    d.rounded_rectangle([pad, pad, size - pad, size - pad], radius=size // 6, fill=bg)
    inset = size // 4
    d.rectangle([inset, inset, size - inset, size - inset], outline=accent, width=max(1, size // 16))
    return img


def load_tray_image(size=64):
    return _draw(size)


if __name__ == '__main__':
    import os
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [_draw(s) for s in sizes]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'icon.ico')
    images[0].save(out_path, format='ICO', sizes=[(s, s) for s in sizes], append_images=images[1:])
    print(f"Wrote {out_path}")
