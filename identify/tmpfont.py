"""
The game's own text renderer, reproduced: TextMeshPro signed-distance-field fonts.

EFT draws the stash labels with TextMeshPro.  A TMP font asset is a 1024x1024 SDF atlas
(``Texture2D``, Alpha8) plus per-glyph metrics (glyph rect in the atlas, bearings, advance) and
face info (sampling point size, ascender ...), and the material decides how the distance field
becomes pixels (face dilate, outline, underlay, gradient scale).  The game is an IL2CPP build, so
the MonoBehaviour has no type tree; :func:`parse_font_asset` reads the TMP 1.1.0 font-asset layout
directly (validated by the glyph / character table counts agreeing, and by every glyph rect
lying inside the atlas).

:func:`extract` pulls all of it out of ``resources.assets`` with UnityPy, once, into
``data/fonts/`` (atlas PNG + JSON).  :class:`TMPRenderer` then renders a string the way the TMP
"Distance Field" shader does (``GetColor`` + the underlay pass), at any font size and sub-pixel
position, producing premultiplied colour + alpha that is composited over the observed background.
"""
from __future__ import annotations

import json
import os
import struct

import cv2
import numpy as np

GLYPH_REC = 52          # uint index, 5 float metrics, 4 int rect, float scale, int atlas index, int class


class _Rd:
    def __init__(self, b: bytes, p: int = 0):
        self.b, self.p = b, p

    def i(self):
        v = struct.unpack_from('<i', self.b, self.p)[0]
        self.p += 4
        return v

    def f(self):
        v = struct.unpack_from('<f', self.b, self.p)[0]
        self.p += 4
        return v

    def q(self):
        v = struct.unpack_from('<q', self.b, self.p)[0]
        self.p += 8
        return v

    def s(self):
        n = self.i()
        if not 0 <= n < 4096:
            raise ValueError('bad string length')
        v = self.b[self.p:self.p + n].decode('utf-8', 'replace')
        self.p = (self.p + n + 3) & ~3
        return v


FACE_FIELDS = ['lineHeight', 'ascent', 'cap', 'mean', 'baseline', 'descent', 'supOff', 'supSize', 'subOff',
               'subSize', 'ulOff', 'ulThick', 'stOff', 'stThick', 'tab']


def parse_font_asset(raw: bytes) -> dict:
    """TMP_FontAsset (font asset version 1.1.0) from a MonoBehaviour's raw bytes."""
    k = raw.find(b'1.1.0')
    if k < 4:
        raise ValueError('not a TMP 1.1.0 font asset')
    r = _Rd(raw, k - 4)
    r.s()                                      # m_Version
    r.s()                                      # m_SourceFontFileGUID
    r.i(), r.q()                               # m_SourceFontFile
    r.i()                                      # m_AtlasPopulationMode
    face = {'index': r.i(), 'family': r.s(), 'style': r.s(), 'pointSize': r.i(), 'scale': r.f(), 'upem': r.i()}
    for n in FACE_FIELDS:
        face[n] = r.f()
    n = r.i()
    if not 0 < n < 20000:
        raise ValueError('bad glyph count')
    glyphs = {}
    for _ in range(n):
        g = struct.unpack_from('<I5f4ifii', raw, r.p)
        r.p += GLYPH_REC
        glyphs[g[0]] = {'w': g[1], 'h': g[2], 'bx': g[3], 'by': g[4], 'adv': g[5],
                        'rx': g[6], 'ry': g[7], 'rw': g[8], 'rh': g[9], 'scale': g[10]}
    nc = r.i()
    if not 0 < nc < 20000:
        raise ValueError('bad character count')
    chars = {}
    for _ in range(nc):
        et, uni, gi, sc = struct.unpack_from('<iIIf', raw, r.p)
        r.p += 16
        if gi in glyphs:
            chars[uni] = gi
    # tail: normalStyle, normalSpacingOffset, boldStyle, boldSpacing (floats), italicStyle, tabSize (bytes)
    ns, nso, bs, bsp = struct.unpack_from('<4f', raw, len(raw) - 24)
    for g in glyphs.values():
        if g['rx'] < 0 or g['ry'] < 0 or g['rx'] + g['rw'] > 4096 or g['ry'] + g['rh'] > 4096:
            raise ValueError('glyph rect outside any atlas')
    return {'face': face, 'glyphs': {str(k): v for k, v in glyphs.items()}, 'chars': {str(k): v for k, v in chars.items()},
            'normalStyle': ns, 'normalSpacingOffset': nso, 'boldStyle': bs, 'boldSpacing': bsp}


def extract(data_dir: str, out_dir: str, log=print) -> list[str]:
    """Every Bender TMP font asset (+ its atlas and material) and the raw Bender fonts of the
    installed game -> ``out_dir``.  Returns the written font-asset names."""
    import UnityPy
    env = UnityPy.load(os.path.join(data_dir, 'resources.assets'))
    os.makedirs(out_dir, exist_ok=True)
    mats, atlases, assets = {}, {}, []
    for obj in env.objects:
        t = obj.type.name
        if t not in ('Font', 'Material', 'Texture2D', 'MonoBehaviour'):
            continue
        try:
            name = obj.peek_name() or ''
        except Exception:
            name = ''
        if 'bender' not in name.lower() or "DON'T USE" in name:
            continue
        if t == 'Font':
            d = obj.read()
            data = bytes(getattr(d, 'm_FontData', b'') or b'')
            if len(data) > 1000:
                style = name.split(' - ')[-1].strip().replace(' ', '_')
                _write(os.path.join(out_dir, style + '.otf'), data)
        elif t == 'Material':
            d = obj.read()
            sp = d.m_SavedProperties
            mats[name.replace(' Material', '')] = {
                'floats': {k: float(v) for k, v in sp.m_Floats},
                'colors': {k: [float(v.r), float(v.g), float(v.b), float(v.a)] for k, v in sp.m_Colors},
                'keywords': list(getattr(d, 'm_ValidKeywords', None) or getattr(d, 'm_ShaderKeywords', None) or []),
                'atlas': int(dict(sp.m_TexEnvs).get('_MainTex').m_Texture.path_id) if '_MainTex' in dict(sp.m_TexEnvs) else 0}
        elif t == 'Texture2D' and name.endswith('Atlas'):
            atlases.setdefault(name.replace(' Atlas', ''), []).append(obj)
        elif t == 'MonoBehaviour' and name.endswith('SDF'):
            assets.append((name, obj))
    written = []
    for name, obj in assets:
        base = name.replace(' SDF', '')                       # "Jovanny Lemonad - Bender Outline"
        try:
            fa = parse_font_asset(obj.get_raw_data())
        except Exception as e:
            log(f'[font] {name}: {e}')
            continue
        mat = mats.get(base)
        tex_objs = atlases.get(base) or []
        if mat and mat.get('atlas'):
            tex_objs = sorted(tex_objs, key=lambda o: o.path_id != mat['atlas'])
        if not tex_objs:
            continue
        img = tex_objs[0].read().image                        # PIL, Unity's bottom-up rows flipped
        a = np.asarray(img.convert('RGBA'))[:, :, 3]
        short = base.split(' - ')[-1].replace(' ', '_')       # Bender_Outline
        cv2.imwrite(os.path.join(out_dir, short + '_atlas.png'), a)
        fa['material'] = mat or {}
        fa['atlas_size'] = [int(a.shape[1]), int(a.shape[0])]
        _write(os.path.join(out_dir, short + '_sdf.json'), json.dumps(fa).encode('utf-8'))
        written.append(short)
    if written:
        log(f'[font] extracted TMP font assets {written} from the game into {out_dir}')
    return written


def _write(path: str, data: bytes) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

class TMPFont:
    def __init__(self, json_path: str, atlas_path: str):
        with open(json_path, encoding='utf-8') as fh:
            d = json.load(fh)
        self.face = d['face']
        self.glyphs = {int(k): v for k, v in d['glyphs'].items()}
        self.chars = {int(k): v for k, v in d['chars'].items()}
        self.normal_spacing = float(d.get('normalSpacingOffset', 0.0))
        self.bold_spacing = float(d.get('boldSpacing', 7.0))
        self.mat = d.get('material') or {}
        a = cv2.imread(atlas_path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise OSError(atlas_path)
        # the PNG is top-down; glyph rects count rows from the bottom (Unity texture space)
        self.atlas = a.astype(np.float32) / 255.0
        self.atlas_h = self.atlas.shape[0]
        self.point = float(self.face['pointSize'])
        self.gscale = float(self.mat.get('floats', {}).get('_GradientScale', 11.0))
        self.padding = self.gscale - 1.0

    def glyph(self, ch: str):
        gi = self.chars.get(ord(ch))
        return None if gi is None else self.glyphs.get(gi)

    def advance(self, text: str, size: float, extra_spacing: float = 0.0) -> float:
        """Pen advance of ``text`` in screen px at font size ``size`` (TMP: glyph advance x
        size/pointSize + (normalSpacingOffset + spacing) x size/100 per character)."""
        s = size / self.point
        em = size * 0.01
        x = 0.0
        for ch in text:
            g = self.glyph(ch)
            adv = g['adv'] if g else self.face['tab']
            x += adv * s + (self.normal_spacing + extra_spacing) * em
        return x


def _f(m, k, d=0.0):
    return float(m.get('floats', {}).get(k, d))


def _c(m, k, d=(0.0, 0.0, 0.0, 1.0)):
    return tuple(m.get('colors', {}).get(k, d))


class TMPRenderer:
    """Renders a string exactly as TMP's Distance Field shader would, into a strip.

    ``render`` returns premultiplied BGR float32 [h, w, 3] and alpha [h, w]; the screen is
    ``bg (1 - a) + rgb``.  Coordinates: column/row 0 = the strip's top-left pixel; ``pen_x`` is the
    pen position of the first character, ``base_y`` the baseline (both in px, fractional)."""

    def __init__(self, font: TMPFont, sharp_scale: float = 1.0, shade: dict | None = None):
        self.font = font
        self.k = sharp_scale             # calibration of the shader's screen-space scale (1 = as derived)
        # measured shading of the label component: face / outline edges as signed distances in
        # atlas texels (ef, eu; negative = dilated), anti-aliasing ramps in screen px (wf, wu) and
        # the outline opacity (au).  None = the material's own values through TMP's shader maths.
        self.shade = shade

    def render(self, text: str, w: int, h: int, size: float, pen_x: float, base_y: float,
               color=(1.0, 1.0, 1.0), bold: bool = False, extra_spacing: float = 0.0):
        F = self.font
        m = F.mat
        s = size / F.point                       # screen px per atlas texel
        em = size * 0.01
        P = F.padding
        sra, src = _f(m, '_ScaleRatioA', 1.0), _f(m, '_ScaleRatioC', 1.0)
        scale = s * F.gscale * (_f(m, '_Sharpness') + 1.0) * self.k
        weight = ((_f(m, '_WeightBold') if bold else _f(m, '_WeightNormal')) / 4.0 + _f(m, '_FaceDilate')) * sra * 0.5
        bias = (0.5 - weight) + 0.5 / scale
        outline = _f(m, '_OutlineWidth') * sra * scale
        soft = _f(m, '_OutlineSoftness') * sra * scale
        under = 'UNDERLAY_ON' in (m.get('keywords') or [])
        if under:
            bscale = scale / (1.0 + _f(m, '_UnderlaySoftness') * src * scale)
            bbias = (0.5 - weight) * bscale - 0.5 - (_f(m, '_UnderlayDilate') * src * 0.5 * bscale)
            uox = -(_f(m, '_UnderlayOffsetX') * src) * F.gscale        # in atlas texels
            uoy = -(_f(m, '_UnderlayOffsetY') * src) * F.gscale
            ucol = _c(m, '_UnderlayColor')
        fcol = _c(m, '_FaceColor', (1, 1, 1, 1))
        ocol = _c(m, '_OutlineColor')
        # premultiplied colours, BGR order (material colours are RGBA)
        face = np.array([fcol[2] * color[0], fcol[1] * color[1], fcol[0] * color[2]], np.float32) * fcol[3]
        oc = np.array([ocol[2], ocol[1], ocol[0]], np.float32) * ocol[3]
        rgb = np.zeros((h, w, 3), np.float32)
        al = np.zeros((h, w), np.float32)
        x = pen_x
        A = F.atlas
        H_at = F.atlas_h
        for ch in text:
            g = F.glyph(ch)
            if g is None:
                x += F.face['tab'] * s + (F.normal_spacing + extra_spacing) * em
                continue
            if g['rw'] > 0 and g['rh'] > 0:
                left = x + (g['bx'] - P) * s
                top = base_y - (g['by'] + P) * s
                qw, qh = (g['rw'] + 2 * P) * s, (g['rh'] + 2 * P) * s
                x0, y0 = max(0, int(np.floor(left))), max(0, int(np.floor(top)))
                x1, y1 = min(w, int(np.ceil(left + qw))), min(h, int(np.ceil(top + qh)))
                if x1 > x0 and y1 > y0:
                    X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32) + 0.5,
                                       np.arange(y0, y1, dtype=np.float32) + 0.5)
                    inside = (X >= left) & (X < left + qw) & (Y >= top) & (Y < top + qh)
                    ty = H_at - g['ry'] - g['rh']               # glyph rect top row in the PNG
                    ax = (g["rx"] - P + (X - left) / s - 0.5).astype(np.float32)     # texel-centre convention
                    ay = (ty - P + (Y - top) / s - 0.5).astype(np.float32)
                    c = cv2.remap(A, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    if self.shade is not None:
                        sh = self.shade
                        dt = (c - 0.5) * (2.0 * P)                # signed distance, atlas texels
                        fa = np.clip((dt - sh['ef']) * s / sh['wf'] + 0.5, 0, 1)
                        ua = sh['au'] * np.clip((dt - sh['eu']) * s / sh['wu'] + 0.5, 0, 1) * (1 - fa)
                        a_g = (fa + ua) * inside
                        c_g = (fa[..., None] * face) * inside[..., None]
                        sl = (slice(y0, y1), slice(x0, x1))
                        rgb[sl] = c_g + rgb[sl] * (1 - a_g)[..., None]
                        al[sl] = a_g + al[sl] * (1 - a_g)
                        x += g['adv'] * s + (F.normal_spacing + extra_spacing) * em
                        continue
                    sd = (bias - c) * scale
                    fa = 1.0 - np.clip((sd - outline * 0.5 + soft * 0.5) / (1.0 + soft), 0, 1)
                    oa = np.clip(sd + outline * 0.5, 0, 1) * np.sqrt(min(1.0, outline))
                    # lerp(face, outline, oa) * fa  (premultiplied, face alpha 1, outline alpha ocol[3])
                    a_g = ((1 - oa) * fcol[3] + oa * ocol[3]) * fa
                    c_g = ((1 - oa)[..., None] * face + oa[..., None] * oc) * fa[..., None]
                    if under:
                        cu = c if (uox == 0 and uoy == 0) else cv2.remap(
                            A, ax + uox, ay - uoy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        ua = np.clip(cu * bscale - bbias, 0, 1) * (1 - a_g)
                        c_g = c_g + (np.array([ucol[2], ucol[1], ucol[0]], np.float32) * ucol[3])[None, None] * ua[..., None]
                        a_g = a_g + ucol[3] * ua
                    a_g = a_g * inside
                    c_g = c_g * inside[..., None]
                    # this quad over what is already drawn (premultiplied "One OneMinusSrcAlpha")
                    sl = (slice(y0, y1), slice(x0, x1))
                    rgb[sl] = c_g + rgb[sl] * (1 - a_g)[..., None]
                    al[sl] = a_g + al[sl] * (1 - a_g)
            x += g['adv'] * s + (F.normal_spacing + extra_spacing) * em
        return rgb, al


QUANT = 8               # sub-pixel positions of cached glyph sprites (1/8 px)


class SpriteRenderer(TMPRenderer):
    """Same output as :meth:`TMPRenderer.render` with a measured ``shade``, composed from cached
    per-glyph sprites at 1/8 px phases (an order of magnitude faster for many candidates)."""

    def __init__(self, font: TMPFont, shade: dict):
        super().__init__(font, 1.0, shade)
        self._spr: dict = {}

    def _sprite(self, ch: str, size: float, fx: float, fy: float):
        key = (ch, round(size, 3), fx, fy)
        sp = self._spr.get(key)
        if sp is not None:
            return sp
        F, sh = self.font, self.shade
        g = F.glyph(ch)
        s = size / F.point
        P = F.padding
        qw, qh = (g['rw'] + 2 * P) * s, (g['rh'] + 2 * P) * s
        W, H = int(np.ceil(fx + qw)) + 1, int(np.ceil(fy + qh)) + 1
        X, Y = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)
        inside = (X >= fx) & (X < fx + qw) & (Y >= fy) & (Y < fy + qh)
        ty = F.atlas_h - g['ry'] - g['rh']
        ax = (g['rx'] - P + (X - fx) / s - 0.5).astype(np.float32)
        ay = (ty - P + (Y - fy) / s - 0.5).astype(np.float32)
        c = cv2.remap(F.atlas, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        dt = (c - 0.5) * (2.0 * P)
        fa = np.clip((dt - sh['ef']) * s / sh['wf'] + 0.5, 0, 1) * inside
        ua = sh['au'] * np.clip((dt - sh['eu']) * s / sh['wu'] + 0.5, 0, 1) * (1 - fa) * inside
        sp = (fa.astype(np.float32), (fa + ua).astype(np.float32))
        if len(self._spr) > 50000:
            self._spr.clear()
        self._spr[key] = sp
        return sp

    def render(self, text: str, w: int, h: int, size: float, pen_x: float, base_y: float,
               color=(1.0, 1.0, 1.0), bold: bool = False, extra_spacing: float = 0.0):
        F = self.font
        s = size / F.point
        em = size * 0.01
        P = F.padding
        face = np.array(color, np.float32)
        fl = np.zeros((h, w), np.float32)        # face weight (premultiplied)
        al = np.zeros((h, w), np.float32)
        x = pen_x
        for ch in text:
            g = F.glyph(ch)
            if g is None:
                x += F.face['tab'] * s + (F.normal_spacing + extra_spacing) * em
                continue
            if g['rw'] > 0 and g['rh'] > 0:
                left = x + (g['bx'] - P) * s
                top = base_y - (g['by'] + P) * s
                il, it = int(np.floor(left)), int(np.floor(top))
                fx = round((left - il) * QUANT) / QUANT
                fy = round((top - it) * QUANT) / QUANT
                fa, a = self._sprite(ch, size, fx, fy)
                sh_, sw_ = fa.shape
                x0, y0 = max(0, il), max(0, it)
                x1, y1 = min(w, il + sw_), min(h, it + sh_)
                if x1 > x0 and y1 > y0:
                    fs = fa[y0 - it:y1 - it, x0 - il:x1 - il]
                    as_ = a[y0 - it:y1 - it, x0 - il:x1 - il]
                    sl = (slice(y0, y1), slice(x0, x1))
                    fl[sl] = fs + fl[sl] * (1 - as_)
                    al[sl] = as_ + al[sl] * (1 - as_)
            x += g['adv'] * s + (F.normal_spacing + extra_spacing) * em
        return fl[..., None] * face[None, None], al
