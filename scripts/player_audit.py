#!/usr/bin/env python3
"""
Player-perspective routing audit.

Runs the REAL sell-scan path (``app._sell_scan_inner(frame_bgr=...)``, the code the Sell page and
auto-scan use) on every labelled eval screenshot and every auto-scan debug bundle under a realistic
progress state, then writes

  <out>/<frame>_NN.png     contact sheets: crop | identified icon | name | route and why
  <out>/audit_report.md    per-frame counts, every item's route, automatic red flags

so a human can judge each route the way a Tarkov player would (is the icon what the name says, is a
still-needed task item kept with the right count / FiR requirement, is everything else sent to the
best trader or the flea at a sane price).

Progress state
--------------
Quest completions come from EFT's own logs (``push-notifications*.log`` carry the quest chat
messages: type 12 + ``<questId> successMessageText`` = completed, 11 = failed, 10 = started), counted
only after the last prestige (``/client/prestige/obtain`` in ``*backend*.log``).  The hideout and the
"have" counts cannot be read from the logs: nothing is marked built (the app's default), so
hideout-driven KEEP rows are reported as such.  Failed quests count as closed (they can never be
handed in).  Pass ``--progress FILE`` to use a progress.json instead of the logs.

Everything runs against a throw-away copy of the settings / keep list / progress / learned icon
names (``--out``/sandbox); data/*.json is never written.  Prices are read from the cache file as is
(no network).

  python scripts/player_audit.py                       # all eval + debug frames
  python scripts/player_audit.py --frames data/debug/scan-1790985227648-sell/frame.png
  python scripts/player_audit.py --all-tasks           # ignore the kappa_only_tasks setting
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DATA = os.path.join(ROOT, 'data')
DEFAULT_LOGS = r'C:\Battlestate Games\Escape from Tarkov\Logs'
CURRENCY = {'roubles', 'dollars', 'euros'}


# ---------------------------------------------------------------------------
# progress from the game logs
# ---------------------------------------------------------------------------

_QUEST_MSG = re.compile(r'"type":\s*(\d+),\s*"dt":\s*(\d+),\s*"text":\s*"[^"]*",\s*'
                        r'"templateId":\s*"([0-9a-f]{24}) (\w+)"')
STARTED, FAILED, COMPLETED = 10, 11, 12


def _log_dirs(logs_dir):
    return sorted(d for d in os.listdir(logs_dir) if d.startswith('log_'))


def find_last_prestige(logs_dir):
    """Epoch seconds of the newest ``/client/prestige/obtain`` call in the backend logs (None if
    the account never prestiged), scanning newest session first."""
    pat = re.compile(rb'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+\|[^\n]*client/prestige/obtain')
    for d in reversed(_log_dirs(logs_dir)):
        best = None
        for f in glob.glob(os.path.join(logs_dir, d, '*backend*.log')):
            with open(f, 'rb') as fh:
                data = fh.read()
            if b'prestige/obtain' not in data:
                continue
            for m in pat.finditer(data):
                best = m.group(1).decode()
        if best:
            return time.mktime(time.strptime(best, '%Y-%m-%d %H:%M:%S'))
    return None


def quest_states(logs_dir, since_epoch):
    """{quest id: (state, epoch)} - the newest quest chat message per quest after ``since_epoch``."""
    ev = {}
    floor = time.strftime('log_%Y.%m.%d', time.localtime((since_epoch or 0) - 86400))
    for d in _log_dirs(logs_dir):
        if d < floor:
            continue
        for f in glob.glob(os.path.join(logs_dir, d, '*push-notifications*.log')):
            with open(f, encoding='utf-8', errors='replace') as fh:
                txt = fh.read()
            for m in _QUEST_MSG.finditer(txt):
                t, dt, qid, suffix = int(m[1]), int(m[2]), m[3], m[4]
                if t in (STARTED, FAILED, COMPLETED) and suffix in ('description', 'failMessageText', 'successMessageText'):
                    if since_epoch and dt < since_epoch:
                        continue
                    if qid not in ev or dt >= ev[qid][1]:
                        ev[qid] = (t, dt)
    return ev


def progress_from_logs(logs_dir, since=None):
    if since is None:
        since = find_last_prestige(logs_dir)
    states = quest_states(logs_dir, since)
    done = sorted(q for q, (t, _) in states.items() if t == COMPLETED)
    failed = sorted(q for q, (t, _) in states.items() if t == FAILED)
    started = sorted(q for q, (t, _) in states.items() if t == STARTED)
    return {'completed_tasks': sorted(set(done) | set(failed)), 'completed_hideout': [], 'have': {}}, {
        'since': since, 'completed': len(done), 'failed': len(failed), 'started': len(started)}


# ---------------------------------------------------------------------------
# sandbox + the real scan
# ---------------------------------------------------------------------------

def make_sandbox(progress, all_tasks, no_dino, keep_list_path=None):
    """Throw-away data files for one audit; returns (dir, settings dict)."""
    box = tempfile.mkdtemp(prefix='tsh_audit_')
    with open(os.path.join(DATA, 'settings.json'), encoding='utf-8') as fh:
        settings = json.load(fh)
    learned = os.path.join(box, 'learned_icons.json')
    src = os.path.join(DATA, 'learned_icons.json')
    if os.path.exists(src):
        shutil.copy(src, learned)
    settings.update({'debug_dumps': False, 'auto_scan': False})
    if all_tasks:
        settings['kappa_only_tasks'] = False
    ident = dict(settings.get('identify_v2') or {})
    ident['extra'] = {'learned_path': learned}
    if no_dino:
        ident['use_dino'] = False
    settings['identify_v2'] = ident
    with open(os.path.join(box, 'settings.json'), 'w', encoding='utf-8') as fh:
        json.dump(settings, fh, indent=1)
    with open(os.path.join(box, 'progress.json'), 'w', encoding='utf-8') as fh:
        json.dump(progress, fh)
    kl = keep_list_path or os.path.join(DATA, 'keep_list.json')
    shutil.copy(kl, os.path.join(box, 'keep_list.json'))
    return box, settings


def load_app(box):
    import app as A
    A.SETTINGS_PATH = os.path.join(box, 'settings.json')
    A.PROGRESS_PATH = os.path.join(box, 'progress.json')
    A.KEEPLIST_PATH = os.path.join(box, 'keep_list.json')

    def offline_prices():
        with open(A.PRICES_PATH, encoding='utf-8') as fh:
            return json.load(fh)
    A.get_prices = offline_prices
    return A


def run_real_scan(A, frame_bgr):
    """(json payload of the sell scan, the raw detections it was built from)."""
    captured = {}
    orig = A.scan_with_v2

    def spy(img, settings, warnings):
        out = orig(img, settings, warnings)
        captured['raw'] = out[0]
        return out
    A.scan_with_v2 = spy
    try:
        with A.app.app_context():
            payload = A._sell_scan_inner(frame_bgr=frame_bgr).get_json()
    finally:
        A.scan_with_v2 = orig
    return payload, captured.get('raw', [])


# ---------------------------------------------------------------------------
# independent expectation of what is still needed ("oracle")
# ---------------------------------------------------------------------------

def open_needs(tasks, progress, kappa_only):
    """{item id: {'need', 'fir', 'src': [...]}} and the open any-of objectives, restated from the
    raw task data without app.compute_tasks_view, to cross-check the plan."""
    done = set(progress.get('completed_tasks', []))
    levels = set(progress.get('completed_hideout', []))
    need, any_of = {}, []
    for t in tasks.get('tasks', []):
        if t['id'] in done or (kappa_only and not t.get('kappaRequired')):
            continue
        for o in t.get('objectives') or []:
            if o.get('type') not in ('giveItem', 'plantItem') or not o.get('count'):
                continue
            alts = [a for a in (o.get('items') or ([o['item']] if o.get('item') else []))
                    if (a.get('name') or '').lower() not in CURRENCY]
            if not alts:
                continue
            label = f"{(t.get('trader') or {}).get('name', '?')} - {t['name']}"
            if len(alts) > 1:
                any_of.append({'label': label, 'count': o['count'], 'fir': bool(o.get('foundInRaid')),
                               'ids': {a['id'] for a in alts}})
                continue
            rec = need.setdefault(alts[0]['id'], {'need': 0, 'fir': 0, 'src': []})
            rec['need'] += o['count']
            rec['fir'] += o['count'] if o.get('foundInRaid') else 0
            rec['src'].append(label)
    for st in tasks.get('hideoutStations', []):
        for lv in st.get('levels') or []:
            if lv['id'] in levels:
                continue
            for req in lv.get('itemRequirements') or []:
                it = req.get('item') or {}
                if not it.get('id') or (it.get('name') or '').lower() in CURRENCY or not req.get('count'):
                    continue
                rec = need.setdefault(it['id'], {'need': 0, 'fir': 0, 'src': []})
                rec['need'] += req['count']
                rec['src'].append(f"hideout {st['name']} L{lv['level']}")
    return need, any_of


# ---------------------------------------------------------------------------
# cards: one per detected footprint
# ---------------------------------------------------------------------------

def _rk(d):
    return (d['px'], d['py'], d['pw'], d['ph'])


def build_cards(A, payload, raw, items, needs, any_of, keep_ids):
    """Merge the scan's sell / keep rows (and the skipped guns / tags) back onto the raw detections."""
    by_rect = {}
    for r in payload.get('results', []):
        by_rect.setdefault((r['px'], r['py'], r['pw'], r['ph']), []).append(r)
    cards = []
    any_ids = set().union(*[g['ids'] for g in any_of]) if any_of else set()
    for d in sorted(raw, key=lambda r: (r['panel'], r['row'], r['col'])):
        item = items.get(d['item_id'])
        rows = by_rect.get(_rk(d), [])
        sells = [r for r in rows if r['recommend'] in ('trader', 'flea')]
        keeps = [r for r in rows if r['recommend'] == 'keep']
        badge = A.skip_badge(item, d.get('category')) if not rows else None
        c = {'det': d, 'item': item, 'name': (item or {}).get('name') or d['name'] or '?',
             'sells': sells, 'keeps': keeps, 'skip': badge, 'flags': [], 'item_id': d['item_id'],
             'count': d.get('count'), 'fir': d.get('fir'), 'uncertain': bool(d.get('uncertain'))}
        if keeps and sells:
            c['route'] = 'PARTKEEP'
        elif keeps:
            c['route'] = 'KEEP'
        elif sells:
            c['route'] = 'QUEUE' if sells[0].get('flea_queue') else sells[0]['recommend'].upper()
        elif badge:
            c['route'] = 'SKIP-' + badge
        else:
            c['route'] = 'NONE'
        _flag(A, c, needs, any_of, any_ids, keep_ids)
        cards.append(c)
    _check_needs(cards, needs, any_ids)
    return cards


def _check_needs(cards, needs, any_ids):
    """NEEDED-BUT-SOLD, judged per item over every scanned copy: the units kept across the whole
    stash must reach min(copies that could be handed in, units still needed)."""
    by_id = {}
    for c in cards:
        if c['item_id'] in needs and c['item_id'] not in any_ids and not c['uncertain']:
            by_id.setdefault(c['item_id'], []).append(c)
    for iid, group in by_id.items():
        need = needs[iid]
        fir_only = need['fir'] >= need['need']
        usable = sum(_stack(c) for c in group if not (fir_only and c['fir'] is False))
        kept = sum(r['count'] for c in group for r in c['keeps'])
        if kept < min(usable, need['need']):
            more = ' +%d' % (len(need['src']) - 1) if len(need['src']) > 1 else ''
            for c in group:
                if c['sells']:
                    c['flags'].append(f"NEEDED-BUT-SOLD: need {need['need']} ({need['src'][0]}{more}), "
                                      f"{usable} usable copies in the stash, kept {kept}")


def _stack(c):
    return c['count'] or 1


def _flag(A, c, needs, any_of, any_ids, keep_ids):
    """Automatic red flags; they are prompts for the human review, not verdicts."""
    f, item, iid = c['flags'], c['item'], c['item_id']
    kept = sum(r['count'] for r in c['keeps'])
    cat = c['det'].get('category')
    types = set((item or {}).get('types') or ())
    if kept and not needs.get(iid) and iid not in keep_ids and iid not in any_ids:
        f.append('KEPT-BUT-NOT-NEEDED: no open task / hideout / keep-list need found')
    if c['sells']:
        s = c['sells'][0]
        if cat == 'container' or 'container' in types:
            f.append('CONTAINER-SOLD: may hold items - empty it first')
        if c['uncertain']:
            f.append(f"UNCERTAIN-SOLD: engine unsure ({c['det'].get('score')}%) but routed to {c['route']}")
        offers = A.sellcalc.trader_offers(item or {}, None) if item else []
        reg = [o for o in offers if o['name'].lower() != 'fence']
        best = (reg or offers or [None])[0]
        if best and s['recommend'] == 'trader' and s['trader_name'] != best['name'] \
                and s.get('trader_price', 0) < best['price']:
            f.append(f"TRADER-NOT-BEST: {s['trader_name']} {s['trader_price']:,} < {best['name']} {best['price']:,}")
        if s['recommend'] == 'flea':
            avg = (item or {}).get('avg24hPrice') or 0
            if avg and s.get('flea_list') and not (0.55 * avg <= s['flea_list'] <= 1.15 * avg):
                f.append(f"FLEA-PRICE: list {s['flea_list']:,} vs 24h avg {avg:,}")
            if c['fir'] is None:
                f.append('FLEA-FIR-UNCLEAR: Found-in-Raid tick could not be read')
        if cat == 'ammo' and c['count'] is None:
            f.append('AMMO-COUNT-UNREAD: ammo stack with no readable count (priced as 1)')
    if c['skip'] == 'GUN' and c['uncertain']:
        f.append('GUN-UNCERTAIN')


# ---------------------------------------------------------------------------
# contact sheets
# ---------------------------------------------------------------------------

def _font(size, bold=False):
    for p in (('C:/Windows/Fonts/consolab.ttf' if bold else 'C:/Windows/Fonts/consola.ttf'),
              'C:/Windows/Fonts/arial.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


ROUTE_COLORS = {'KEEP': (0, 190, 230), 'PARTKEEP': (0, 190, 230), 'TRADER': (230, 190, 40),
                'FLEA': (60, 200, 90), 'QUEUE': (110, 150, 110), 'SKIP-GUN': (150, 150, 150),
                'SKIP-TAG': (150, 150, 150), 'NONE': (230, 80, 80)}
ROUTE_ORDER = ['KEEP', 'PARTKEEP', 'FLEA', 'QUEUE', 'TRADER', 'SKIP-GUN', 'SKIP-TAG', 'NONE']
CARD_W, CARD_H, COLS, ROWS = 560, 122, 2, 14


def _wrap(text, width):
    out, line = [], ''
    for w in str(text).split():
        if len(line) + len(w) + 1 > width and line:
            out.append(line)
            line = w
        else:
            line = (line + ' ' + w).strip()
    if line:
        out.append(line)
    return out


def _fit(img, w, h, up=3.0):
    sc = min(w / img.shape[1], h / img.shape[0], up)
    return cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)


def _icon(item_id):
    p = os.path.join(DATA, 'tmpl_src', f'{item_id}.png')
    if not item_id or not os.path.exists(p):
        return None
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    a = im[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.full(im[:, :, :3].shape, (52, 50, 46), np.float32)
    return (im[:, :, :3] * a + bg * (1 - a)).astype(np.uint8)


def _route_text(c):
    """(headline, detail) for a card."""
    if c['skip']:
        return ('SKIP ' + ('gun' if c['skip'] == 'GUN' else 'dogtag'), 'left out of the sell list (never priced)')
    parts, detail = [], []
    for k in c['keeps']:
        parts.append(f"KEEP x{k['count']}")
        detail.append(k['reason'])
    for s in c['sells']:
        if s['recommend'] == 'flea':
            tag = 'QUEUE' if s.get('flea_queue') else 'FLEA'
            parts.append(f"{tag} x{s['count']} list {s['flea_list']:,} net {s['flea_net']:,}")
            detail.append(f"trader {s['trader_name']} {s['trader_price']:,}; {s['reason']}")
        else:
            parts.append(f"TRADER {s['trader_name']} {s['trader_price']:,} x{s['count']}")
            detail.append(s['reason'] + (f"; flea {s['flea_list']:,}" if s.get('flea_list') else ''))
    return ' | '.join(parts), ' || '.join(d for d in detail if d)


def render_sheets(frame_bgr, cards, out_prefix, title):
    f_name, f_small, f_route = _font(14, True), _font(12), _font(14, True)
    order = sorted(range(len(cards)), key=lambda i: (ROUTE_ORDER.index(cards[i]['route']),
                                                     cards[i]['sells'][0].get('num', 0) if cards[i]['sells'] else 0, i))
    per = COLS * ROWS
    paths, pages = [], {}
    for si in range(0, len(order), per):
        chunk = order[si:si + per]
        sheet = Image.new('RGB', (CARD_W * COLS, CARD_H * ((len(chunk) + COLS - 1) // COLS) + 24), (22, 22, 26))
        dr = ImageDraw.Draw(sheet)
        pageno = si // per + 1
        dr.text((6, 4), f'{title}  page {pageno}', fill=(200, 200, 200), font=f_small)
        for k, ci in enumerate(chunk):
            c = cards[ci]
            pages[ci] = pageno
            ox, oy = (k % COLS) * CARD_W, (k // COLS) * CARD_H + 24
            d = c['det']
            x, y, w, h = d['px'], d['py'], d['pw'], d['ph']
            crop = frame_bgr[max(0, y):y + h, max(0, x):x + w]
            col = ROUTE_COLORS[c['route']]
            dr.rectangle([ox + 1, oy + 1, ox + CARD_W - 2, oy + CARD_H - 2], outline=(60, 60, 66))
            dr.rectangle([ox + 1, oy + 1, ox + 5, oy + CARD_H - 2], fill=col)
            dr.text((ox + 9, oy + 4), f'#{ci}', fill=(255, 220, 90), font=f_small)
            if crop.size:
                cr = _fit(crop, 118, 92)
                sheet.paste(Image.fromarray(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)), (ox + 9, oy + 20))
            icon = _icon(c['item_id'])
            if icon is not None:
                ic = _fit(icon, 74, 74, up=1.5)
                sheet.paste(Image.fromarray(cv2.cvtColor(ic, cv2.COLOR_BGR2RGB)), (ox + 132, oy + 20))
            tx = ox + 214
            flags = []
            if c['uncertain']:
                flags.append('UNSURE')
            if c['fir'] is True:
                flags.append('FiR')
            elif c['fir'] is None:
                flags.append('fir?')
            nm = c['name'] + (f" x{c['count']}" if c['count'] and c['count'] > 1 else '')
            for li, t in enumerate(_wrap(nm, 40)[:2]):
                dr.text((tx, oy + 4 + li * 16), t, fill=(240, 240, 240), font=f_name)
            head, detail = _route_text(c)
            dr.text((tx, oy + 38), head[:44], fill=col, font=f_route)
            dr.text((tx, oy + 54), f"{d.get('score')}% {' '.join(flags)}  {d['W']}x{d['H']}", fill=(170, 170, 170), font=f_small)
            for li, t in enumerate(_wrap(detail, 46)[:3]):
                dr.text((tx, oy + 68 + li * 13), t, fill=(190, 200, 210), font=f_small)
            if c['flags']:
                dr.text((ox + 9, oy + CARD_H - 16), '! ' + c['flags'][0][:70], fill=(255, 110, 110), font=f_small)
        path = f'{out_prefix}_{pageno:02d}.png'
        sheet.save(path)
        paths.append(path)
    return paths, pages


# ---------------------------------------------------------------------------
# truth comparison (labelled eval frames)
# ---------------------------------------------------------------------------

def truth_mismatches(frame_path, cards):
    """[(card index, truth name)] where the detection disagrees with the labelled truth."""
    tp = os.path.splitext(frame_path)[0] + '.truth.full.json'
    if not os.path.exists(tp):
        return None
    import test_scan as T
    with open(tp, encoding='utf-8') as fh:
        truth = json.load(fh)
    mt = T.match_rects([t['rect'] for t in truth],
                       [[c['det']['px'], c['det']['py'], c['det']['pw'], c['det']['ph']] for c in cards])
    out, n, ok = [], 0, 0
    for ti, ci in mt.items():
        t = truth[ti]
        if t.get('uncertain') or not t.get('item_id'):
            continue
        c = cards[ci]
        n += 1
        same = c['item_id'] == t['item_id'] or c['name'] == t.get('name')
        if t.get('category') == 'weapon':
            same = c['det'].get('category') == 'weapon'
        elif (t.get('name') or '').startswith('Dogtag ') and 'case' not in t['name']:
            same = c['name'].startswith('Dogtag ') and 'case' not in c['name']
        if same:
            ok += 1
        else:
            out.append((ci, t.get('name')))
    return {'n': n, 'ok': ok, 'wrong': out, 'truth_rows': len(truth), 'matched': len(mt)}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def md_escape(s):
    return str(s).replace('|', '\\|')


def frame_report(name, shape, cards, sheets, pages, tm, took):
    cnt = {}
    for c in cards:
        cnt[c['route']] = cnt.get(c['route'], 0) + 1
    unsure = sum(1 for c in cards if c['uncertain'])
    lines = [f'## {name}', '',
             f'{shape[1]}x{shape[0]} px, {len(cards)} footprints identified in {took:.1f}s, {unsure} flagged uncertain.', '',
             'Routes: ' + ', '.join(f'{k} {cnt[k]}' for k in ROUTE_ORDER if k in cnt) + '.']
    if tm:
        lines.append(f"Identification vs labelled truth: {tm['ok']}/{tm['n']} = {100 * tm['ok'] / max(1, tm['n']):.1f}% "
                     f"(segmentation {tm['matched']}/{tm['truth_rows']}).")
        for ci, tname in tm['wrong']:
            lines.append(f"- WRONG #{ci}: truth '{tname}', scan says '{cards[ci]['name']}'"
                         f"{' (flagged uncertain)' if cards[ci]['uncertain'] else ''}, routed {cards[ci]['route']}")
    lines.append('')
    lines.append('Contact sheets: ' + ', '.join(os.path.basename(p) for p in sheets))
    flagged = [(i, c) for i, c in enumerate(cards) if c['flags']]
    if flagged:
        lines += ['', '### Automatic red flags', '', '| # | page | item | route | flag |', '|---|---|---|---|---|']
        for i, c in flagged:
            for fl in c['flags']:
                lines.append(f"| {i} | {pages.get(i)} | {md_escape(c['name'])} | {c['route']} | {md_escape(fl)} |")
    lines += ['', '### Every item', '', '| # | page | item | conf | FiR | n | route | detail |', '|---|---|---|---|---|---|---|---|']
    for i, c in enumerate(cards):
        head, detail = _route_text(c)
        lines.append(f"| {i} | {pages.get(i)} | {md_escape(c['name'])}{' ?' if c['uncertain'] else ''} | {c['det'].get('score')} "
                     f"| {c['fir']} | {c['count'] or ''} | {md_escape(head)} | {md_escape(detail)[:200]} |")
    lines.append('')
    return '\n'.join(lines)


def collect_frames(args):
    if args.frames:
        return [p for g in args.frames for p in (sorted(glob.glob(g)) if any(ch in g for ch in '*?[') else [g])]
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, 'eval', '**', '*.png'), recursive=True)):
        if '.sheet_' not in p and os.path.exists(os.path.splitext(p)[0] + '.truth.full.json'):
            out.append(p)
    out += sorted(glob.glob(os.path.join(DATA, 'debug', 'scan-*', 'frame.png')))
    return out


def frame_name(p):
    p = os.path.abspath(p)
    if os.path.basename(p) == 'frame.png':
        return os.path.basename(os.path.dirname(p))
    return os.path.splitext(os.path.basename(p))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', nargs='*', help='images / globs (default: every labelled eval shot + debug bundle)')
    ap.add_argument('--out', default=os.path.join(DATA, 'audit'), help='output directory')
    ap.add_argument('--logs', default=DEFAULT_LOGS, help='EFT Logs directory')
    ap.add_argument('--progress', help='use this progress.json instead of reading the game logs')
    ap.add_argument('--since', help="count quests after this local time 'YYYY-MM-DD HH:MM:SS' (default: last prestige)")
    ap.add_argument('--all-tasks', action='store_true', help='count non-Kappa tasks too (kappa_only_tasks off)')
    ap.add_argument('--no-dino', action='store_true')
    ap.add_argument('--keep-list', help='keep_list.json to use (default data/keep_list.json)')
    a = ap.parse_args()

    if a.progress:
        with open(a.progress, encoding='utf-8') as fh:
            progress, pinfo = json.load(fh), {'source': a.progress}
    elif os.path.isdir(a.logs):
        since = time.mktime(time.strptime(a.since, '%Y-%m-%d %H:%M:%S')) if a.since else None
        progress, pinfo = progress_from_logs(a.logs, since)
        pinfo['source'] = 'EFT logs'
    else:
        progress, pinfo = {'completed_tasks': [], 'completed_hideout': [], 'have': {}}, {'source': 'none (no logs found)'}
    os.makedirs(a.out, exist_ok=True)

    box, settings = make_sandbox(progress, a.all_tasks, a.no_dino, a.keep_list)
    A = load_app(box)
    with open(A.PRICES_PATH, encoding='utf-8') as fh:
        prices = json.load(fh)
    items = {it['id']: it for it in prices['items']}
    tasks = A.get_tasks(allow_fetch=False)
    kappa_only = settings.get('kappa_only_tasks', True)
    needs, any_of = open_needs(tasks, progress, kappa_only)
    keep_list = json.load(open(os.path.join(box, 'keep_list.json'), encoding='utf-8'))
    mapped, _ = A.map_keep_entries_to_ids(keep_list, A.build_price_index(prices))
    keep_ids = {tid for tid, e in mapped.items() if not e.get('acquired')}

    report = ['# Player routing audit', '',
              f"Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/player_audit.py.", '',
              f"Progress: {pinfo}; {len(progress['completed_tasks'])} tasks closed, hideout none marked built "
              f"(not readable from the logs).  kappa_only_tasks = {kappa_only}; flea needs FiR = "
              f"{(prices.get('rules') or {}).get('flea', {}).get('foundInRaidRequired')} (tarkov.dev).",
              f"Open item needs per the task data: {len(needs)} items + {len(any_of)} any-of objectives.", '']
    totals = {'n': 0, 'ok': 0}
    summary = []
    for path in collect_frames(a):
        img = cv2.imread(path)
        if img is None:
            print('cannot read', path)
            continue
        name = frame_name(path)
        print(f'--- {name} ({img.shape[1]}x{img.shape[0]})', flush=True)
        t0 = time.perf_counter()
        payload, raw = run_real_scan(A, img)
        took = time.perf_counter() - t0
        if payload.get('error'):
            print('  scan error:', payload['error'])
            report += [f'## {name}', '', f"scan error: {payload['error']}", '']
            continue
        cards = build_cards(A, payload, raw, items, needs, any_of, keep_ids)
        sheets, pages = render_sheets(img, cards, os.path.join(a.out, name), name)
        tm = truth_mismatches(path, cards)
        if tm:
            totals['n'] += tm['n']
            totals['ok'] += tm['ok']
        report.append(frame_report(name, img.shape, cards, sheets, pages, tm, took))
        summary.append((name, len(cards), tm))
        with open(os.path.join(a.out, name + '.cards.json'), 'w', encoding='utf-8') as fh:
            json.dump([{'i': i, 'name': c['name'], 'item_id': c['item_id'], 'route': c['route'], 'uncertain': c['uncertain'],
                        'score': c['det'].get('score'), 'count': c['count'], 'fir': c['fir'], 'rect': _rk(c['det']),
                        'flags': c['flags'], 'detail': _route_text(c)[1]} for i, c in enumerate(cards)],
                      fh, indent=1)
    if totals['n']:
        report.insert(7, f"Identification on labelled eval frames: {totals['ok']}/{totals['n']} = "
                         f"{100 * totals['ok'] / totals['n']:.1f}%.")
    with open(os.path.join(a.out, 'audit_report.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(report))
    print(f"\nreport + sheets in {a.out}")
    shutil.rmtree(box, ignore_errors=True)


if __name__ == '__main__':
    main()
