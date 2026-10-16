"""Task progress read from Escape from Tarkov's own log files (stdlib only).

The game writes one folder per session under ``<install>\\Logs\\`` (``log_<date>_<time>_<version>``).
Everything the app needs to know about quests is in there, because the trader sends a chat
message whenever a quest is started / failed / completed and the client logs every push
notification it receives:

* ``... push-notifications_000.log`` (older builds: ``... notifications.log``) holds the
  notifications as pretty-printed JSON blocks.  A ``new_message`` whose ``message.type`` is
  10 / 11 / 12 is a quest started / failed / completed, and the quest id is the first token of
  ``message.templateId``.  ``message.dt`` is the unix time.  (``message.text`` is unreliable:
  completed quests say "quest started".)
* ``... application_000.log`` says which profile a session plays: ``Session mode: Regular``
  (the normal PvP profile), ``Pve``, or ``PvpSeason`` (the seasonal profile that came with
  1.1).  Every mode is its own profile with its own quest progress.  A session can switch mode
  (the line appears again), and a request's host gives the same information as a fallback
  (``gw-pve*`` / ``gw-pvp-season`` / the rest).
* ``... backend_000.log`` logs every HTTPS request.  ``/client/prestige/obtain`` (a prestige) and
  ``/client/game/profile/create`` (a brand new profile - a wipe, a prestige, or the first PMC
  on a fresh mode) wipe the profile's task progress: everything before the last of them does
  not count.

Reading ~1900 session folders takes tens of seconds, so each folder's extracted events are
cached (``data/eftlogs_cache.json``, keyed by folder name and the sizes of the files read) and
only folders that changed are read again.  The newest folder is always re-read: it is the live
session.

``LogScanner.progress()`` turns the events of one mode into ``completed`` / ``failed`` /
``active`` quest-id lists.  Given the task data it also fills holes (a completed task's
prerequisites are completed too) and detects the player's faction from the BEAR/USEC-only
tasks they started.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys
import tempfile
import threading
import time

CACHE_VERSION = 1
LOG_DIR_NAME = 'Logs'
DEFAULT_MODE = 'pvp'
MODES = ('pvp', 'pve', 'season')

_FOLDER_RE = re.compile(r'^log_(\d{4})\.(\d\d)\.(\d\d)_(\d+)-(\d+)-(\d+)_(.+)$')
_QUEST_ID_RE = re.compile(r'^[0-9a-f]{24}$')

# message.type of a trader chat message -> what happened to the quest
QUEST_MESSAGE_KINDS = {10: 'start', 11: 'fail', 12: 'complete'}

# "Session mode: X" values -> our mode names (anything else is lower-cased and kept)
SESSION_MODES = {'regular': 'pvp', 'pvp': 'pvp', 'pve': 'pve', 'pvpseason': 'season', 'season': 'season'}

# A completed quest that "completes" again less than this long after is one notification delivered
# twice, not a reset.
DUPLICATE_WINDOW = 600

# time column at the start of a log line: "2026-06-20 17:06:41.776" (newer builds, local time) or
# "2025-05-31 20:56:10.467 -04:00" (older builds, with the UTC offset)
_TS_RE = re.compile(r'(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)(?:\.\d+)?(?: ([+-]\d\d:\d\d))?\|')
_TS_RE_B = re.compile(rb'(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)(?:\.\d+)?(?: ([+-]\d\d:\d\d))?\|')
_NOTIF_MARK = 'Got notification | '
_MODE_MARK = '|application|Session mode: '
_BACKEND_REQUEST_RE = re.compile(
    rb'---> Request HTTPS, id \[\d+\]: URL: https://(?P<host>[a-z0-9.-]+)\.escapefromtarkov\.com'
    rb'/client/(?P<what>prestige/obtain|game/profile/create)')
_BACKEND_START_RE = re.compile(
    rb'URL: https://(?P<host>[a-z0-9.-]+)\.escapefromtarkov\.com/client/game/start')


# ---------------------------------------------------------------------------
# Install folder
# ---------------------------------------------------------------------------

_REGISTRY_KEYS = (
    (r'SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov', 'HKLM'),
    (r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov', 'HKLM'),
    (r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov', 'HKCU'),
)
_UNINSTALL_ROOTS = (
    (r'SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall', 'HKLM'),
    (r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall', 'HKLM'),
    (r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall', 'HKCU'),
)
_STEAM_KEYS = (
    (r'Software\Valve\Steam', 'HKCU', 'SteamPath'),
    (r'SOFTWARE\WOW6432Node\Valve\Steam', 'HKLM', 'InstallPath'),
    (r'SOFTWARE\Valve\Steam', 'HKLM', 'InstallPath'),
)
_COMMON_SUBPATHS = (
    r'Battlestate Games\Escape from Tarkov',
    r'Games\Battlestate Games\Escape from Tarkov',
    r'Program Files\Battlestate Games\Escape from Tarkov',
    r'Program Files (x86)\Battlestate Games\Escape from Tarkov',
    r'Program Files (x86)\Steam\steamapps\common\Escape from Tarkov',
    r'Program Files\Steam\steamapps\common\Escape from Tarkov',
    r'SteamLibrary\steamapps\common\Escape from Tarkov',
    r'Steam\steamapps\common\Escape from Tarkov',
    r'Games\SteamLibrary\steamapps\common\Escape from Tarkov',
    r'Games\Escape from Tarkov',
    r'EFT',
)
GAME_NAME = 'Escape from Tarkov'
GAME_EXE_NAME = 'EscapeFromTarkov.exe'
DATA_DIR_NAME = 'EscapeFromTarkov_Data'

try:                                    # Windows only; tests replace it with a fake
    import winreg
except ImportError:                     # pragma: no cover
    winreg = None


def _hive(name):
    return winreg.HKEY_LOCAL_MACHINE if name == 'HKLM' else winreg.HKEY_CURRENT_USER


def _reg_values(hive, key, names):
    """{value name: str} of the values that exist in a registry key ({} when it does not)."""
    out = {}
    if winreg is None:
        return out
    try:
        with winreg.OpenKey(_hive(hive), key) as k:
            for n in names:
                try:
                    v, _ = winreg.QueryValueEx(k, n)
                except OSError:
                    continue
                v = str(v).strip().strip('"')
                if v:
                    out[n] = v
    except OSError:
        pass
    return out


def _registry_install_dirs():
    """Battlestate launcher installs: InstallLocation (or the folder of UninstallString) from the
    uninstall registry key."""
    out = []
    for key, hive in _REGISTRY_KEYS:
        v = _reg_values(hive, key, ('InstallLocation', 'UninstallString'))
        if 'InstallLocation' in v:
            out.append(v['InstallLocation'])
        if 'UninstallString' in v:
            out.append(os.path.dirname(v['UninstallString']))
    return out


def _launcher_install_dirs(appdata=None):
    """The BSG launcher records its games folder (``gamesRootDir``) in its own settings file; an
    install moved to another drive is only known there."""
    appdata = appdata if appdata is not None else os.environ.get('APPDATA')
    if not appdata:
        return []
    try:
        with open(os.path.join(appdata, 'Battlestate Games', 'BsgLauncher', 'settings'), encoding='utf-8') as f:
            root = (json.load(f) or {}).get('gamesRootDir')
    except (OSError, ValueError, AttributeError):
        return []
    if not root or not isinstance(root, str):
        return []
    return [os.path.join(root, GAME_NAME), os.path.join(root, 'EFT')]


_VDF_PATH_RE = re.compile(r'"path"\s+"((?:[^"\\]|\\.)*)"', re.I)
_ACF_RE = re.compile(r'"(name|installdir)"\s+"((?:[^"\\]|\\.)*)"', re.I)


def _unvdf(s):
    return s.replace('\\\\', '\\')


def steam_library_dirs(steam_path):
    """Every Steam library folder: Steam's own plus those in ``steamapps/libraryfolders.vdf``."""
    out = [steam_path] if steam_path else []
    try:
        with open(os.path.join(steam_path, 'steamapps', 'libraryfolders.vdf'), encoding='utf-8', errors='replace') as f:
            out += [_unvdf(m) for m in _VDF_PATH_RE.findall(f.read())]
    except OSError:
        pass
    seen, uniq = set(), []
    for d in out:
        k = os.path.normcase(os.path.normpath(d))
        if k not in seen:
            seen.add(k)
            uniq.append(os.path.normpath(d))
    return uniq


def _is_tarkov_folder_name(name):
    n = name.lower()
    return n.startswith(GAME_NAME.lower()) and 'arena' not in n


def _steam_install_dirs():
    """Steam installs: every library's ``steamapps/common/Escape from Tarkov*`` folder, plus the
    folder named by an ``appmanifest_*.acf`` for the game (a renamed install dir), plus the
    Steam app's own Uninstall key."""
    out = []
    steam = None
    for key, hive, val in _STEAM_KEYS:
        v = _reg_values(hive, key, (val,))
        if v.get(val):
            steam = v[val].replace('/', '\\')
            break
    for lib in steam_library_dirs(steam) if steam else []:
        apps = os.path.join(lib, 'steamapps')
        common = os.path.join(apps, 'common')
        try:
            for name in os.listdir(common):
                if _is_tarkov_folder_name(name):
                    out.append(os.path.join(common, name))
        except OSError:
            pass
        try:
            for fn in os.listdir(apps):
                if fn.startswith('appmanifest_') and fn.endswith('.acf'):
                    with open(os.path.join(apps, fn), encoding='utf-8', errors='replace') as f:
                        kv = {k.lower(): _unvdf(v) for k, v in _ACF_RE.findall(f.read())}
                    if kv.get('installdir') and _is_tarkov_folder_name(kv.get('name', '')):
                        out.append(os.path.join(common, kv['installdir']))
        except OSError:
            pass
    for key, hive in _UNINSTALL_ROOTS:                       # "Steam App <id>" entries
        if winreg is None:
            break
        try:
            with winreg.OpenKey(_hive(hive), key) as root:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(root, i)
                    except OSError:
                        break
                    i += 1
                    if not sub.startswith('Steam App '):
                        continue
                    v = _reg_values(hive, key + '\\' + sub, ('DisplayName', 'InstallLocation'))
                    if _is_tarkov_folder_name(v.get('DisplayName', '')) and v.get('InstallLocation'):
                        out.append(v['InstallLocation'])
        except OSError:
            continue
    return out


def _common_install_dirs():
    drives = [f'{c}:\\' for c in 'CDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.isdir(f'{c}:\\')] \
        if sys.platform == 'win32' else []
    return [os.path.join(d, sub) for d in drives for sub in _COMMON_SUBPATHS]


def _has_logs(path):
    return bool(path) and os.path.isdir(os.path.join(path, LOG_DIR_NAME))


def is_install(path):
    """A folder that looks like an EFT install (the game exe or its data folder), Arena excluded."""
    if not path or not os.path.isdir(path):
        return False
    if 'arena' in os.path.basename(os.path.normpath(path)).lower():
        return False
    return (os.path.isfile(os.path.join(path, GAME_EXE_NAME))
            or os.path.isdir(os.path.join(path, DATA_DIR_NAME)))


def _candidates(override=None):
    out = []
    if override:
        o = os.path.normpath(str(override).strip().strip('"'))
        if os.path.basename(o).lower() == LOG_DIR_NAME.lower():
            o = os.path.dirname(o)
        out.append(o)
    for fn in (_registry_install_dirs, _launcher_install_dirs, _steam_install_dirs, _common_install_dirs):
        try:
            out += fn()
        except Exception:                                    # noqa: BLE001 - one broken source must not hide the rest
            continue
    seen, uniq = set(), []
    for i, c in enumerate(out):
        k = os.path.normcase(os.path.normpath(c))
        if override and i == 0:                              # the user's own choice is never second-guessed
            pass
        elif 'arena' in os.path.basename(k):                 # Escape from Tarkov: Arena is another game
            continue
        if k not in seen:
            seen.add(k)
            uniq.append(os.path.normpath(c))
    return uniq


def install_candidates(override=None):
    """Every folder that looks like an EFT install, best first (ones that have logs lead)."""
    c = [d for d in _candidates(override) if is_install(d) or _has_logs(d)]
    return sorted(c, key=lambda d: not _has_logs(d))


def find_install_dir(override=None, require_logs=True):
    """The EFT install folder: the ``eft_install_dir`` setting, then the BSG launcher's registry
    key and settings file, Steam libraries, and the usual places on every drive.  A setting that
    points at the ``Logs`` folder itself is accepted too.  With ``require_logs`` (the log reader)
    the folder must have been played from at least once (``Logs\\`` exists); without it any
    install qualifies (font extraction), preferring one that has logs.  None when nothing is found."""
    cands = _candidates(override)
    for cand in cands:
        if _has_logs(cand):
            return cand
    if not require_logs:
        for cand in cands:
            if is_install(cand):
                return cand
    return None


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def parse_log_time(date_s, time_s, tz_s=None):
    """A log timestamp -> unix time.  With a UTC offset (older builds) it is exact; without
    (newer builds) the machine's local time zone is assumed - the game runs on this machine."""
    dt = datetime.datetime.strptime(f'{date_s} {time_s}', '%Y-%m-%d %H:%M:%S')
    if tz_s:
        sign = -1 if tz_s[0] == '-' else 1
        off = datetime.timedelta(hours=int(tz_s[1:3]), minutes=int(tz_s[4:6])) * sign
        dt = dt.replace(tzinfo=datetime.timezone(off))
    return dt.timestamp()


def folder_start_time(name):
    """The session start encoded in a log folder name, or None for a folder that is not one."""
    m = _FOLDER_RE.match(name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
    try:
        return datetime.datetime(y, mo, d, h, mi, s).timestamp()
    except (ValueError, OverflowError, OSError):
        return None


# ---------------------------------------------------------------------------
# Reading one session folder
# ---------------------------------------------------------------------------

def normalize_mode(raw):
    raw = (raw or '').strip().lower()
    return SESSION_MODES.get(raw, raw or DEFAULT_MODE)


def host_mode(host):
    """The game mode a backend host belongs to (``gw-pve-02`` -> pve, ``gw-pvp-season`` -> season,
    ``gw-pvp`` / ``prod-03`` -> pvp)."""
    host = (host or '').lower()
    if host.startswith('gw-pve'):
        return 'pve'
    if host.startswith('gw-pvp-season') or host.startswith('gw-season'):
        return 'season'
    return 'pvp'


def _read_text(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read()


def _read_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


def _log_files(folder_path):
    """``{'application': [...], 'backend': [...], 'notifications': [...]}`` of (path, size) in a
    session folder.  Both the older (``notifications.log``, ``..._backend.log``) and the newer
    (``push-notifications_000.log``, ``backend_000.log``) names are recognised; the 2 GB
    ``output`` / ``errors`` / ``traces`` logs are never opened."""
    groups = {'application': [], 'backend': [], 'notifications': []}
    try:
        with os.scandir(folder_path) as it:
            for e in it:
                low = e.name.lower()
                if not low.endswith('.log') or not e.is_file():
                    continue
                if 'push-notifications' in low or 'notifications' in low:
                    key = 'notifications'
                elif 'backendcache' in low or 'backend_queue' in low:
                    continue
                elif 'backend' in low:
                    key = 'backend'
                elif 'application' in low:
                    key = 'application'
                else:
                    continue
                try:
                    size = e.stat().st_size
                except OSError:
                    continue
                groups[key].append((e.path, size))
    except OSError:
        pass
    for v in groups.values():
        v.sort()
    return groups


def folder_signature(folder_path):
    """``[[file name, size], ...]`` of the files a folder is read from - what a cached record is
    keyed on."""
    g = _log_files(folder_path)
    return [[os.path.basename(p), s] for key in sorted(g) for p, s in g[key]]


def _mode_at(modes, ts, fallback):
    """The mode in effect at ``ts`` given ``[[time, mode], ...]`` sorted by time."""
    cur = modes[0][1] if modes else fallback
    for t, m in modes:
        if t <= ts:
            cur = m
        else:
            break
    return cur


def _line_time(text, pos, ts_re):
    """Unix time of the log line that contains ``pos``, None if the line has no time column."""
    ls = text.rfind('\n' if isinstance(text, str) else b'\n', 0, pos) + 1
    m = ts_re.match(text, ls)
    if not m:
        return None
    parts = [g.decode() if isinstance(g, bytes) else g for g in m.groups() if g is not None]
    tz = parts[2] if len(parts) > 2 else None
    try:
        return parse_log_time(parts[0], parts[1], tz)
    except ValueError:
        return None


def extract_quest_messages(text):
    """Quest started/failed/completed trader messages in one notifications log:
    ``[(log_time, dt, kind, quest_id), ...]``.  ``log_time`` is when the client logged the
    notification, ``dt`` the message's own unix time.

    The log is plain text: a "Got notification | <Name>" line, then the notification as
    pretty-printed JSON, closed by a ``}`` in column 0 (everything nested is indented)."""
    out = []
    pos = 0
    while True:
        i = text.find(_NOTIF_MARK, pos)
        if i < 0:
            break
        eol = text.find('\n', i)
        if eol < 0:
            break
        start = eol + 1
        pos = start
        if text[start:start + 1] != '{':
            continue
        end = text.find('\n}', start)
        if end < 0:
            break                                       # block still being written
        pos = end + 2
        block = text[start:end + 2]
        if '"new_message"' not in block[:80]:
            continue
        try:
            msg = (json.loads(block) or {}).get('message')
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        kind = QUEST_MESSAGE_KINDS.get(msg.get('type'))
        if not kind:
            continue
        parts = str(msg.get('templateId') or '').split()
        if not parts or not _QUEST_ID_RE.match(parts[0]):
            continue
        log_time = _line_time(text, i, _TS_RE)
        try:
            dt = int(msg.get('dt'))
        except (TypeError, ValueError):
            if log_time is None:
                continue
            dt = int(log_time)
        out.append((log_time if log_time is not None else float(dt), dt, kind, parts[0]))
    return out


def extract_session_modes(text):
    """``[[time, mode], ...]`` from the "Session mode: X" lines of an application log."""
    out = []
    pos = 0
    while True:
        i = text.find(_MODE_MARK, pos)
        if i < 0:
            break
        eol = text.find('\n', i)
        eol = len(text) if eol < 0 else eol
        pos = eol
        word = text[i + len(_MODE_MARK):eol].strip().split()
        t = _line_time(text, i, _TS_RE)
        if word and t is not None:
            out.append([t, normalize_mode(word[0])])
    return out


def extract_resets(data):
    """``[[time, mode, 'prestige'|'create'], ...]`` and the mode of the first ``game/start``
    request, from the bytes of a backend log.  The mode comes from the request's host."""
    first = None
    ms = _BACKEND_START_RE.search(data)
    if ms:
        first = host_mode(ms.group('host').decode('ascii', 'replace'))
    out = []
    if b'prestige/obtain' in data or b'profile/create' in data:
        for m in _BACKEND_REQUEST_RE.finditer(data):
            t = _line_time(data, m.start(), _TS_RE_B)
            if t is None:
                continue
            out.append([t, host_mode(m.group('host').decode('ascii', 'replace')),
                        'prestige' if m.group('what') == b'prestige/obtain' else 'create'])
    return out, first


def read_folder(folder_path, start_ts=None):
    """One session folder -> the cacheable record::

        {'start': ts, 'modes': [[ts, mode], ...], 'host_mode': mode|None,
         'events': [[dt, 'start'|'fail'|'complete', quest_id, mode], ...],
         'resets': [[ts, mode, 'prestige'|'create'], ...]}
    """
    g = _log_files(folder_path)
    modes = []
    for path, _size in g['application']:
        try:
            modes += extract_session_modes(_read_text(path))
        except OSError:
            continue
    modes.sort()

    fallback = None
    resets = []
    for path, _size in g['backend']:
        try:
            r, first = extract_resets(_read_bytes(path))
        except OSError:
            continue
        resets += r
        fallback = fallback or first
    resets.sort()

    events, seen = [], set()
    for path, _size in g['notifications']:
        try:
            msgs = extract_quest_messages(_read_text(path))
        except OSError:
            continue
        for log_time, dt, kind, qid in msgs:
            key = (dt, kind, qid)
            if key in seen:
                continue
            seen.add(key)
            events.append([dt, kind, qid, _mode_at(modes, log_time, fallback or DEFAULT_MODE)])
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return {'start': start_ts, 'modes': modes, 'host_mode': fallback, 'events': events, 'resets': resets}


# ---------------------------------------------------------------------------
# From events to progress
# ---------------------------------------------------------------------------

def _timeline(records, mode, known_ids):
    """Everything that happened to one mode's profile, in order: resets (kind 'reset') and quest
    events.  Events for the same quest, kind and second are one notification (also logged by a
    second session or log file)."""
    tl, seen = [], set()
    for rec in records:
        for ts, m, why in rec.get('resets') or ():
            if m == mode:
                tl.append((ts, 0, 'reset', why))
        for dt, kind, qid, m in rec.get('events') or ():
            if m != mode or (known_ids is not None and qid not in known_ids):
                continue
            key = (dt, kind, qid)
            if key not in seen:
                seen.add(key)
                tl.append((dt, 1, kind, qid))
    tl.sort(key=lambda e: (e[0], e[1], e[2], str(e[3])))
    return tl


def replay(records, mode, known_ids=None):
    """Replay one mode's events into the quest states.  Returns::

        {'states': {quest_id: 'active'|'completed'|'failed'}, 'reset_at': ts|None,
         'reset_kind': 'prestige'|'create'|'heuristic'|None, 'events': n, 'last_event_at': ts|None,
         'event_at': {quest_id: ts of its latest event}}

    A prestige / new profile wipes everything before it.  Beyond that: a completed quest that
    is started - or completed again some time later - can only mean the profile was wiped
    without a trace in the logs we have, so that also resets (everything before is dropped).
    Only quests in ``known_ids`` count for that (repeatable "daily" quests reuse templates).
    A failed quest that is started again is active again."""
    states, event_at = {}, {}
    reset_at = reset_kind = None
    n = 0
    last = None
    for ts, _o, kind, arg in _timeline(records, mode, known_ids):
        if kind == 'reset':
            states, event_at = {}, {}
            reset_at, reset_kind, n, last = ts, arg, 0, None
            continue
        qid = arg
        cur = states.get(qid)
        if kind == 'start':
            if cur == 'completed':                  # cannot be started again without a wipe
                states, event_at, n = {}, {}, 0
                reset_at, reset_kind = ts, 'heuristic'
            states[qid] = 'active'
        elif kind == 'complete':
            if cur == 'completed' and ts - event_at.get(qid, ts) > DUPLICATE_WINDOW:
                states, event_at, n = {}, {}, 0
                reset_at, reset_kind = ts, 'heuristic'
            states[qid] = 'completed'
        elif kind == 'fail':
            if cur != 'completed':
                states[qid] = 'failed'
        event_at[qid] = ts
        n += 1
        last = ts if last is None else max(last, ts)
    return {'states': states, 'reset_at': reset_at, 'reset_kind': reset_kind,
            'events': n, 'last_event_at': last, 'event_at': event_at}


def latest_session_mode(records):
    """``(mode, time)`` of the most recent session that says one, ``(None, None)`` if none does.
    The time is that of its last ``Session mode:`` line (None when only the backend host said)."""
    for rec in sorted(records, key=lambda r: r.get('start') or 0, reverse=True):
        modes = rec.get('modes') or ()
        if modes:
            return modes[-1][1], modes[-1][0]
        if rec.get('host_mode'):
            return rec['host_mode'], None
    return None, None


def latest_mode(records):
    """The mode of the most recent session that says one."""
    return latest_session_mode(records)[0] or DEFAULT_MODE


def _requirement_ids(task, statuses):
    """Ids of the tasks ``task`` requires whose status requirement is a subset of ``statuses``."""
    out = []
    for req in task.get('taskRequirements') or ():
        tid = (req.get('task') or {}).get('id') if isinstance(req.get('task'), dict) else req.get('task')
        st = set(req.get('status') or ())
        if tid and st and st <= statuses:
            out.append(tid)
    return out


def close_prerequisites(completed, failed, tasks):
    """Fill the holes the logs leave: a task that is completed requires its prerequisites to
    have been completed (a requirement of status ``complete`` means completed; ``complete`` or
    ``failed`` means resolved either way, taken as completed unless known failed; ``failed`` means
    failed).  Returns ``(completed, failed, inferred)``: new sets and the ids that were added."""
    by_id = {t['id']: t for t in tasks or () if t.get('id')}
    completed, failed = set(completed), set(failed)
    inferred = set()
    stack = list(completed)
    while stack:
        t = by_id.get(stack.pop())
        if not t:
            continue
        for req in t.get('taskRequirements') or ():
            rt = req.get('task')
            rid = rt.get('id') if isinstance(rt, dict) else rt
            st = set(req.get('status') or ())
            if not rid or not st or not st <= {'complete', 'failed'}:
                continue                              # needs "active": proves nothing about the end state
            if st == {'failed'}:
                if rid not in completed and rid not in failed:
                    failed.add(rid)
                    inferred.add(rid)
                continue
            if rid in failed and 'failed' in st:
                continue                              # resolved by failing: leave it
            if rid not in completed:
                failed.discard(rid)
                completed.add(rid)
                inferred.add(rid)
                stack.append(rid)
    return completed, failed, inferred


def detect_faction(task_states, tasks):
    """'BEAR' / 'USEC' from the faction-specific tasks the player has started (a task with
    ``factionName`` BEAR or USEC is only offered to that faction), None when none was seen or
    both were (a profile cannot be both)."""
    side = {t['id']: t.get('factionName') for t in tasks or ()
            if t.get('id') and t.get('factionName') in ('BEAR', 'USEC')}
    seen = {side[q] for q in task_states if q in side}
    return seen.pop() if len(seen) == 1 else None


def build_progress(records, mode='auto', tasks=None, faction='auto'):
    """The progress of one mode from cached folder records (see :func:`read_folder`).  Pure.

    ``tasks`` (the tarkov.dev task list) makes unknown quest ids (repeatables) drop out, enables
    the prerequisite closure and the faction detection.  ``faction`` is 'auto', 'BEAR' or 'USEC';
    tasks of the other faction are listed under ``other_faction``."""
    records = list(records)
    if mode in (None, '', 'auto'):
        mode = latest_mode(records)
    known = {t['id'] for t in tasks if t.get('id')} if tasks else None
    rep = replay(records, mode, known)
    states = rep['states']
    completed = {q for q, s in states.items() if s == 'completed'}
    failed = {q for q, s in states.items() if s == 'failed'}
    active = {q for q, s in states.items() if s == 'active'}
    inferred = set()
    if tasks:
        completed, failed, inferred = close_prerequisites(completed, failed, tasks)
        active -= completed | failed
    detected = detect_faction(states, tasks) if tasks else None
    if faction in ('BEAR', 'USEC'):
        side = faction
    else:
        side = detected
    other = []
    if side and tasks:
        other = sorted(t['id'] for t in tasks
                       if t.get('factionName') in ('BEAR', 'USEC') and t['factionName'] != side)
    return {
        'mode': mode,
        'reset_at': rep['reset_at'], 'reset_kind': rep['reset_kind'],
        'completed': sorted(completed), 'failed': sorted(failed), 'active': sorted(active),
        'inferred': sorted(inferred),
        'events': rep['events'], 'last_event_at': rep['last_event_at'],
        'event_at': rep['event_at'],
        'faction': side, 'faction_detected': detected, 'other_faction': other,
        'error': None,
    }


# ---------------------------------------------------------------------------
# The scanner: folder cache + incremental reads
# ---------------------------------------------------------------------------

def _write_json_atomic(path, data):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.', suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class LogScanner:
    """Folder records for one EFT install, cached on disk.

    ``scan()`` brings the records up to date (cheap when nothing changed), ``progress()`` builds
    the quest progress of a mode from them.  Thread-safe."""

    def __init__(self, cache_path=None):
        self.cache_path = cache_path
        self.install_dir = None
        self._folders = {}        # name -> {'sig': [...], 'rec': {...}}
        self._loaded = False
        self._lock = threading.RLock()
        self.last_scan_at = None
        self.last_error = None

    # -- persistence --------------------------------------------------------
    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, encoding='utf-8') as f:
                data = json.load(f)
            if data.get('version') == CACHE_VERSION and isinstance(data.get('folders'), dict):
                self._folders = data['folders']
                self.install_dir = data.get('install_dir')
        except (OSError, ValueError):
            pass

    def _save(self):
        if not self.cache_path:
            return
        try:
            _write_json_atomic(self.cache_path, {'version': CACHE_VERSION, 'install_dir': self.install_dir,
                                                 'folders': self._folders})
        except OSError:
            pass

    # -- scanning -----------------------------------------------------------
    def records(self):
        with self._lock:
            self._load()
            return [f['rec'] for f in self._folders.values()]

    def scan(self, install_dir, full=True):
        """Read what changed.  ``full`` checks every folder's file sizes against the cache;
        otherwise only new folders and the newest (live) one are looked at.  Returns
        ``{'folders': n, 'read': k}``.  Raises OSError when the Logs folder cannot be listed."""
        with self._lock:
            self._load()
            if install_dir and os.path.normpath(install_dir) != os.path.normpath(self.install_dir or ''):
                if self.install_dir:                      # a different install: its folders are not ours
                    self._folders = {}
                self.install_dir = install_dir
            logs = os.path.join(self.install_dir, LOG_DIR_NAME)
            names = {}
            with os.scandir(logs) as it:
                for e in it:
                    ts = folder_start_time(e.name) if e.is_dir() else None
                    if ts is not None:
                        names[e.name] = ts
            newest = max(names, key=names.get) if names else None
            read = 0
            for name, ts in sorted(names.items(), key=lambda kv: kv[1]):
                path = os.path.join(logs, name)
                cached = self._folders.get(name)
                if cached is not None and not full and name != newest:
                    continue
                sig = folder_signature(path)
                if cached is not None and name != newest and cached.get('sig') == sig:
                    continue
                if cached is not None and name == newest and cached.get('sig') == sig and not full:
                    continue                              # live folder, nothing was appended
                self._folders[name] = {'sig': sig, 'rec': read_folder(path, ts)}
                read += 1
            self.last_scan_at = time.time()
            if read:
                self._save()
            return {'folders': len(names), 'read': read}

    # -- results ------------------------------------------------------------
    def progress(self, mode='auto', tasks=None, faction='auto'):
        recs = self.records()
        out = build_progress(recs, mode, tasks, faction)
        out['install_dir'] = self.install_dir
        out['folders'] = len(recs)
        out['scanned_at'] = self.last_scan_at
        return out


# ---------------------------------------------------------------------------
# Live raid state (lobby / matching / loading / in raid) from the running session's log
# ---------------------------------------------------------------------------

# Lines of the live ``application`` log that move the state machine.  Bytes: the log is tailed raw.
_RAID_EVENTS = (
    (b'|application|Matching with group id', 'matching'),
    (b'|application|MatchingCompleted', 'loading'),
    (b'|application|LocationLoaded', 'loading'),
    (b'TRACE-NetworkGameCreate profileStatus', 'loading'),
    (b'|application|GameStarted', 'in_raid'),
    (b'TRACE-NetworkGameCreate 6', 'in_raid'),            # the step right after GameStarted
    (b'|application|BEClient exit', 'lobby'),             # raid over (and once at game launch)
    (b'|application|Application awaken', 'lobby'),
)
_LINE_TS_RE = re.compile(rb'^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)(?:\.\d+)?(?: ([+-]\d\d:\d\d))?\|')
RAID_FIRST_READ_MAX = 4 * 1024 * 1024        # a first look at an old, huge log reads only its tail
RAID_FOLDER_RECHECK_S = 5.0                  # how often the Logs folder is listed for a newer session
RAID_HISTORY_MAX = 512
_MODE_MARK_B = _MODE_MARK.encode()
MODE_HEAD_READ = 256 * 1024                   # where "Session mode:" is logged: the first lines of the session


class RaidState:
    """Where the player is, from the game's own live log: ``lobby`` / ``matching`` / ``loading`` /
    ``in_raid``.  Tails the newest session folder's ``application`` log (only appended bytes are
    read; a newer folder takes over), so ``get()`` is cheap enough to call on every poll.

    ``install_dir`` is a path or a callable returning one (None = not installed / not found)."""

    def __init__(self, install_dir=None, clock=time.time):
        self._install = install_dir
        self._clock = clock
        self._lock = threading.Lock()
        self._folder = None            # name of the session folder being tailed
        self._path = None              # the application log file in it
        self._offset = 0
        self._state = 'lobby'
        self._since = None
        self._history = []             # [(ts, state)] transitions, oldest first
        self._mode = None              # 'pvp' | 'pve' | 'season' from the latest "Session mode:" line
        self._mode_since = None
        self._checked = -1e9
        self._ok = False

    # -- public ---------------------------------------------------------------
    def get(self):
        """``{'state', 'since', 'source': 'logs'}`` or None when the logs cannot be read."""
        with self._lock:
            try:
                self._poll()
            except Exception:
                self._ok = False
            if not self._ok:
                return None
            return {'state': self._state, 'since': self._since, 'source': 'logs'}

    def mode(self):
        """``{'mode': 'pvp'|'pve'|'season', 'since', 'source': 'log'}`` - the game mode the newest
        session announced (``Session mode:`` lines; the latest one wins when the player switches),
        or None when the logs cannot be read or that session has not said yet."""
        with self._lock:
            try:
                self._poll()
            except Exception:
                self._ok = False
            if not self._ok or self._mode is None:
                return None
            return {'mode': self._mode, 'since': self._mode_since, 'source': 'log'}

    def in_raid(self):
        """True / False from the logs, None when unknown (callers fall back to the menu-bar check)."""
        st = self.get()
        return None if st is None else st['state'] == 'in_raid'

    def state_at(self, ts):
        """The state at a past time of this session (a session starts in the lobby)."""
        with self._lock:
            out = 'lobby'
            for t, s in self._history:
                if t > ts:
                    break
                out = s
            return out

    # -- internals --------------------------------------------------------------
    def _install_dir(self):
        inst = self._install() if callable(self._install) else self._install
        return inst or None

    def _reset_session(self):
        self._path, self._offset = None, 0
        self._state, self._since, self._history = 'lobby', None, []
        self._mode = self._mode_since = None

    def _newest_folder(self, inst):
        best, best_ts = None, None
        with os.scandir(os.path.join(inst, LOG_DIR_NAME)) as it:
            for e in it:
                ts = folder_start_time(e.name) if e.is_dir() else None
                if ts is not None and (best_ts is None or ts > best_ts):
                    best, best_ts = e.name, ts
        return best

    def _poll(self):
        inst = self._install_dir()
        if not inst:
            self._ok = False
            return
        now = self._clock()
        if self._folder is None or now - self._checked >= RAID_FOLDER_RECHECK_S:
            self._checked = now
            newest = self._newest_folder(inst)
            if newest is None:
                self._folder = None
                self._ok = False
                return
            if newest != self._folder:
                self._folder = newest
                self._reset_session()
            elif self._path is not None:                   # the log may have rotated to a newer part
                files = _log_files(os.path.join(inst, LOG_DIR_NAME, self._folder))['application']
                if files and files[-1][0] != self._path:
                    self._path, self._offset = files[-1][0], 0
        if self._path is None or not os.path.isfile(self._path):
            files = _log_files(os.path.join(inst, LOG_DIR_NAME, self._folder))['application']
            if not files:
                self._ok = True            # session just started: nothing logged yet = lobby
                return
            self._path, self._offset = files[-1][0], 0     # (a rotated part, _001, starts fresh)
        self._tail()
        self._ok = True

    def _tail(self):
        size = os.path.getsize(self._path)
        if size < self._offset:            # truncated / replaced: start over
            self._reset_session_keep_path()
        if size == self._offset:
            return
        start = self._offset
        mid_line = False
        if start == 0 and size > RAID_FIRST_READ_MAX:
            start, mid_line = size - RAID_FIRST_READ_MAX, True
            self._read_head_mode()
        with open(self._path, 'rb') as f:
            f.seek(start)
            data = f.read(size - start)
        end = data.rfind(b'\n')
        if end < 0:                        # no complete line yet
            return
        self._offset = start + end + 1
        lines = data[:end].split(b'\n')
        for line in (lines[1:] if mid_line else lines):
            self._feed(line)

    def _read_head_mode(self):
        """A huge log is only read from its tail; the mode is announced at the top - read that."""
        try:
            with open(self._path, 'rb') as f:
                head = f.read(MODE_HEAD_READ).decode('utf-8', 'replace')
            modes = extract_session_modes(head)
            if modes:
                self._mode_since, self._mode = modes[-1][0], modes[-1][1]
        except OSError:
            pass

    def _reset_session_keep_path(self):
        path = self._path
        self._reset_session()
        self._path = path

    @staticmethod
    def _line_ts(line):
        m = _LINE_TS_RE.match(line)
        if m:
            try:
                return parse_log_time(m.group(1).decode(), m.group(2).decode(),
                                      m.group(3).decode() if m.group(3) else None)
            except ValueError:
                pass
        return None

    def _feed(self, line):
        i = line.find(_MODE_MARK_B)
        if i >= 0:
            word = line[i + len(_MODE_MARK_B):].decode('utf-8', 'replace').strip().split()
            if word:
                self._mode = normalize_mode(word[0])
                self._mode_since = self._line_ts(line)
            return
        new = None
        for mark, state in _RAID_EVENTS:
            if mark in line:
                new = state
                break
        if new is None or new == self._state:
            return
        ts = self._line_ts(line)
        if ts is None:
            ts = self._clock()
        self._state, self._since = new, ts
        self._history.append((ts, new))
        del self._history[:-RAID_HISTORY_MAX]
