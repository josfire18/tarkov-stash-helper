"""
Which language the Tarkov client is set to (stdlib only).

The stash prints every item's *short name* in the client's language, so the label anchor and the OCR
stage must compare against that language's short names.  The client keeps the choice in
``%APPDATA%\\Battlestate Games\\Escape from Tarkov\\Settings\\Game.ini`` (a JSON file with a
``"Language"`` key); the launcher's own ``BsgLauncher\\settings`` has a ``language`` key too and is
the fallback before the game has ever been started.

tarkov.dev publishes a translation document per language (``items_<code>``).  A language is
``label_ok`` when its labels are Latin script (Tesseract's English model and the game's Bender label
font can read them); Russian, Japanese, Korean and Chinese labels cannot be read that way, so for them
the app identifies items by their picture only and says so.
"""
from __future__ import annotations

import json
import os

# client code -> (tarkov.dev document suffix, display name, label text readable by the label anchor / OCR)
LANGUAGES = {
    'en': ('en', 'English', True),
    'ru': ('ru', 'Russian', False),
    'fr': ('fr', 'French', True),
    'ge': ('de', 'German', True), 'de': ('de', 'German', True),
    'es': ('es', 'Spanish', True),
    'es-mx': ('es', 'Spanish (Mexico)', True),
    'pl': ('pl', 'Polish', True),
    'cz': ('cs', 'Czech', True), 'cs': ('cs', 'Czech', True),
    'hu': ('hu', 'Hungarian', True),
    'it': ('it', 'Italian', True),
    'jp': ('ja', 'Japanese', False), 'ja': ('ja', 'Japanese', False),
    'kr': ('ko', 'Korean', False), 'ko': ('ko', 'Korean', False),
    'po': ('pt', 'Portuguese', True), 'pt': ('pt', 'Portuguese', True), 'pt-br': ('pt', 'Portuguese', True),
    'ro': ('ro', 'Romanian', True),
    'sk': ('sk', 'Slovak', True),
    'tu': ('tr', 'Turkish', True), 'tr': ('tr', 'Turkish', True),
    'ch': ('zh', 'Chinese', False), 'cn': ('zh', 'Chinese', False), 'zh': ('zh', 'Chinese', False),
}


def _read_json(path):
    try:
        with open(path, encoding='utf-8-sig') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def client_language_code(appdata=None) -> str | None:
    """The raw language code the client (or, before it ever ran, the launcher) is set to."""
    appdata = appdata if appdata is not None else os.environ.get('APPDATA')
    if not appdata:
        return None
    root = os.path.join(appdata, 'Battlestate Games')
    game = _read_json(os.path.join(root, 'Escape from Tarkov', 'Settings', 'Game.ini'))
    for v in (game.get('Language'), _read_json(os.path.join(root, 'BsgLauncher', 'settings')).get('language')):
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return None


def describe(code: str | None, override: str | None = None) -> dict:
    """What the app does for a language code: ``{'code', 'name', 'doc', 'supported', 'label_ok'}``.
    ``doc`` is the tarkov.dev translation suffix (None for English / unknown); ``supported`` is
    False for a code the app has no translation for (then it behaves like English and says so);
    ``label_ok`` False means picture-only identification."""
    code = (override or code or 'en').strip().lower()
    if code in ('', 'auto'):
        code = 'en'
    info = LANGUAGES.get(code)
    if info is None:
        return {'code': code, 'name': code, 'doc': None, 'supported': False, 'label_ok': False}
    doc, name, label_ok = info
    return {'code': code, 'name': name, 'doc': None if doc == 'en' else doc, 'supported': True,
            'label_ok': label_ok}


def detect(override: str | None = None, appdata=None) -> dict:
    return describe(client_language_code(appdata), override)


def notice(lang: dict) -> str | None:
    """One plain sentence when the language limits what the app can do, else None."""
    if lang['supported'] and lang['label_ok']:
        return None
    nm = lang['name'] if lang['supported'] else f"\"{lang['code']}\""
    return (f"Your game is set to {nm}: item names can't be read from the screen in that language, "
            f"so items are identified by their picture only.")
