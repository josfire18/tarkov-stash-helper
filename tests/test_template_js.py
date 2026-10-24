"""Every inline <script> in the page templates must parse: one syntax error (a stray ' in a help text)
leaves a page on its "Loading..." placeholder forever.  Needs Node.js on PATH; skipped without it."""
import glob
import os
import shutil
import subprocess
from html.parser import HTMLParser

import pytest

TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
NODE = shutil.which('node')


class _InlineScripts(HTMLParser):
    """Collects the text of every <script> without a src attribute."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.scripts, self._cur = [], None

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and 'src' not in dict(attrs):
            self._cur = []

    def handle_data(self, data):
        if self._cur is not None:
            self._cur.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self._cur is not None:
            self.scripts.append(''.join(self._cur))
            self._cur = None


def _scripts():
    for path in sorted(glob.glob(os.path.join(TEMPLATES, '*.html'))):
        with open(path, encoding='utf-8') as f:
            parser = _InlineScripts()
            parser.feed(f.read())
        for i, js in enumerate(parser.scripts):
            yield pytest.param(js, id=f'{os.path.basename(path)}#{i}')


@pytest.mark.skipif(NODE is None, reason='Node.js not installed')
@pytest.mark.parametrize('js', list(_scripts()))
def test_inline_script_parses(js, tmp_path):
    src = tmp_path / 'page.js'
    src.write_text(js, encoding='utf-8')
    r = subprocess.run([NODE, '--check', str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
