"""Every inline <script> in the page templates must parse: one syntax error (a stray ' in a help text)
leaves a page on its "Loading..." placeholder forever.  Needs Node.js on PATH; skipped without it."""
import glob
import os
import re
import shutil
import subprocess

import pytest

TEMPLATES = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
NODE = shutil.which('node')
SCRIPT_RE = re.compile(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', re.S)


def _scripts():
    for path in sorted(glob.glob(os.path.join(TEMPLATES, '*.html'))):
        with open(path, encoding='utf-8') as f:
            for i, m in enumerate(SCRIPT_RE.finditer(f.read())):
                yield pytest.param(m.group(1), id=f'{os.path.basename(path)}#{i}')


@pytest.mark.skipif(NODE is None, reason='Node.js not installed')
@pytest.mark.parametrize('js', list(_scripts()))
def test_inline_script_parses(js, tmp_path):
    src = tmp_path / 'page.js'
    src.write_text(js, encoding='utf-8')
    r = subprocess.run([NODE, '--check', str(src)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
