"""
Export the stage-2 appearance model (DINOv2-small) to ONNX so the packaged exe can run the
re-rank with onnxruntime instead of torch + transformers.

    python scripts/export_dino_onnx.py [--out data/dino_v2s.onnx] [--fp16]

The graph takes ``pixel_values`` [n, 3, 224, 224] (ImageNet-normalised RGB) and returns the
same L2-normalised [n, 768] embedding as :func:`identify.dino.embed` (CLS token + mean patch
token).  Prints the SHA-256 that ``identify.dino.ONNX_SHA256`` must carry.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    import numpy as np
    import torch
    from transformers import AutoModel

    from identify.dino import MODEL_ID, RES
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'dino_v2s.onnx'))
    ap.add_argument('--fp16', action='store_true', help='store weights as fp16 (half the file size)')
    a = ap.parse_args()

    base = AutoModel.from_pretrained(MODEL_ID).eval().float()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            h = self.m(pixel_values=x).last_hidden_state
            e = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
            return torch.nn.functional.normalize(e, dim=1)

    w = Wrap(base).eval()
    x = torch.randn(2, 3, RES, RES)
    torch.onnx.export(w, (x,), a.out, input_names=['pixel_values'], output_names=['embedding'],
                      dynamic_axes={'pixel_values': {0: 'n'}, 'embedding': {0: 'n'}},
                      opset_version=17, dynamo=False)
    if a.fp16:
        import onnx
        from onnx import numpy_helper
        m = onnx.load(a.out)
        # weights only (initializers) -> fp16 storage + Cast back to fp32 at load is what
        # onnxconverter does; keep it simple: full fp32 graph is the default
        raise SystemExit('fp16 export not implemented; the fp32 file is used')
    import onnxruntime as ort
    s = ort.InferenceSession(a.out, providers=['CPUExecutionProvider'])
    ref = w(x).detach().numpy()
    got = s.run(None, {'pixel_values': x.numpy()})[0]
    print('max |onnx - torch| =', float(np.abs(ref - got).max()))
    h = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
    print(f'{a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB  sha256={h}')


if __name__ == '__main__':
    main()
