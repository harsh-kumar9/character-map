"""Boundary-token reconstruction: FVU with each frame's own variance vs relative error ||x - x_hat||^2 / ||x||^2,
and FVU against the in-distribution boundary variance. Reads cm.stimx acts (float32 boundary readouts)."""
import json, glob, sys
import numpy as np, torch
sys.path.insert(0, "/ada1/u/harsh/character-map")
from cm.paths import sae_dir
from cm.sae import JumpReLUSAE
rng = np.random.default_rng(0)
X = {}
for p in sorted(glob.glob("/ada1/u/harsh/character-map/outputs/acts/stimuli/*.npz")):
    z = np.load(p, mmap_mode="r"); meta = json.loads(str(z["meta"]))
    for f in "abcd":
        idx = [i for i, m in enumerate(meta) if m["frame"] == f and m["readout"] == "boundary"]
        if idx:
            pick = rng.choice(idx, size=min(20, len(idx)), replace=False)
            X.setdefault(f, []).append(np.asarray(z["boundary"][np.sort(pick)]))
torch.set_num_threads(32)
print("frame  L   n    FVU(own var)  rel_err   mean pairwise cos")
for L, hi in ((9, 10), (17, 18), (22, 23), (29, 30)):
    sae = JumpReLUSAE.load(sae_dir("gemma-3-4b-it", L))
    for f in "abcd":
        x = torch.from_numpy(np.concatenate(X[f])[:, hi].astype(np.float32))
        xh = sae.decode(sae.encode(x))
        se = ((x - xh) ** 2).sum()
        fvu = se / ((x - x.mean(0)) ** 2).sum()
        rel = se / (x ** 2).sum()
        xn = x / x.norm(dim=1, keepdim=True)
        cos = (xn @ xn.T).fill_diagonal_(0).sum() / (len(x) * (len(x) - 1))
        print("%s     %2d %4d   %.3f        %.4f    %.3f" % (f, L, len(x), fvu, rel, cos))
