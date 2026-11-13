"""Write one config's weights (fp32 safetensors) + shared initial noise, so every engine sees identical inputs.
XL/2 uses the official checkpoint; the rest are random. adaLN/final layers are zero-init in models.py, which would make
v=0 and skip the whole network's effect, so they get N(0, 0.02) here."""
import sys, time
sys.path.insert(0, "sit")
import numpy as np, torch
from safetensors.torch import save_file
from models import SiT_models
name, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.manual_seed(seed)
m = SiT_models[name](input_size=32, num_classes=1000, learn_sigma=True)
if name == "SiT-XL/2":
    m.load_state_dict(torch.load("pretrained_models/SiT-XL-2-256x256.pt", map_location="cpu", weights_only=True))
else:
    for k, v in m.named_parameters():
        if "adaLN_modulation" in k or "final_layer" in k:
            torch.nn.init.normal_(v, std=0.02)
save_file({k: v.detach().contiguous() for k, v in m.state_dict().items()}, out + ".safetensors")
z = torch.randn(8, 4, 32, 32, generator=torch.Generator().manual_seed(seed)).numpy()
np.save(out + ".noise.npy", z)
