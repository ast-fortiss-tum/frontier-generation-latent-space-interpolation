import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dnnlib
import torch

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# StyleGAN2 model checkpoint
INIT_PKL = '/Users/gigimerabishvili/Desktop/frontier-generation-latent-space-interpolation/Checkpoints/Checkpoints/stylegan2_svhn_gray_32x32-con.pkl'
# Model used for prediction
MODEL = 'svhn/models/Model1_svhn_gray.weights'
num_classes = 10

# Path to save the generated frontier pairs
FRONTIER_PAIRS = 'svhn/eval'
# List of layers to perform stylemix
STYLEMIX_LAYERS = [[7], [6], [5], [4], [3], [5,6], [3,4], [3,4,5,6]]
# Number of frontier pair samples to generate
SEARCH_LIMIT = 30
# Max number of stylemix seeds
STYLEMIX_SEED_LIMIT = 100

SSIM_THRESHOLD = 0.95
L2_RANGE = 0.2

# Value for truncation psi
TRUNC_PSI = 0
TRUNC_CUTOFF = 0

STYLEGAN_INIT = {
    "generator_params": dnnlib.EasyDict(),
    "params": {
        "w0_seeds": [[0, 1]],
        "w_load": None,
        "class_idx": None,
        "mixclass_idx": None,
        "stylemix_idx": [],
        "patch_idxs": None,
        "stylemix_seed": None,
        "trunc_psi": 1,
        "trunc_cutoff": 0,
        "random_seed": 0,
        "noise_mode": 'const',
        "force_fp32": False,
        "layer_name": None,
        "sel_channels": 3,
        "base_channel": 0,
        "img_scale_db": 0,
        "img_normalize": True,
        "to_pil": True,
        "input_transform" : None,
        "untransform": False,
    },
    "device": DEVICE,
    "renderer": None,
    'pretrained_weight': INIT_PKL
}
