import os
import os.path as osp
import copy
import json
import numpy as np
from PIL import Image
import dnnlib
from stylegan.renderer_v2 import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL
from predictor import Predictor
from utils import validate_mutation
from multiprocessing import Process, Pool, set_start_method
class mimicry:

    def __init__(self, class_idx=None, w0_seed=0, stylemix_seed=0, search_limit=SEARCH_LIMIT , step_size=1):
        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.stylemix_seed = stylemix_seed
        self.search_limit = search_limit
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        self.layers = [[7], [6], [5], [4], [3], [5,6], [3,4], [3,4,5,6]]
        self.step_size = step_size
        self.state['renderer'] = Renderer()
        # self.res = dnnlib.EasyDict()

    def render_state(self, state=None):
        if state is None:
            state = self.state
        state['renderer']._render_impl(
            res = state['generator_params'],  # res
            pkl = INIT_PKL,  # pkl
            w0_seeds= state['params']['w0_seeds'],  # w0_seed,
            class_idx = state['params']['class_idx'],  # class_idx,
            mixclass_idx = state['params']['mixclass_idx'],  # mix_idx,
            stylemix_idx = state['params']['stylemix_idx'],  # stylemix_idx,
            stylemix_seed = state['params']['stylemix_seed'],  # stylemix_seed,
            img_normalize = state['params']['img_normalize'],
            to_pil = state['params']['to_pil'],
        )

        info =  copy.deepcopy(state['params'])

        return state, info

    def search(self):
        max_seed_count = 100
        root = f"svhn/ices"

        frontier_seed_count_100c = 0
        frontier_seed_count_not_100c = 0
        # while frontier_seed_count_100c < max_seed_count or frontier_seed_count_not_100c < max_seed_count:
        # while frontier_seed_count_not_100c < max_seed_count:
        while frontier_seed_count_100c < max_seed_count:
            state = self.state

            state["params"]["class_idx"] = self.class_idx
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
            state["params"]["stylemix_idx"] = []
            state["params"]["mixclass_idx"] = None
            state["params"]["stylemix_seed"] = None

            digit, digit_info = self.render_state()


            label = digit["params"]["class_idx"]
            image = digit['generator_params'].image

            image_array = np.array(image)

            accepted, confidence, predictions = Predictor().predict_datapoint(
                np.reshape(image_array, (-1, 32, 32, 1)),
                label
            )


            if accepted:
                _ , second_cls = np.argsort(-predictions)[:2]
                second_cls_confidence = predictions[second_cls]
                if second_cls_confidence and frontier_seed_count_not_100c < max_seed_count:
                  frontier_seed_count_not_100c += 1
                  img_folder = f"{root}/not_100/{self.class_idx}"
                  img_path = f"{img_folder}/{self.w0_seed}.png"
                  if not os.path.exists(img_folder):
                      os.makedirs(img_folder, exist_ok=True)
                  image.save(img_path)

                  digit_info["accepted"] = accepted.tolist()
                  digit_info["exp-confidence"] = float(confidence)
                  digit_info["predictions"] = predictions.tolist()
                  with open(f"{img_folder}/{self.w0_seed}.json", 'w') as f:
                      (json.dump(digit_info, f, sort_keys=True, indent=4))
                elif not second_cls_confidence and frontier_seed_count_100c < max_seed_count:
                  frontier_seed_count_100c += 1
                  img_folder = f"{root}/100/{self.class_idx}"
                  img_path = f"{img_folder}/{self.w0_seed}.png"
                  if not os.path.exists(img_folder):
                      os.makedirs(img_folder, exist_ok=True)
                  image.save(img_path)

                  digit_info["accepted"] = accepted.tolist()
                  digit_info["exp-confidence"] = float(confidence)
                  digit_info["predictions"] = predictions.tolist()
                  with open(f"{img_folder}/{self.w0_seed}.json", 'w') as f:
                      (json.dump(digit_info, f, sort_keys=True, indent=4))
            self.w0_seed += self.step_size





def run_mimicry(class_idx, w0_seed=0, step_size=1):
    mimicry(class_idx=class_idx, w0_seed=w0_seed, step_size=step_size).search()

if __name__ == "__main__":
    for i in range(10):
        run_mimicry(i)
    # run_mimicry(6, 11449)

