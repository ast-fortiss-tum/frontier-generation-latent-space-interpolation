import os
import copy
import json
import time
import numpy as np
from stylegan.renderer import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL, STYLEMIX_LAYERS, FRONTIER_PAIRS
from predictor import Predictor
from utils import validate_mutation
import time

class mimicry:

    def __init__(self, class_idx=None, w0_seed=0, stylemix_seed=0, search_limit=SEARCH_LIMIT , step_size=1):
        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.stylemix_seed = stylemix_seed
        self.search_limit = search_limit
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        self.layers = STYLEMIX_LAYERS
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
        targeted = False
        # seeds_root = "svhn/final-svhn/100"
        # p_100 = True
        seeds_root = "svhn/final-svhn/not_100"
        p_100 = False
        iter = [[] for _ in range(10)]
        t = [[] for _ in range(10)]
        pairs = [[] for _ in range(10)]
        bf_pairs = [0 for _ in range(10)]
        seed_classes = [f for f in os.listdir(seeds_root) if os.path.isdir(os.path.join(seeds_root, f))]
        for seed_class in seed_classes:
            self.class_idx = int(seed_class)
            c_i = 0
            c_t0 = time.time()
            c_pairs = 0
            for seed in os.listdir(os.path.join(seeds_root, seed_class)):
                if seed.endswith(".json"):
                  seed, _ = os.path.splitext(seed)
                  self.w0_seed = int(seed)
                  json_path = os.path.join(seeds_root, seed_class, f"{seed}.json")
                  with open(json_path, 'r') as f:
                    data = json.load(f)
                    predictions = data['predictions']
                    print(f"Predictions: {predictions}")

                    _ , stylemix_cls = np.argsort(-np.array(predictions))[:2]
                    stylemix_cls = int(stylemix_cls)
                    second_cls_confidence = predictions[stylemix_cls]


                  print(os.path.join(seeds_root, seed_class, f"{seed}.png"))
                  print(f"Class: {self.class_idx} - Seed: {self.w0_seed} - Second Class: {stylemix_cls} - Confidence: {second_cls_confidence}")


                  state = self.state
                  state["params"]["class_idx"] = self.class_idx
                  state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
                  state["params"]["stylemix_idx"] = []
                  state["params"]["mixclass_idx"] = None
                  state["params"]["stylemix_seed"] = None
                  digit, digit_info = self.render_state()
                  image = digit['generator_params'].image
                  image_array = np.array(image)

                  found_at_least_one = False
                  found_mutation = False
                  tried_all_layers = False
                  self.stylemix_seed = 1
                  best_found = {}
                  state["params"]["mixclass_idx"] = stylemix_cls
                  while not found_mutation and not tried_all_layers and self.stylemix_seed <= self.stylemix_seed_limit:

                      state["params"]["stylemix_seed"] = np.random.randint(0, 100000)
                      if p_100:
                        state["params"]["mixclass_idx"] = np.random.randint(0, 10)


                      for idx, layer in enumerate(self.layers):
                          state["params"]["stylemix_idx"] = layer

                          m_digit, m_digit_info = self.render_state()
                          m_image = m_digit['generator_params'].image

                          m_image_array = np.array(m_image)


                          m_accepted, confidence , m_predictions = Predictor().predict_datapoint(
                              np.reshape(m_image_array, (-1, 32, 32, 1)),
                              self.class_idx
                          )

                          m_class = np.argsort(-m_predictions)[:1]
                          # misclassification and decision boundary check
                          if not m_accepted and (not targeted or stylemix_cls == m_class):
                              path = f"{FRONTIER_PAIRS}/{self.class_idx}/{self.w0_seed}/"
                              seed_name = f"0-{stylemix_cls}"
                              img_path = f"{path}/{seed_name}.png"
                              if not os.path.exists(img_path):
                                  os.makedirs(path, exist_ok=True)
                                  image.save(img_path)

                                  with open(f"{path}/{seed_name}.json", 'w') as f:
                                      (json.dump(digit_info, f, sort_keys=True, indent=4))

                              valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)
                              digit_info["l2_norm"] = img_l2

                              if valid_mutation:
                                  if not found_at_least_one:
                                      c_pairs += 1
                                      found_at_least_one = True


                                  found_mutation = True
                                  m_digit_info["accepted"] = m_accepted.tolist()
                                  m_digit_info["predicted-class"] = m_class.tolist()
                                  m_digit_info["exp-confidence"] = float(confidence)
                                  m_digit_info["predictions"] = m_predictions.tolist()
                                  m_digit_info["ssi"] = float(ssi)
                                  m_digit_info["l2_norm"] = m_img_l2
                                  m_digit_info["l2_distance"] = l2_distance
                                  m_digit_info["iteration-count"] = self.stylemix_seed

                                  m_path = f"{path}/{stylemix_cls}"
                                  m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{m_class}"
                                  os.makedirs(m_path, exist_ok=True)
                                  with open(f"{m_path}/{m_name}.json", 'w') as f:
                                      (json.dump(m_digit_info, f, sort_keys=True, indent=4))
                                  m_image.save(f"{m_path}/{m_name}.png")
                              else:
                                if not best_found or ssi < best_found["ssi"]:
                                  print(f"Best Found: {ssi}")
                                  best_found =  copy.deepcopy(m_digit_info)
                                  best_found["accepted"] = m_accepted.tolist()
                                  best_found["predicted-class"] = m_class.tolist()
                                  best_found["exp-confidence"] = float(confidence)
                                  best_found["predictions"] = m_predictions.tolist()
                                  best_found["ssi"] = float(ssi)
                                  best_found["l2_norm"] = m_img_l2
                                  best_found["l2_distance"] = l2_distance
                          if idx == len(self.layers) and found_mutation:
                              tried_all_layers = True
                              break
                      self.stylemix_seed += 1
                      c_i += 1
                  if not found_mutation and best_found:
                      bf_pairs[self.class_idx] += 1
                      l2_distance = best_found["l2_distance"]
                      ssi = best_found["ssi"]
                    #   self.stylemix_seed = best_found["stylemix_seed"]
                      stylemix_cls = best_found["mixclass_idx"]
                      layer = best_found["stylemix_idx"]
                      m_class = best_found["predicted-class"]
                      best_found["iteration-count"] = self.stylemix_seed

                      m_path = f"{path}/{stylemix_cls}-bf/"
                      m_name = f"/{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{m_class}"
                      os.makedirs(m_path, exist_ok=True)
                      with open(f"{m_path}/{m_name}.json", 'w') as f:
                          (json.dump(best_found, f, sort_keys=True, indent=4))
                      m_image.save(f"{m_path}/{m_name}.png")
            c_tf = time.time()
            iter[self.class_idx] = c_i
            t[self.class_idx] = c_tf-c_t0
            pairs[self.class_idx] = c_pairs
        print(f"Total Iterations: {iter}")
        print(f"Total Time: {t}")
        results = {
            "iterations": iter,
            "time": t,
            "pairs": pairs,
            "bf_pairs": bf_pairs
        }
        with open(f"{FRONTIER_PAIRS}/results.json", 'w') as f:
            (json.dump(results, f, sort_keys=True, indent=4))
        print(f"Targeted: {targeted}")



if __name__ == "__main__":
    mimicry().search()
