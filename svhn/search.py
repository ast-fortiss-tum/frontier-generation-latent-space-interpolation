import os
import copy
import json
import numpy as np
import sys
import random

from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stylegan.renderer import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL, STYLEMIX_LAYERS, FRONTIER_PAIRS
from predictor import Predictor
from utils import validate_mutation


def preprocess_image_for_predictor(pil_img, grayscale=True):

    img_arr = np.array(pil_img, dtype=np.float32)

    if grayscale:
       
        if img_arr.ndim == 3 and img_arr.shape[-1] == 3:
            img_arr = np.mean(img_arr, axis=-1, keepdims=True)

    img_arr /= 255.0

    if grayscale:
        if img_arr.ndim == 2:
            img_arr = np.reshape(img_arr, (1, 32, 32, 1))
        elif img_arr.ndim == 3 and img_arr.shape[-1] == 1:
            img_arr = np.expand_dims(img_arr, 0)
    else:
        if img_arr.shape == (32, 32, 3):
            img_arr = np.expand_dims(img_arr, 0) 

    return img_arr


class mimicry:
    def __init__(self,
                 class_idx=None,
                 w0_seed=0,
                 stylemix_seed=0,
                 search_limit=SEARCH_LIMIT,
                 step_size=1,
                 save_base_if_no_pairs=False):

        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.stylemix_seed = stylemix_seed
        self.search_limit = search_limit
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        self.layers = STYLEMIX_LAYERS
        self.step_size = step_size
        self.save_base_if_no_pairs = save_base_if_no_pairs

        self.state['renderer'] = Renderer()

    def render_state(self, state=None, alpha=1.0):

        if state is None:
            state = self.state

        state["params"]["INTERPOLATION_ALPHA"] = alpha

        state['renderer']._render_impl(
            res=state['generator_params'],
            pkl=INIT_PKL,
            w0_seeds=state['params']['w0_seeds'],
            class_idx=state['params']['class_idx'],
            mixclass_idx=state['params']['mixclass_idx'],
            stylemix_idx=state['params']['stylemix_idx'],
            stylemix_seed=state['params']['stylemix_seed'],
            trunc_psi=state['params']['trunc_psi'],
            trunc_cutoff=state['params']['trunc_cutoff'],
            img_normalize=state['params']['img_normalize'],
            to_pil=state['params']['to_pil'],
            INTERPOLATION_ALPHA=state['params']['INTERPOLATION_ALPHA'],
            noise_mode='const'
        )
        info = copy.deepcopy(state['params'])
        return state, info

    def search(self):
        """
           - Render the base image
           - Check if it's correctly classified as `self.class_idx`
           - If correct, attempt stylemix w/ alpha=1.0(fully switch) to see if it misclassifies to a target class.
           - If misclassified, do a binary search for interpolation factor to refine the frontier pairs.

        """
        root = f"{FRONTIER_PAIRS}/{self.class_idx}/"
        frontier_seed_count = 0

        while frontier_seed_count < self.search_limit:
            state = self.state

            state["params"]["class_idx"] = self.class_idx
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
            state["params"]["stylemix_idx"] = []
            state["params"]["mixclass_idx"] = None
            state["params"]["stylemix_seed"] = None

            # Render base image
            digit, digit_info = self.render_state(alpha=1.0)
            label = digit["params"]["class_idx"]
            base_pil = digit['generator_params'].image

            # Convert PIL -> NumPy -> shape (1, 32, 32, 1).
            base_arr_for_pred = preprocess_image_for_predictor(base_pil, grayscale=True)

            # Predict
            accepted, confidence, predictions = Predictor().predict_datapoint(
                base_arr_for_pred,
                label
            )

            print(
                f"Seed: {self.w0_seed} | Label: {label} | "
                f"Accepted: {accepted} | Confidence: {confidence:.4f} | "
                f"Top Predictions: {[round(p,4) for p in predictions]}"
            )

            digit_info["accepted"] = bool(accepted)
            digit_info["exp-confidence"] = float(confidence)
            digit_info["predictions"] = [float(p) for p in predictions]

            frontier_found_for_this_seed = False

            # If Accepted, attempt stylemix for misclassification
            if accepted:
                sorted_classes = np.argsort(-predictions)
                top_cls = sorted_classes[0]
                second_cls = sorted_classes[1]
                second_conf = predictions[second_cls]

                # If second most-likely class is too tiny, skip
                if second_conf < 0.01:
                    self.w0_seed += self.step_size
                    continue

                target_cls = int(second_cls)
                state["params"]["mixclass_idx"] = target_cls
                found_mutation = False

                # Try different stylemix_seed values
                self.stylemix_seed = 0
                while not found_mutation and self.stylemix_seed < self.stylemix_seed_limit:
                    # Avoid stylemix_seed == w0_seed
                    if self.stylemix_seed == self.w0_seed:
                        self.stylemix_seed += 1
                        continue

                    state["params"]["stylemix_seed"] = int(self.stylemix_seed)

                    
                    for layer in self.layers:
                        state["params"]["stylemix_idx"] = layer

                        # Quick test at alpha=1.0
                        _, m_info = self.render_state(alpha=1.0)
                        m_pil = state['generator_params'].image

                        # Preprocess
                        m_for_pred = preprocess_image_for_predictor(m_pil, grayscale=True)
                        m_accepted, m_confidence, m_preds = Predictor().predict_datapoint(
                            m_for_pred, label
                        )
                        m_class = np.argmax(m_preds)

                        # If misclassifies exactly to 'target_cls', do a binary search
                        if (not m_accepted) and (m_class == target_cls):
                            alpha_min = 0.0
                            alpha_max = 1.0
                            max_iterations = 15
                            iteration = 0
                            tolerance = 1e-8

                            last_correct_image = None
                            last_correct_alpha = None
                            last_correct_confidence = None
                            last_correct_preds = None

                            # Base array for validation checks
                            base_np = np.array(base_pil, dtype=np.float32)

                            while iteration < max_iterations and (alpha_max - alpha_min) > tolerance:
                                alpha_mid = (alpha_min + alpha_max) / 2.0
                                # Render with alpha_mid
                                _, _ = self.render_state(alpha=alpha_mid)
                                candidate_pil = state['generator_params'].image
                                cand_for_pred = preprocess_image_for_predictor(candidate_pil, grayscale=True)

                                c_accepted, c_conf, c_preds = Predictor().predict_datapoint(
                                    cand_for_pred, label
                                )
                                c_class = np.argmax(c_preds)

                                if c_accepted:
                                    # Still recognized 
                                    alpha_min = alpha_mid
                                    last_correct_image = np.array(candidate_pil, dtype=np.uint8)
                                    last_correct_alpha = alpha_mid
                                    last_correct_confidence = c_conf
                                    last_correct_preds = c_preds.copy()
                                else:
                                    # Misclass
                                    alpha_max = alpha_mid

                                iteration += 1

                            # Check final alpha at alpha_max
                            _, _ = self.render_state(alpha=alpha_max)
                            final_pil = state['generator_params'].image
                            final_arr = np.array(final_pil, dtype=np.uint8)
                            f_for_pred = preprocess_image_for_predictor(final_pil, grayscale=True)

                            f_accepted, f_conf, f_preds = Predictor().predict_datapoint(
                                f_for_pred, label
                            )
                            f_class = np.argmax(f_preds)

                            if (not f_accepted) and (f_class == target_cls):
                                # Validate mutation
                                valid_mutation, ssi, l2_dist, img_l2, m_img_l2 = validate_mutation(
                                    base_np, final_arr
                                )

                                if valid_mutation:
                                    found_mutation = True
                                    frontier_found_for_this_seed = True
                                    frontier_seed_count += 1

                                    # Make sure directories exist
                                    path_cls = os.path.join(root, str(self.w0_seed), str(target_cls))
                                    os.makedirs(path_cls, exist_ok=True)
                                    base_path = os.path.join(root, str(self.w0_seed))
                                    os.makedirs(base_path, exist_ok=True)

                                    # Save base if not already
                                    base_name = f"0-{int(label)}"
                                    base_png = os.path.join(base_path, base_name + ".png")
                                    if not os.path.exists(base_png):
                                        base_pil.save(base_png)
                                        digit_info["l2_norm"] = float(np.linalg.norm(base_np))
                                        with open(os.path.join(base_path, base_name + ".json"), 'w') as f:
                                            json.dump(digit_info, f, sort_keys=True, indent=4)

                                    # Save final misclassified
                                    frontier_info = copy.deepcopy(m_info)
                                    frontier_info["accepted"] = bool(f_accepted)
                                    frontier_info["predicted-class"] = int(f_class)
                                    frontier_info["exp-confidence"] = float(f_conf)
                                    frontier_info["predictions"] = [float(p) for p in f_preds]
                                    frontier_info["ssi"] = float(ssi)
                                    frontier_info["l2_norm"] = float(m_img_l2)
                                    frontier_info["l2_distance"] = float(l2_dist)
                                    frontier_info["alpha"] = float(alpha_max)

                                    layer_str = '-'.join(map(str, layer))
                                    mis_name = (
                                        f"{int(l2_dist)}-"
                                        f"{int(ssi*100)}-"
                                        f"{self.stylemix_seed}-"
                                        f"{int(target_cls)}-"
                                        f"{layer_str}-"
                                        f"{alpha_max:.6f}-mis"
                                    )

                                    final_path = os.path.join(path_cls, mis_name + ".png")
                                    final_json = os.path.join(path_cls, mis_name + ".json")

                                    Image.fromarray(final_arr).save(final_path)
                                    with open(final_json, 'w') as f:
                                        json.dump(frontier_info, f, sort_keys=True, indent=4)

                                    # Save the last correct image
                                    if last_correct_image is not None:
                                        correct_pil = Image.fromarray(last_correct_image)
                                        correct_name = (
                                            f"{int(l2_dist)}-"
                                            f"{int(ssi*100)}-"
                                            f"{self.stylemix_seed}-"
                                            f"{int(target_cls)}-"
                                            f"{layer_str}-"
                                            f"{last_correct_alpha:.6f}-correct.png"
                                        )
                                        correct_path = os.path.join(path_cls, correct_name)
                                        correct_pil.save(correct_path)

                                        # JSON for correct image
                                        correct_info = copy.deepcopy(frontier_info)
                                        correct_info["accepted"] = True
                                        correct_info["predicted-class"] = int(label)
                                        correct_info["exp-confidence"] = float(last_correct_confidence)
                                        correct_info["predictions"] = [float(x) for x in last_correct_preds]
                                        correct_info["alpha"] = float(last_correct_alpha)

                                        with open(correct_path + '.json', 'w') as f:
                                            json.dump(correct_info, f, sort_keys=True, indent=4)

                                else:
                                    print("Invalid mutation (SSIM/L2 check failed). Skipping.")
                            else:
                                print("Refined boundary not misclassified to target_cls. Skipping...")

                        if found_mutation:
                            break  # from stylemix_idx

                    if found_mutation:
                        break  # from stylemix_seed

                    self.stylemix_seed += 1

            # If not accepted
            else:
                print(f"Seed {self.w0_seed} not accepted. Skipping any interpolation.")

            if frontier_found_for_this_seed or self.save_base_if_no_pairs:
                path_seed = os.path.join(root, str(self.w0_seed))
                os.makedirs(path_seed, exist_ok=True)
                base_label_name = f"0-{int(label)}"
                base_png_path = os.path.join(path_seed, base_label_name + ".png")
                if not os.path.exists(base_png_path):
                    base_pil.save(base_png_path)
                    digit_info["l2_norm"] = float(np.linalg.norm(np.array(base_pil)))
                    with open(os.path.join(path_seed, base_label_name + ".json"), 'w') as f:
                        json.dump(digit_info, f, sort_keys=True, indent=4)

            # Move on to the next seed
            self.w0_seed += self.step_size

        print(f"Search complete for class {self.class_idx}. Frontier seeds found: {frontier_seed_count}")


def run_mimicry(class_idx, w0_seed=0, step_size=1, save_base_if_no_pairs=False):
    mimicry(
        class_idx=class_idx,
        w0_seed=w0_seed,
        step_size=step_size,
        save_base_if_no_pairs=save_base_if_no_pairs
    ).search()


if __name__ == "__main__":
    for cls_id in range(10):
        run_mimicry(class_idx=cls_id, save_base_if_no_pairs=False)
