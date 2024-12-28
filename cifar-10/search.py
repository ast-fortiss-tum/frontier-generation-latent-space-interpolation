import os
import copy
import json
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stylegan.renderer import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL, STYLEMIX_LAYERS, FRONTIER_PAIRS
from predictor import Predictor
from utils import validate_mutation
from PIL import Image

class mimicry:
    def __init__(self, class_idx=None, w0_seed=0, stylemix_seed=0, search_limit=SEARCH_LIMIT, step_size=1):
        self.state = STYLEGAN_INIT
        self.class_idx = class_idx
        self.w0_seed = w0_seed
        self.stylemix_seed = stylemix_seed
        self.search_limit = search_limit
        self.stylemix_seed_limit = STYLEMIX_SEED_LIMIT
        self.layers = STYLEMIX_LAYERS
        self.step_size = step_size
        self.state['renderer'] = Renderer()

    def render_state(self, state=None, alpha=1.0):

        if state is None:
            state = self.state

        state['params']['INTERPOLATION_ALPHA'] = alpha

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
        )

        info = copy.deepcopy(state['params'])
        return state, info

    def search(self):
        """
           - Render the base image.
           - Check if the predicted class is correct (matching `class_idx`). If so do stylemix layer switching.
           - If stylemix at alpha=1.0 triggers misclassification to the second most likely class,
             do a binary-search for interpolation factor to refine the frontier pair.(It makes the search process faster)
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

            digit, digit_info = self.render_state()

            label = digit["params"]["class_idx"]
            image = digit["generator_params"].image
            image_array = np.array(image)

            accepted, confidence, predictions = Predictor().predict_datapoint(
                Image.fromarray(image_array),
                label
            )
            print(
                f"Seed {self.w0_seed}, label={label}, "
                f"Accepted: {accepted}, Confidence: {confidence:.4f}, "
                f"Top Predictions: {[f'{p:.4f}' for p in predictions[:5]]}"
            )

            digit_info["accepted"] = bool(accepted)
            digit_info["exp-confidence"] = float(confidence)
            digit_info["predictions"] = predictions.tolist()
           
            if accepted:
                # Find top-2 classes.
                sorted_classes = np.argsort(-predictions)
                top_cls = sorted_classes[0]
                second_cls = sorted_classes[1]
                second_cls_conf = predictions[second_cls]

                # If second_cls_conf is too tiny, skip
                if second_cls_conf < 0.01:
                    
                    self.w0_seed += self.step_size
                    continue

                print(f"Attempting stylemix with second_cls={second_cls} (conf={second_cls_conf:.4f})")
                stylemix_cls = int(second_cls)

                found_mutation = False

                state["params"]["mixclass_idx"] = stylemix_cls
                self.stylemix_seed = 0

                while not found_mutation and self.stylemix_seed < self.stylemix_seed_limit:
                    # Avoid stylemix_seed == w0_seed
                    if self.stylemix_seed == self.w0_seed:
                        self.stylemix_seed += 1

                    state["params"]["stylemix_seed"] = self.stylemix_seed

                    for layer in self.layers:
                        state["params"]["stylemix_idx"] = layer

                        # Render with alpha=1.0 for these layers
                        m_digit, m_digit_info = self.render_state()

                        m_image = m_digit['generator_params'].image
                        m_image_array = np.array(m_image)

                        m_accepted, m_confidence, m_predictions = Predictor().predict_datapoint(
                            Image.fromarray(m_image_array),
                            label
                        )
                        # Predicted top class
                        m_class = np.argmax(m_predictions)

                        print(
                            f"  stylemix_seed={self.stylemix_seed}, layers={layer}, "
                            f"Accepted:{m_accepted}, Conf={m_confidence:.4f}, "
                            f"PredictedClass={m_class}"
                        )

                        # Stylemix triggers misclassification to stylemix_cls
                        if (not m_accepted) and (m_class == stylemix_cls):
                            
                            alpha_min = 0.0
                            alpha_max = 1.0
                            iteration = 0
                            max_iterations = 15
                            tolerance = 1e-8

                            last_correct_image = None
                            last_correct_alpha = None
                            last_correct_confidence = None
                            last_correct_predictions = None

                            # Binary search
                            while iteration < max_iterations and (alpha_max - alpha_min) > tolerance:
                                alpha = (alpha_min + alpha_max) / 2.0
                                # Render at alpha
                                _, _ = self.render_state(alpha=alpha)

                                candidate_image = state['generator_params'].image
                                candidate_arr = np.array(candidate_image)

                                c_accepted, c_conf, c_preds = Predictor().predict_datapoint(
                                    Image.fromarray(candidate_arr),
                                    label
                                )
                                c_class = np.argmax(c_preds)

                                print(
                                    f"    alpha={alpha:.4f}, "
                                    f"Accepted={c_accepted}, c_class={c_class}"
                                )

                                if c_accepted:
                                    # Still recognized as label, so move alpha_min up
                                    alpha_min = alpha
                                    
                                    last_correct_image = candidate_arr.copy()
                                    last_correct_alpha = alpha
                                    last_correct_confidence = c_conf
                                    last_correct_predictions = c_preds.copy()
                                else:
                                    # Misclassified
                                    alpha_max = alpha

                                iteration += 1

                            _, _ = self.render_state(alpha=alpha_max)
                            final_image = state['generator_params'].image
                            final_arr = np.array(final_image)

                            f_accepted, f_conf, f_preds = Predictor().predict_datapoint(
                                Image.fromarray(final_arr),
                                label
                            )
                            f_class = np.argmax(f_preds)

                            if f_class == stylemix_cls:
                                valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(
                                    image_array,
                                    final_arr
                                )

                                if valid_mutation:
                                    
                                    frontier_seed_count += 1
                                    found_mutation = True

                                    path = f"{root}{self.w0_seed}/"
                                    os.makedirs(path, exist_ok=True)

                                    # Save the original seed image
                                    seed_name = f"0-{second_cls}"
                                    seed_image_path = os.path.join(path, f"{seed_name}.png")
                                    if not os.path.exists(seed_image_path):
                                        image.save(seed_image_path)

                                        digit_info["l2_norm"] = float(np.linalg.norm(image_array))
                                        with open(os.path.join(path, f"{seed_name}.json"), 'w') as f:
                                            json.dump(digit_info, f, sort_keys=True, indent=4)

                                    # Save the final misclassified image
                                    m_digit_info["accepted"] = bool(f_accepted)
                                    m_digit_info["predicted-class"] = int(f_class)
                                    m_digit_info["exp-confidence"] = float(f_conf)
                                    m_digit_info["predictions"] = f_preds.tolist()
                                    m_digit_info["ssi"] = float(ssi)
                                    m_digit_info["l2_norm"] = float(m_img_l2)
                                    m_digit_info["l2_distance"] = float(l2_distance)
                                    m_digit_info["alpha"] = float(alpha_max)

                                    stylemix_path = os.path.join(path, str(stylemix_cls))
                                    os.makedirs(stylemix_path, exist_ok=True)

                                    # filename referencing alpha, stylemix_seed and layer
                                    mis_name = (
                                        f"{int(l2_distance)}-"
                                        f"{int(ssi*100)}-"
                                        f"{self.stylemix_seed}-"
                                        f"{stylemix_cls}-"
                                        f"{layer[0]}-"
                                        f"{alpha_max:.6f}-mis.png"
                                    )

                                    with open(os.path.join(stylemix_path, mis_name + '.json'), 'w') as f:
                                        json.dump(m_digit_info, f, sort_keys=True, indent=4)

                                    Image.fromarray(final_arr).save(os.path.join(stylemix_path, mis_name + '.png'))

                                    
                                    # step before misclassification:
                                    if last_correct_image is not None:
                                        correct_img_uint8 = np.clip(last_correct_image, 0, 255).astype(np.uint8)
                                        correct_pil = Image.fromarray(correct_img_uint8)

                                        correct_name = (
                                            f"{int(l2_distance)}-"
                                            f"{int(ssi*100)}-"
                                            f"{self.stylemix_seed}-"
                                            f"{stylemix_cls}-"
                                            f"{layer[0]}-"
                                            f"{last_correct_alpha:.6f}-correct.png"
                                        )
                                        correct_path = os.path.join(path, correct_name)
                                        correct_pil.save(correct_path)

                                        # JSON for correct image
                                        correct_info = m_digit_info.copy()
                                        correct_info["accepted"] = True
                                        correct_info["predicted-class"] = label
                                        correct_info["exp-confidence"] = float(last_correct_confidence)
                                        correct_info["predictions"] = last_correct_predictions.tolist()
                                        correct_info["alpha"] = float(last_correct_alpha)

                                        with open(correct_path + '.json', 'w') as f:
                                            json.dump(correct_info, f, sort_keys=True, indent=4)

                                    break  # from stylemix_idx
                                else:
                                    print("Invalid mutation - skipping")
                            else:
                                print("Refined boundary did not misclassify to stylemix_cls. Skipping...")

                        if found_mutation:
                            break  # from self.layers

                    if found_mutation:
                        break  # from stylemix_seed

                    self.stylemix_seed += 1

            self.w0_seed += self.step_size

            # If found enough frontier pairs
            if frontier_seed_count >= self.search_limit:
                break

def run_mimicry(class_idx, w0_seed=0, step_size=1):
    mimicry_instance = mimicry(class_idx=class_idx, w0_seed=w0_seed, step_size=step_size)
    mimicry_instance.search()

if __name__ == "__main__":

    #run_mimicry(class_idx=0)
    run_mimicry(class_idx=1)
    #run_mimicry(class_idx=2)
    #run_mimicry(class_idx=3)
    #run_mimicry(class_idx=4)
    #run_mimicry(class_idx=5)
    #run_mimicry(class_idx=6)
    #run_mimicry(class_idx=7)
    #run_mimicry(class_idx=8)
    #run_mimicry(class_idx=9)
