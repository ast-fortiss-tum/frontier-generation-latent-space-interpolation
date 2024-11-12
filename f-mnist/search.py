import os
import copy
import json
import numpy as np
import sys
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stylegan.renderer import Renderer
from config import STYLEGAN_INIT, SEARCH_LIMIT, STYLEMIX_SEED_LIMIT, INIT_PKL, STYLEMIX_LAYERS, FRONTIER_PAIRS
from predictor import Predictor
from utils import validate_mutation

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

    def render_state(self, state=None):
        if state is None:
            state = self.state
        result = state['renderer'].render(
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
            INTERPOLATION_ALPHA=state['params'].get('INTERPOLATION_ALPHA', 1.0), 
        )

        info = copy.deepcopy(state['params'])

        return result, info

    def search(self):
        root = f"{FRONTIER_PAIRS}/{self.class_idx}/"

        frontier_seed_count = 0
        tolerance = 1e-10  # For precision

        while frontier_seed_count < self.search_limit:
            state = self.state

            state["params"]["class_idx"] = self.class_idx
            state["params"]["w0_seeds"] = [[self.w0_seed, 1.0]]
            state["params"]["stylemix_idx"] = []
            state["params"]["mixclass_idx"] = None
            state["params"]["stylemix_seed"] = None
            state["params"]["INTERPOLATION_ALPHA"] = 1.0  

            # Start without truncation
            state['params']['trunc_psi'] = 1.0
            state['params']['trunc_cutoff'] = None

            digit, digit_info = self.render_state()

            # Check if 'image' exists in 'digit'
            if 'image' not in digit:
                print(f"Render failed with error: {digit.get('error', 'Unknown error')}")
                self.w0_seed += self.step_size
                continue  

            label = digit_info["class_idx"]
            image = digit['image']
            image = image.crop((2, 2, image.width - 2, image.height - 2))
            image_array = np.array(image)

            accepted, confidence, predictions = Predictor().predict_datapoint(
                np.reshape(image_array, (-1, 28, 28, 1)),
                label
            )

            digit_info["accepted"] = bool(accepted)
            digit_info["exp-confidence"] = float(confidence)
            digit_info["predictions"] = predictions.tolist()

            if accepted:
                # Seed accepted without truncation
                trunc_psi = state['params']['trunc_psi']
                digit_info["trunc_psi"] = trunc_psi
            else:
                # Apply truncation
                truncation_values = np.arange(1.0, 0.09, -0.1)  # From 1.0 to 0.1 with -0.1 steps
                accepted = False
                for trunc_psi in truncation_values:
                    state['params']['trunc_psi'] = trunc_psi
                    digit, digit_info = self.render_state()

                    if 'image' not in digit:
                        print(f"Render failed with error: {digit.get('error', 'Unknown error')} at trunc_psi={trunc_psi}")
                        continue  

                    image = digit['image']
                    image = image.crop((2, 2, image.width - 2, image.height - 2))
                    image_array = np.array(image)

                    accepted, confidence, predictions = Predictor().predict_datapoint(
                        np.reshape(image_array, (-1, 28, 28, 1)),
                        label
                    )

                    digit_info["accepted"] = bool(accepted)
                    digit_info["exp-confidence"] = float(confidence)
                    digit_info["predictions"] = predictions.tolist()

                    if accepted:
                        # Found acceptable seed with truncation
                        digit_info["trunc_psi"] = trunc_psi
                        break

                # If still not accepted, move to next seed
                if not accepted:
                    self.w0_seed += self.step_size
                    continue 

                # Use accepted trunc_psi for renderings
                state['params']['trunc_psi'] = digit_info["trunc_psi"]


            found_at_least_one = False
            _, second_cls = np.argsort(-predictions)[:2]
            second_cls_confidence = predictions[second_cls]
            if second_cls_confidence:
                for stylemix_cls, cls_confidence in enumerate(predictions):
                    if stylemix_cls != label and cls_confidence:
                        found_mutation = False
                        self.stylemix_seed = 0

                        while not found_mutation and self.stylemix_seed < self.stylemix_seed_limit:

                            if self.stylemix_seed == self.w0_seed:
                                self.stylemix_seed += 1
                            state["params"]["stylemix_seed"] = self.stylemix_seed

                            for idx, layer in enumerate(self.layers):
                                state["params"]["stylemix_idx"] = layer

                                # First check if full interpolation triggers the classifier
                                state["params"]["INTERPOLATION_ALPHA"] = 1.0
                                m_digit, m_digit_info = self.render_state()

                                if 'image' not in m_digit:
                                    print(f"Render failed with error: {m_digit.get('error', 'Unknown error')}")
                                    continue  

                                m_image = m_digit['image']
                                m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                                m_image_array = np.array(m_image)

                                m_accepted, confidence, m_predictions = Predictor().predict_datapoint(
                                    np.reshape(m_image_array, (-1, 28, 28, 1)),
                                    label
                                )
                                m_class = np.argmax(m_predictions)

                                if m_class != label:
                                    # Full interpolation causes misclassification
                                    # Proceed to binary search to find optimal alpha
                                    alpha_min = 0.0
                                    alpha_max = 1.0
                                    iteration = 0
                                    max_iterations = 20  

                                    last_correct_image = None
                                    last_correct_alpha = None
                                    last_correct_predictions = None
                                    last_correct_confidence = None

                                    while iteration < max_iterations and (alpha_max - alpha_min) > tolerance:
                                        alpha = (alpha_min + alpha_max) / 2.0
                                        state["params"]["INTERPOLATION_ALPHA"] = alpha

                                        m_digit, m_digit_info = self.render_state()

                                        if 'image' not in m_digit:
                                            print(f"Render failed with error: {m_digit.get('error', 'Unknown error')}")
                                            break 

                                        m_image = m_digit['image']
                                        m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                                        m_image_array = np.array(m_image)

                                        m_accepted, confidence, m_predictions = Predictor().predict_datapoint(
                                            np.reshape(m_image_array, (-1, 28, 28, 1)),
                                            label
                                        )
                                        m_class = np.argmax(m_predictions)
                                        print(f"Iteration {iteration}, Alpha: {alpha:.10f}, Accepted: {m_accepted}, Predicted Class: {m_class}, Confidence: {confidence}")

                                        if m_accepted:
                                            alpha_min = alpha
                                            last_correct_image = m_image_array.copy()
                                            last_correct_alpha = alpha
                                            last_correct_predictions = m_predictions.copy()
                                            last_correct_confidence = confidence
                                        else:
                                            alpha_max = alpha

                                        iteration += 1

                                    
                                    if alpha_max != 1.0:
                                        state["params"]["INTERPOLATION_ALPHA"] = alpha_max
                                        m_digit, m_digit_info = self.render_state()

                                        if 'image' not in m_digit:
                                            print(f"Render failed with error: {m_digit.get('error', 'Unknown error')}")
                                            continue

                                        m_image = m_digit['image']
                                        m_image = m_image.crop((2, 2, m_image.width - 2, m_image.height - 2))
                                        m_image_array = np.array(m_image)

                                        m_accepted, confidence, m_predictions = Predictor().predict_datapoint(
                                            np.reshape(m_image_array, (-1, 28, 28, 1)),
                                            label
                                        )
                                        m_class = np.argmax(m_predictions)

                                        if m_class == stylemix_cls:
                                            valid_mutation, ssi, l2_distance, img_l2, m_img_l2 = validate_mutation(image_array, m_image_array)

                                            if valid_mutation:
                                                if not found_at_least_one:
                                                    frontier_seed_count += 1
                                                    found_at_least_one = True

                                                path = f"{root}{self.w0_seed}/"
                                                seed_name = f"0-{second_cls}"
                                                img_path = f"{path}/{seed_name}.png"
                                                if not os.path.exists(img_path):
                                                    os.makedirs(path, exist_ok=True)
                                                    image.save(img_path)

                                                    digit_info["l2_norm"] = float(img_l2)
                                                    with open(f"{path}/{seed_name}.json", 'w') as f:
                                                        json.dump(digit_info, f, sort_keys=True, indent=4)

                                                found_mutation = True

                                                # Save the last correctly classified image
                                                if last_correct_image is not None:
                                                    correct_img_uint8 = np.clip(last_correct_image, 0, 255).astype(np.uint8)
                                                    correct_pil_image = Image.fromarray(correct_img_uint8)
                                                    correct_img_name = f"{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{last_correct_alpha:.6f}-correct.png"
                                                    correct_img_path = f"{path}/{correct_img_name}"
                                                    correct_pil_image.save(correct_img_path)

                                                    # Save metadata for correct image
                                                    correct_info = m_digit_info.copy()
                                                    correct_info["alpha"] = float(last_correct_alpha)
                                                    correct_info["accepted"] = True
                                                    correct_info["predictions"] = last_correct_predictions.tolist()
                                                    correct_info["exp-confidence"] = float(last_correct_confidence)
                                                    with open(f"{path}/{correct_img_name}.json", 'w') as f:
                                                        json.dump(correct_info, f, sort_keys=True, indent=4)

                                                # Save misclassified image
                                                m_path = f"{path}/{stylemix_cls}"
                                                m_name = f"{int(l2_distance)}-{int(ssi * 100)}-{self.stylemix_seed}-{stylemix_cls}-{layer[0]}-{alpha_max:.6f}-misclassified.png"
                                                os.makedirs(m_path, exist_ok=True)
                                                m_digit_info["accepted"] = False
                                                m_digit_info["predicted-class"] = int(m_class)
                                                m_digit_info["exp-confidence"] = float(confidence)
                                                m_digit_info["predictions"] = m_predictions.tolist()
                                                m_digit_info["ssi"] = float(ssi)
                                                m_digit_info["l2_norm"] = float(m_img_l2)
                                                m_digit_info["l2_distance"] = float(l2_distance)
                                                m_digit_info["alpha"] = float(alpha_max)
                                                with open(f"{m_path}/{m_name}.json", 'w') as f:
                                                    json.dump(m_digit_info, f, sort_keys=True, indent=4)

                                                # Save misclassified image
                                                m_image_uint8 = np.clip(m_image_array, 0, 255).astype(np.uint8)
                                                m_pil_image = Image.fromarray(m_image_uint8)
                                                m_pil_image.save(f"{m_path}/{m_name}.png")

                                                break  
                                            else:
                                                print("Invalid mutation - skipping")
                                        else:
                                            print(f"Misclassification to unexpected class {m_class}, expected {stylemix_cls}")
                                    else:
                                        print("Could not find alpha where acceptance changes from True to False")
                                else:
                                    print("Full interpolation did not cause misclassification; skipping binary search.")
                            if found_mutation:
                                break 
                            self.stylemix_seed += 1
            self.w0_seed += self.step_size

def run_mimicry(class_idx, w0_seed=0, step_size=1):
    mimicry_instance = mimicry(class_idx=class_idx, w0_seed=w0_seed, step_size=step_size)
    mimicry_instance.search()

if __name__ == "__main__":

    #run_mimicry(class_idx=0)
    #run_mimicry(class_idx=1)
    run_mimicry(class_idx=2)
    #run_mimicry(class_idx=3)
    #run_mimicry(class_idx=4)
    #run_mimicry(class_idx=5)
    #run_mimicry(class_idx=6)
    #run_mimicry(class_idx=7)
    #run_mimicry(class_idx=8)
    #run_mimicry(class_idx=9)