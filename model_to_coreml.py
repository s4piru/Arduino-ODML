import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import coremltools as ct
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from train import SimpleCNN
from constants import MODEL_PATH, COREML_PATH, MEAN, STD, CLASSES, IMG_SIZE 
import pillow_heif

pillow_heif.register_heif_opener()

class SimpleCNNWithSoftmax(SimpleCNN):
    def __init__(self, num_classes=2):
        super(SimpleCNNWithSoftmax, self).__init__(num_classes)
    
    def forward(self, x):
        logits = super(SimpleCNNWithSoftmax, self).forward(x)
        probs = F.softmax(logits, dim=1)
        return probs

def validate_conversion(pytorch_model, coreml_model, class_names, test_image_path):
    """Compare inference results between PyTorch and Core ML models using the same image."""
    
    transform_for_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform_for_test(image)
    input_tensor = input_tensor.unsqueeze(0)
    
    pytorch_model.eval()
    with torch.no_grad():
        outputs_pt = pytorch_model(input_tensor)
    probs_pt = outputs_pt.squeeze(0).numpy()
    pt_pred_idx = np.argmax(probs_pt)
    pt_pred_class = class_names[pt_pred_idx]
    pt_pred_conf = probs_pt[pt_pred_idx]
    
    pt_input_flat = input_tensor.cpu().numpy().ravel()
    print("PyTorch input_tensor[0..4] =", pt_input_flat[:5], "...")
    print("PyTorch input_tensor shape =", input_tensor.shape, "dtype =", input_tensor.dtype)
    
    coreml_input = input_tensor.cpu().numpy().astype(np.float32)
    coreml_out = coreml_model.predict({"input_tensor": coreml_input})
    
    if "classLabel" in coreml_out and "classLabel_probs" in coreml_out:
        cm_pred_class = coreml_out["classLabel"]
        cm_probs_dict = coreml_out["classLabel_probs"]
        cm_pred_conf = cm_probs_dict[cm_pred_class]
    else:
        found_key = [k for k in coreml_out.keys() if k.startswith("var_")]
        if len(found_key) == 1:
            logits_or_probs = coreml_out[found_key[0]]
            probs_cm = logits_or_probs.squeeze(0)
            cm_pred_idx = np.argmax(probs_cm)
            cm_pred_class = class_names[cm_pred_idx]
            cm_pred_conf = probs_cm[cm_pred_idx]
        else:
            raise ValueError(f"Unknown output keys in coreml_out: {list(coreml_out.keys())}")
    
    print("\n===== Validation =====")
    print(f"Test image: {test_image_path}")
    print(f"[PyTorch]  Predicted class: {pt_pred_class}  (Prob = {pt_pred_conf*100:.2f}%)")
    print(f"[Core ML]  Predicted class: {cm_pred_class}  (Prob = {cm_pred_conf*100:.2f}%)")

def main():
    class_names = CLASSES
    num_classes = len(class_names)
    print("Number of classes:", num_classes)
    
    model_pt = SimpleCNNWithSoftmax(num_classes)
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    state_dict = torch.load(model_path, map_location='cpu')
    model_pt.load_state_dict(state_dict)
    model_pt.eval()
    
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    traced_model = torch.jit.trace(model_pt, dummy_input)
    
    classifier_config = ct.ClassifierConfig(
        class_labels=class_names,
        predicted_feature_name="classLabel"
    )
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input_tensor",
                shape=(1, 3, IMG_SIZE, IMG_SIZE),
                dtype=np.float32
            )
        ],
        classifier_config=classifier_config,
        convert_to="mlprogram"
    )
    
    mlmodel.save(COREML_PATH)
    print("Core ML model saved")
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        if os.path.isfile(test_image_path):
            validate_conversion(model_pt, mlmodel, class_names, test_image_path)
        else:
            print(f"File not found: {test_image_path}")
    else:
        print("No test image provided for validation.")

if __name__ == "__main__":
    main()
