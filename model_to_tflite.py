"""
- Using a pre-trained .pth file with PyTorch to:
  1) Convert it to ONNX → TensorFlow → TFLite (.tflite)
  2) Compare the inference results of the PyTorch model and the TFLite model with the same input image
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pillow_heif
import onnx
from onnx2tf import convert as onnx2tf_convert
import tensorflow as tf
import torchvision.transforms as transforms
from train import SimpleCNN
from constants import MEAN, STD, CLASSES, MODEL_PATH, TFLITE_PATH, ONNX_PATH, TF_MODEL_PATH

pillow_heif.register_heif_opener()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleCNNWithSoftmax(SimpleCNN):
    def __init__(self, num_classes=2):
        super(SimpleCNNWithSoftmax, self).__init__(num_classes)
    
    def forward(self, x):
        logits = super(SimpleCNNWithSoftmax, self).forward(x)
        probs = F.softmax(logits, dim=1)
        return probs

def export_to_onnx(pytorch_model, onnx_model_path="model.onnx"):
    """Export PyTorch model to ONNX format"""
    pytorch_model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True
    )
    print(f"[Export ONNX] Saved to {onnx_model_path}")

def convert_onnx_to_tf_with_onnx2tf(onnx_model_path="model.onnx", tf_model_path="saved_model_tf"):
    """
    Convert ONNX to TensorFlow SavedModel format using onnx2tf.
    Setting output_signaturedefs=True makes the SavedModel easier to convert to TFLite.
    """
    onnx2tf_convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=tf_model_path,
        output_signaturedefs=True
    )
    print(f"[ONNX -> TF] Saved TensorFlow model to {tf_model_path}")

def convert_tf_to_tflite(tf_model_path="saved_model_tf", tflite_model_path="model.tflite"):
    """Convert TensorFlow SavedModel to TFLite"""
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    # Set quantization or other options here if needed
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"[TF -> TFLite] Saved TFLite model to {tflite_model_path}")


def validate_conversion_tflite(pytorch_model, tflite_model_path, class_names, test_image_path):
    """Compare the results of the PyTorch model and the TFLite model"""
    transform_for_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform_for_test(image).unsqueeze(0).to(device)  # shape=(1,3,224,224)
    
    pytorch_model.eval()
    with torch.no_grad():
        outputs_pt = pytorch_model(input_tensor)
    probs_pt = outputs_pt.squeeze(0).cpu().numpy()
    pt_pred_idx = np.argmax(probs_pt)
    pt_pred_class = class_names[pt_pred_idx]
    pt_pred_conf = probs_pt[pt_pred_idx]
    
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Convert NCHW -> NHWC
    tflite_input = input_tensor.cpu().numpy().astype(np.float32)  # (1,3,224,224)
    tflite_input = np.transpose(tflite_input, (0, 2, 3, 1))       # -> (1,224,224,3)
    
    interpreter.set_tensor(input_details[0]['index'], tflite_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # shape=(1, num_classes)
    
    probs_tf = output_data.squeeze(0)
    tf_pred_idx = np.argmax(probs_tf)
    tf_pred_class = class_names[tf_pred_idx]
    tf_pred_conf = probs_tf[tf_pred_idx]
    
    print("\n===== PyTorch vs. TFLite =====")
    print(f"[PyTorch]  Predicted class: {pt_pred_class}  (Prob = {pt_pred_conf*100:.2f}%)")
    print(f"[TFLite ] Predicted class: {tf_pred_class}  (Prob = {tf_pred_conf*100:.2f}%)")

def main():
    class_names = CLASSES
    num_classes = len(class_names)
    print("Number of classes:", num_classes)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model_pt = SimpleCNNWithSoftmax(num_classes).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model_pt.load_state_dict(state_dict)
    model_pt.eval()
    print("[Load PyTorch] Loaded .pth model.")

    if not os.path.exists(TF_MODEL_PATH):
        os.makedirs(TF_MODEL_PATH)
    
    export_to_onnx(model_pt, ONNX_PATH)
    convert_onnx_to_tf_with_onnx2tf(ONNX_PATH, TF_MODEL_PATH)
    convert_tf_to_tflite(TF_MODEL_PATH, TFLITE_PATH)
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        if os.path.isfile(test_image_path):
            validate_conversion_tflite(model_pt, TFLITE_PATH, class_names, test_image_path)
        else:
            print(f"File not found: {test_image_path}")
    else:
        print("No test image provided for validation.")
        print("Usage: python this_script.py <test_image_path>")

if __name__ == "__main__":
    main()
