import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pillow_heif
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import torchvision.transforms as transforms
from constants import MODEL_PATH, MEAN, STD, CLASSES
from train import SimpleCNN

pillow_heif.register_heif_opener()

class SimpleCNNWithSoftmax(SimpleCNN):
    """Extended SimpleCNN that applies softmax to the logits."""
    def __init__(self, num_classes=2):
        super(SimpleCNNWithSoftmax, self).__init__(num_classes)
    
    def forward(self, x):
        logits = super(SimpleCNNWithSoftmax, self).forward(x)
        probs = F.softmax(logits, dim=1)
        return probs

def export_to_onnx(pytorch_model, onnx_model_path="model.onnx"):
    """Export PyTorch model to ONNX format."""
    pytorch_model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
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

def convert_onnx_to_tf(onnx_model_path="model.onnx", tf_model_path="model_tf"):
    """Convert ONNX model to TensorFlow SavedModel using onnx-tf."""
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)
    print(f"[ONNX -> TF] Saved TensorFlow model to {tf_model_path}")

def convert_tf_to_tflite(tf_model_path="model_tf", tflite_model_path="model.tflite"):
    """Convert TensorFlow SavedModel to TFLite format."""
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Example of quantization or optimization
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"[TF -> TFLite] Saved TFLite model to {tflite_model_path}")

def validate_conversion_tflite(pytorch_model, tflite_model_path, class_names, test_image_path):
    """Compare inference results between PyTorch and TFLite models using the same image."""
    
    # Prepare input
    transform_for_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform_for_test(image).unsqueeze(0)  # shape=(1, 3, 224, 224)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        outputs_pt = pytorch_model(input_tensor)
    probs_pt = outputs_pt.squeeze(0).numpy()  # shape=(num_classes,)
    pt_pred_idx = np.argmax(probs_pt)
    pt_pred_class = class_names[pt_pred_idx]
    pt_pred_conf = probs_pt[pt_pred_idx]
    
    # TFLite inference
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # TFLite expects NHWC by default if it was converted from TensorFlow.
    # But if the model is exported in NCHW form, we might need to transpose or verify shapes.
    # For demonstration, let's assume it's expecting the same shape (1, 3, 224, 224).
    
    # Convert PyTorch tensor to numpy float32
    tflite_input = input_tensor.cpu().numpy().astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], tflite_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # shape could be (1, num_classes)
    
    # If the output is logits or probabilities depends on how it was converted
    probs_tf = output_data.squeeze(0)  # shape=(num_classes,)
    tf_pred_idx = np.argmax(probs_tf)
    tf_pred_class = class_names[tf_pred_idx]
    tf_pred_conf = probs_tf[tf_pred_idx]
    
    print("\n===== PyTorch vs. TFLite =====")
    print(f"[PyTorch]  Predicted class: {pt_pred_class}  (Prob = {pt_pred_conf*100:.2f}%)")
    print(f"[TFLite ] Predicted class: {tf_pred_class}  (Prob = {tf_pred_conf*100:.2f}%)")

def main():
    # Load PyTorch model
    class_names = CLASSES
    num_classes = len(class_names)
    print("Number of classes:", num_classes)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model_pt = SimpleCNNWithSoftmax(num_classes)
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model_pt.load_state_dict(state_dict)
    model_pt.eval()
    print("[Load PyTorch] Loaded .pth model.")

    # Export to ONNX, TF, TFLite
    onnx_model_path = "model.onnx"
    tf_model_dir = "model_tf"
    tflite_model_path = "model.tflite"
    export_to_onnx(model_pt, onnx_model_path)
    convert_onnx_to_tf(onnx_model_path, tf_model_dir)
    convert_tf_to_tflite(tf_model_dir, tflite_model_path)
    
    # If user provides an image path via command line, do comparison
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        if os.path.isfile(test_image_path):
            validate_conversion_tflite(model_pt, tflite_model_path, class_names, test_image_path)
        else:
            print(f"File not found: {test_image_path}")
    else:
        print("No test image provided for validation.\nUsage: python this_script.py <test_image_path>")

if __name__ == "__main__":
    main()