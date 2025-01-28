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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tensorflow as tf
from train import SimpleCNN
from constants import MEAN, STD, CLASSES, MODEL_PATH, TFLITE_PATH, ONNX_PATH, TF_MODEL_PATH, IMG_SIZE

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
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    
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
    try:
        onnx2tf_convert(
            input_onnx_file_path=onnx_model_path,
            output_folder_path=tf_model_path,
            output_signaturedefs=True
        )
        print(f"[ONNX -> TF] Saved TensorFlow model to {tf_model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during ONNX to TF conversion: {e}")
        sys.exit(1)

def convert_tf_to_tflite(tf_model_path="saved_model_tf", tflite_model_path="model.tflite"):
    """Convert TensorFlow SavedModel to quantized TFLite model suitable for Arduino Nano 33."""
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Enable full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Define a representative dataset generator for calibration
    """
    # commented out because Arduino Nano 33 does not have int8 support
    def representative_dataset_gen():
        val_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        val_dataset = datasets.ImageFolder('data/test', transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        for i, (images, _) in enumerate(val_loader):
            # images.shape: (1, 3, IMG_SIZE, IMG_SIZE) -> (1, IMG_SIZE, IMG_SIZE, 3)
            images = images.numpy().transpose(0, 2, 3, 1)  # NCHW to NHWC
            yield [images]
    
    converter.representative_dataset = representative_dataset_gen
    
    # Ensure that if the model has any floating point ops, they are converted to integer
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    """
    
    # Convert the model
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        return
    
    # Save the quantized model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"[TF -> TFLite] Saved quantized TFLite model to {tflite_model_path}")

def validate_conversion_tflite(pytorch_model, tflite_model_path, class_names, test_image_path):
    """Compare the results of the PyTorch model and the quantized TFLite model"""
    transform_for_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform_for_test(image).unsqueeze(0).to(device)  # shape=(1,3,IMG_SIZE,IMG_SIZE)
    
    pytorch_model.eval()
    with torch.no_grad():
        outputs_pt = pytorch_model(input_tensor)
    probs_pt = outputs_pt.squeeze(0).cpu().numpy()
    pt_pred_idx = np.argmax(probs_pt)
    pt_pred_class = class_names[pt_pred_idx]
    pt_pred_conf = probs_pt[pt_pred_idx]
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input data for TFLite (quantized)
    # Assuming the model expects uint8 input
    input_scale, input_zero_point = input_details[0]['quantization']
    tflite_input = input_tensor.cpu().numpy().astype(np.float32)  # (1,3,H,W)
    tflite_input = np.transpose(tflite_input, (0, 2, 3, 1))       # (1,H,W,3)
    
    """
    # commented out because Arduino Nano 33 does not have int8 support
    tflite_input = tflite_input / input_scale + input_zero_point
    tflite_input = np.clip(tflite_input, 0, 255).astype(input_details[0]['dtype'])
    """

    interpreter.set_tensor(input_details[0]['index'], tflite_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # (1, num_classes)

    """
    # commented out because Arduino Nano 33 does not have int8 support
    output_scale, output_zero_point = output_details[0]['quantization']
    probs_tf = (output_data.astype(np.float32) - output_zero_point) * output_scale
    """
    probs_tf = output_data
    tf_pred_idx = np.argmax(probs_tf)
    tf_pred_class = class_names[tf_pred_idx]
    tf_pred_conf = probs_tf[0, tf_pred_idx].item()
    
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
