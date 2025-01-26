import tensorflow as tf
from constants import TFLITE_PATH

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

print("===== Input Tensor Details =====")
for detail in input_details:
    print(detail)

output_details = interpreter.get_output_details()
print("\n===== Output Tensor Details =====")
for detail in output_details:
    print(detail)
