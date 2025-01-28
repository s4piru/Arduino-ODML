#include <ArduTFLite.h>
#include "raw_image.h"

// Pre-trained and converted TFLite model
#include "model.h"

// Memory area (Tensor Arena)
constexpr int TENSOR_ARENA_SIZE = 128 * 1024; 
static byte tensorArena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

// Model class names
static const char* kCategoryNames[] = {"bottle", "can"};

// Expected input image size (16x16, RGB 3 channels)
constexpr int IMG_WIDTH  = 16;
constexpr int IMG_HEIGHT = 16;
constexpr int IMG_CHANNELS = 3;

// RAW data to float
void loadRawToFloat(float* outImage, int outW, int outH) {
  for (int y = 0; y < outH; y++) {
    for (int x = 0; x < outW; x++) {
      for (int c = 0; c < IMG_CHANNELS; c++) {
        int idx = (y * outW + x) * IMG_CHANNELS + c;
        uint8_t pixel = pgm_read_byte(&rawImage[idx]);
        outImage[idx] = pixel / 255.0f;  // Normalize 0.0 - 1.0
      }
    }
  }
}

void setup() {
  Serial.begin(9600);
  delay(1000);
  while (!Serial) {}

  Serial.println("===== Minimal ArduTFLite Image Classification =====");

  // Initialize TensorFlow Lite Micro
  if (!modelInit(bottle_can_simple_cnn_mps_tflite, tensorArena, TENSOR_ARENA_SIZE)) {
    Serial.println("modelInit() failed - Check TENSOR_ARENA_SIZE or model format");
    while (1) { delay(10); }
  }
  Serial.println("modelInit() success!");

  // Convert RAW image to float array
  static float inputImage[IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS];
  loadRawToFloat(inputImage, IMG_WIDTH, IMG_HEIGHT);
  Serial.println("RAW image loaded and converted.");

  // Set input tensor
  const int inputCount = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;
  for (int i = 0; i < inputCount; i++) {
    if (!modelSetInput(inputImage[i], i)) {
      Serial.print("modelSetInput failed at index=");
      Serial.println(i);
      while (1) { delay(10); }
    }
  }
  Serial.println("All input pixels are set.");

  // Run inference
  if (!modelRunInference()) {
    Serial.println("modelRunInference() failed!");
    while (1) { delay(10); }
  }
  Serial.println("Inference done.");

  // Get output
  float score0 = modelGetOutput(0);
  float score1 = modelGetOutput(1);

  Serial.println("Inference result:");
  Serial.print(" Class [0] ");
  Serial.print(kCategoryNames[0]);
  Serial.print(" = ");
  Serial.println(score0, 4);

  Serial.print(" Class [1] ");
  Serial.print(kCategoryNames[1]);
  Serial.print(" = ");
  Serial.println(score1, 4);

  int predicted_idx = (score0 > score1) ? 0 : 1;
  Serial.print("Predicted Class: ");
  Serial.println(kCategoryNames[predicted_idx]);

  Serial.println("===== Setup Done =====");
}

void loop() {
  // Do Nothing
}
