#include <Arduino.h>
#include <JPEGDecoder.h>
#include <ArduTFLite.h>

// Pre-trained and converted TFLite model
#include "model.h"

// Embedded JPEG image
#include "test_can_min_jpg.h"

// Memory area (Tensor Arena)
constexpr int TENSOR_ARENA_SIZE = 5 * 1024;  
static byte tensorArena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

// Model class names
static const char* kCategoryNames[] = {"bottle", "can"};

// Expected input image size (64x64, RGB 3 channels)
constexpr int IMG_WIDTH  = 32;
constexpr int IMG_HEIGHT = 32;
constexpr int IMG_CHANNELS = 3;

// Load JPEG → Convert from 16-bit (RGB565) → float (0..1) × RGB
bool decodeJpegToFloat(const uint8_t* jpegData, size_t jpegLen,
                       float* outImage, int outW, int outH) 
{
  // Decompress the embedded JPEG using JPEGDecoder
  int ret = JpegDec.decodeArray(jpegData, jpegLen);
  if (ret == 0) {
    Serial.println("JPEG decode failed.");
    return false;
  }

  int srcW = JpegDec.width;
  int srcH = JpegDec.height;

  // Nearest neighbor resizing + RGB565 → float (R, G, B)
  for (int y = 0; y < outH; y++) {
    int srcY = map(y, 0, outH, 0, srcH - 1);
    for (int x = 0; x < outW; x++) {
      int srcX = map(x, 0, outW, 0, srcW - 1);

      // pImage is 16-bit (5-6-5)
      uint16_t pixel565 = JpegDec.pImage[srcY * srcW + srcX];
      uint8_t r = (pixel565 & 0xF800) >> 8;  // R5 -> R8
      uint8_t g = (pixel565 & 0x07E0) >> 3;  // G6 -> G8
      uint8_t b = (pixel565 & 0x001F) << 3;  // B5 -> B8

      float rf = r / 255.0f;
      float gf = g / 255.0f;
      float bf = b / 255.0f;

      int dstIdx = (y * outW + x) * IMG_CHANNELS;
      outImage[dstIdx + 0] = rf;
      outImage[dstIdx + 1] = gf;
      outImage[dstIdx + 2] = bf;
    }
  }
  return true;
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

  // Convert JPEG image to float array
  static float inputImage[IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS];
  if (!decodeJpegToFloat(test_can_min_jpg, test_can_min_jpg_len,
                         inputImage, IMG_WIDTH, IMG_HEIGHT)) {
    Serial.println("decodeJpegToFloat failed!");
    while (1) { delay(10); }
  }
  Serial.println("JPEG decode done.");

  // Set input tensor (modelSetInput)
  // Write float array element by element with index specification
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
