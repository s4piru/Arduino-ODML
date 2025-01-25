import SwiftUI
import CoreML
import PhotosUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var classificationLabel: String = "Classification results will be displayed here"
    @State private var isLoading: Bool = false
    @State private var dumpURL: URL? = nil
    let mean: [Float] = [0.485, 0.456, 0.406]
    let std:  [Float] = [0.229, 0.224, 0.225]
    let classNames = ["bottle", "can"]
    let model: bottle_can_simple_cnn_mps? = {
        do {
            return try bottle_can_simple_cnn_mps(configuration: MLModelConfiguration())
        } catch {
            print("Failed to load model: \(error)")
            return nil
        }
    }()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                } else {
                    Rectangle()
                        .fill(Color.secondary)
                        .frame(height: 300)
                        .overlay(Text("Select an image").foregroundColor(.white))
                }
                
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    Text("Select Image")
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .padding(.horizontal)
                .onChange(of: selectedItem) { oldItem, newItem in
                    Task {
                        if let newItem = newItem {
                            await loadImage(from: newItem)
                        }
                    }
                }
                
                if isLoading {
                    ProgressView("Classifying...")
                }
                
                Text(classificationLabel)
                    .multilineTextAlignment(.center)
                    .padding()
                
                if let url = dumpURL {
                    Text("Pixel data saved to: \(url.lastPathComponent)")
                        .foregroundColor(.blue)
                        .onTapGesture {
                            let activityVC = UIActivityViewController(activityItems: [url], applicationActivities: nil)
                            UIApplication.shared.windows.first?.rootViewController?
                                .present(activityVC, animated: true, completion: nil)
                        }
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("Bottle or Can")
        }
    }
    
    func loadImage(from item: PhotosPickerItem) async {
        self.isLoading = true
        self.classificationLabel = "Classifying..."
        
        if let data = try? await item.loadTransferable(type: Data.self),
           let uiImage = UIImage(data: data) {
            self.selectedImage = uiImage
            classifyAndDumpImage(uiImage)
        } else {
            self.classificationLabel = "Failed to load image"
            self.isLoading = false
        }
    }
    
    func classifyAndDumpImage(_ uiImage: UIImage) {
        let maxSize: CGFloat = 512
        let resizedImage = uiImage.resized(toMaxSide: maxSize)
        guard let resized = resizedImage else {
            self.classificationLabel = "Failed to resize image"
            self.isLoading = false
            return
        }
        
        let cropSize = CGSize(width: 224, height: 224)
        guard let croppedImage = resized.centerCrop(to: cropSize) else {
            self.classificationLabel = "Failed to crop image"
            self.isLoading = false
            return
        }
        
        guard let cgImage = croppedImage.cgImage else {
            self.classificationLabel = "Failed to get CGImage"
            self.isLoading = false
            return
        }
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            self.classificationLabel = "Failed to create sRGB color space"
            self.isLoading = false
            return
        }
        
        guard let context = CGContext(data: nil,
                                      width: 224,
                                      height: 224,
                                      bitsPerComponent: 8,
                                      bytesPerRow: 224 * 4,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            self.classificationLabel = "Failed to create CGContext"
            self.isLoading = false
            return
        }
        
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: 224, height: 224))
        
        guard let resizedData = context.data else {
            self.classificationLabel = "Failed to get context data"
            self.isLoading = false
            return
        }
        
        let shape: [NSNumber] = [1, 3, 224, 224]
        guard let multiArray = try? MLMultiArray(shape: shape, dataType: .float32) else {
            self.classificationLabel = "Failed to create MLMultiArray"
            self.isLoading = false
            return
        }
        
        var csvString = ""
        
        func putValue(_ batch: Int, _ channel: Int, _ y: Int, _ x: Int, _ val: Float) {
            let idx = [NSNumber(value: batch),
                       NSNumber(value: channel),
                       NSNumber(value: y),
                       NSNumber(value: x)]
            multiArray[idx] = NSNumber(value: val)
        }
        
        for y in 0..<224 {
            for x in 0..<224 {
                let offset = (y * 224 + x) * 4
                let rVal = resizedData.load(fromByteOffset: offset+0, as: UInt8.self)
                let gVal = resizedData.load(fromByteOffset: offset+1, as: UInt8.self)
                let bVal = resizedData.load(fromByteOffset: offset+2, as: UInt8.self)
                
                let rFloat = Float(rVal) / 255.0
                let gFloat = Float(gVal) / 255.0
                let bFloat = Float(bVal) / 255.0
                
                let rNorm = (rFloat - mean[0]) / std[0]
                let gNorm = (gFloat - mean[1]) / std[1]
                let bNorm = (bFloat - mean[2]) / std[2]
                
                putValue(0, 0, y, x, rNorm) // R
                putValue(0, 1, y, x, gNorm) // G
                putValue(0, 2, y, x, bNorm) // B
                
                csvString += "\(rNorm),\(gNorm),\(bNorm)\n"
            }
        }
        
        let fileName = "pixel_dump.csv"
        if let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = docDir.appendingPathComponent(fileName)
            do {
                try csvString.write(to: fileURL, atomically: true, encoding: .utf8)
                self.dumpURL = fileURL
                print("Pixel data saved to \(fileURL)")
            } catch {
                print("Failed to save pixel data: \(error)")
            }
        }
        
        guard let model = self.model else {
            self.classificationLabel = "Failed to load CoreML model"
            self.isLoading = false
            return
        }
        
        do {
            let output = try model.prediction(input_tensor: multiArray)
            let predClass = output.classLabel
            let probs = output.classLabel_probs
            
            print("Core ML Predicted classLabel: \(predClass)")
            
            if let confidence = probs[predClass] {
                let confidencePct = confidence * 100.0
                DispatchQueue.main.async {
                    self.isLoading = false
                    let className = predClass == "bottle" ? "Bottle" : "Can"
                    self.classificationLabel = "\(className) (\(String(format: "%.2f", confidencePct))%)"
                }
            } else {
                DispatchQueue.main.async {
                    self.isLoading = false
                    self.classificationLabel = "\(predClass) (?)"
                }
            }
        } catch {
            self.classificationLabel = "Model prediction failed: \(error.localizedDescription)"
            self.isLoading = false
        }
    }
}

extension UIImage {
    func resized(toMaxSide maxSide: CGFloat) -> UIImage? {
        let aspectRatio = self.size.width / self.size.height
        var newSize: CGSize
        if aspectRatio > 1 {
            newSize = CGSize(width: maxSide, height: maxSide / aspectRatio)
        } else {
            newSize = CGSize(width: maxSide * aspectRatio, height: maxSide)
        }
        
        UIGraphicsBeginImageContextWithOptions(newSize, false, self.scale)
        self.draw(in: CGRect(origin: .zero, size: newSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func centerCrop(to targetSize: CGSize) -> UIImage? {
        guard let cgImage = self.cgImage else { return nil }
        
        let width = CGFloat(cgImage.width)
        let height = CGFloat(cgImage.height)
        
        let cropRect = CGRect(
            x: max(0, (width - targetSize.width) / 2),
            y: max(0, (height - targetSize.height) / 2),
            width: targetSize.width,
            height: targetSize.height
        )
        
        guard let croppedCGImage = cgImage.cropping(to: cropRect) else { return nil }
        return UIImage(cgImage: croppedCGImage, scale: self.scale, orientation: self.imageOrientation)
    }
}
