import Foundation
import CoreGraphics
import ImageIO
import Darwin

func getResidentMemoryBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rp in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rp, &count)
        }
    }
    guard result == KERN_SUCCESS else { return 0 }
    return info.resident_size
}

func getTimeUs() -> UInt64 {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let now = mach_absolute_time()
    let nanos = now * UInt64(info.numer) / UInt64(info.denom)
    return nanos / 1_000
}

func getBinarySizeBytes() -> Int64 {
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: CommandLine.arguments[0]),
          let size = attrs[.size] as? Int64 else { return 0 }
    return size
}

@_extern(c, "convnet_infer")
func convnet_infer(_: UnsafePointer<UInt8>, _: UnsafeMutablePointer<Float>)

let classes = ["angry", "confused", "cross", "flabbergasted", "happy", "heart", "question", "sad", "tick"]

func findRepoRoot() -> String {
    var dir = FileManager.default.currentDirectoryPath
    while true {
        let trainingPath = (dir as NSString).appendingPathComponent("training")
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: trainingPath, isDirectory: &isDir), isDir.boolValue {
            return dir
        }
        let parent = (dir as NSString).deletingLastPathComponent
        if parent == dir { return FileManager.default.currentDirectoryPath }
        dir = parent
    }
}

func loadAndPreprocessImage(path: String) -> [UInt8]? {
    let optsNil: CFDictionary? = nil
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, optsNil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, optsNil) else {
        return nil
    }
    let width = 56
    let height = 56
    let colorSpace = CGColorSpaceCreateDeviceGray()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)

    guard let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width,
        space: colorSpace,
        bitmapInfo: bitmapInfo.rawValue
    ) else { return nil }

    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    guard let pixelData = context.data else { return nil }
    let ptr = pixelData.assumingMemoryBound(to: UInt8.self)
    return Array(UnsafeBufferPointer(start: ptr, count: width * height))
}

// --- Main ---

let repoRoot = findRepoRoot()
let datasetDir = (repoRoot as NSString).appendingPathComponent("training/dataset")

let binarySize = getBinarySizeBytes()
print("=== ConvNet Benchmark ===")
print("Binary size: \(binarySize) bytes (\(binarySize / 1024) KB)  (includes 1.27 MB embedded weights)")
print("Classes: \(classes.joined(separator: ", "))")
print()

var testPaths: [(cls: String, path: String)] = []
for cls in classes {
    let clsDir = (datasetDir as NSString).appendingPathComponent(cls)
    guard let files = try? FileManager.default.contentsOfDirectory(atPath: clsDir) else {
        print("Warning: cannot read \(clsDir)")
        continue
    }
    let pngs = files.filter { $0.hasSuffix(".png") }.sorted()
    guard let first = pngs.first else {
        print("Warning: no PNGs in \(clsDir)")
        continue
    }
    testPaths.append((cls, (clsDir as NSString).appendingPathComponent(first)))
}

print("Found \(testPaths.count) test images")
print()

// Warmup — loads image, runs inference once to settle caches
if let first = testPaths.first, let pixels = loadAndPreprocessImage(path: first.path) {
    var dummy: [Float] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    pixels.withUnsafeBufferPointer { buf in
        convnet_infer(buf.baseAddress!, &dummy)
    }
    print("Warmup done")
    print()
}

let memBeforeAll = getResidentMemoryBytes()

print("  \(pad("Image", 16)) \(pad("Predicted", 16)) \(pad("Time(µs)", 10))  \(pad("MemΔ(bytes)", 14))")

var totalTime: UInt64 = 0
var maxMemDelta: Int64 = 0

for info in testPaths {
    guard let pixels = loadAndPreprocessImage(path: info.path) else {
        print("  \(info.cls): failed to load image")
        continue
    }

    var logits: [Float] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    let memBefore = getResidentMemoryBytes()
    let t0 = getTimeUs()

    pixels.withUnsafeBufferPointer { buf in
        convnet_infer(buf.baseAddress!, &logits)
    }

    let t1 = getTimeUs()
    let memAfter = getResidentMemoryBytes()

    let elapsed = t1 &- t0
    let memDelta = Int64(memAfter) - Int64(memBefore)

    totalTime += elapsed
    if memDelta > maxMemDelta { maxMemDelta = memDelta }

    let predictedIdx = logits.firstIndex(of: logits.max()!)!
    let predicted = classes[predictedIdx]
    let match = predicted == info.cls ? "✓" : "✗"

    print("  \(pad(info.cls, 16)) \(pad(predicted, 16)) \(padNum(elapsed, 10))  \(padNum(Int64(memDelta), 14))  \(match)")
}

func pad(_ s: String, _ n: Int) -> String {
    if s.count >= n { return String(s.prefix(n)) }
    return s + String(repeating: " ", count: n - s.count)
}

func padNum(_ v: UInt64, _ n: Int) -> String {
    let s = "\(v)"
    return pad(s, n)
}

func padNum(_ v: Int64, _ n: Int) -> String {
    let s = "\(v)"
    return pad(s, n)
}

let memAfterAll = getResidentMemoryBytes()
let totalMemDelta = Int64(memAfterAll) - Int64(memBeforeAll)
let avgTime = Double(totalTime) / Double(testPaths.count)

print()
print("=== Summary ===")
print("  Total time (9 inferences): \(totalTime) µs")
print("  Average time per inference: \(String(format: "%.1f", avgTime)) µs")
print("  Total memory delta: \(totalMemDelta) bytes (\(totalMemDelta / 1024) KB)")
print("  Max per-inference memory delta: \(maxMemDelta) bytes (\(maxMemDelta / 1024) KB)")
