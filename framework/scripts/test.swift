import Foundation
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

func getBinarySizeBytes() -> Int64 {
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: CommandLine.arguments[0]),
          let size = attrs[.size] as? Int64 else { return 0 }
    return size
}

@_extern(c, "generate_conversation")
func generateConversation(_: UnsafePointer<CChar>, _: UnsafePointer<CChar>) -> UnsafeMutablePointer<CChar>?

@_extern(c, "free_string")
func freeString(_: UnsafeMutablePointer<CChar>)

let binarySize = getBinarySizeBytes()
let memBefore = getResidentMemoryBytes()

let prompt = "What is the capital of India?"
let systemPrompt = "You are a helpful assistant."

guard let result = generateConversation(prompt, systemPrompt) else {
    print("Error: generate_conversation returned null")
    exit(1)
}
defer { freeString(result) }

let memAfter = getResidentMemoryBytes()
let output = String(cString: result)

print("Binary size:       \(binarySize) bytes (\(binarySize / 1048576) MB)")
print("Memory (before):   \(memBefore) bytes (\(memBefore / 1048576) MB)")
print("Memory (after):    \(memAfter) bytes (\(memAfter / 1048576) MB)")
print("Memory delta:      \(memAfter &- memBefore) bytes (\((memAfter &- memBefore) / 1048576) MB)")
print("Output: \(output)")
