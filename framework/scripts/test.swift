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

struct ChatMessageC {
    let role: UnsafePointer<CChar>
    let content: UnsafePointer<CChar>
}

@_extern(c, "generate_conversation")
func generateConversation(_: UnsafePointer<CChar>, _: UnsafePointer<CChar>) -> UnsafeMutablePointer<CChar>?

@_extern(c, "generate_turnwise_conversation")
func generateConversationTurnwise(
    _: UnsafeRawPointer,
    _: UInt,
    _: UnsafePointer<CChar>,
    _: Int32,
    _: Double,
    _: UInt
) -> UnsafeMutablePointer<CChar>?

@_extern(c, "free_string")
func freeString(_: UnsafeMutablePointer<CChar>)

let binarySize = getBinarySizeBytes()
let memBefore = getResidentMemoryBytes()

let prompt = "What is the capital of India?"
let systemPrompt = "answer as briefly as possible using as few words as possible"

print("--- Testing generate_conversation ---")

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

print("\n--- Testing generate_turnwise_conversation ---")

let memBefore2 = getResidentMemoryBytes()

"user".withCString { rolePtr in
    "What is the capital of India?".withCString { contentPtr in
        var msg = ChatMessageC(role: rolePtr, content: contentPtr)
        guard let result2 = generateConversationTurnwise(&msg, 1, "answer as briefly as possible using as few words as possible", 512, 0.3, 4096) else {
            print("Error: generate_turnwise_conversation returned null")
            exit(1)
        }
        defer { freeString(result2) }
        let memAfter2 = getResidentMemoryBytes()
        let output2 = String(cString: result2)

        print("Memory (before):   \(memBefore2) bytes (\(memBefore2 / 1048576) MB)")
        print("Memory (after):    \(memAfter2) bytes (\(memAfter2 / 1048576) MB)")
        print("Memory delta:      \(memAfter2 &- memBefore2) bytes (\((memAfter2 &- memBefore2) / 1048576) MB)")
        print("Turnwise Output: \(output2)")
    }
}
