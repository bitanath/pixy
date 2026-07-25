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

func getFileSizeBytes(_ path: String) -> Int64 {
    guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
          let size = attrs[.size] as? Int64 else { return 0 }
    return size
}

func elapsedMs(_ start: UInt64) -> Double {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    let elapsed = mach_absolute_time() - start
    return Double(elapsed) * Double(info.numer) / Double(info.denom) / 1_000_000
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

// Determine which model is embedded by checking the binary name or arg
let prompt = "What is the capital of India?"
let systemPrompt = "answer as briefly as possible using as few words as possible"

// --- Single-turn benchmark ---
print("BEGIN_SINGLE")
let t0 = mach_absolute_time()
let memBefore = getResidentMemoryBytes()
guard let result = generateConversation(prompt, systemPrompt) else {
    print("Error: generate_conversation returned null")
    exit(1)
}
let memAfter = getResidentMemoryBytes()
let genTime = elapsedMs(t0)
let output = String(cString: result)
freeString(result)

let binarySize = getBinarySizeBytes()

// Print structured output for parsing
print("BINARY_SIZE:\(binarySize)")
print("SINGLE_TIME_MS:\(String(format: "%.1f", genTime))")
print("SINGLE_MEM_BEFORE:\(memBefore)")
print("SINGLE_MEM_AFTER:\(memAfter)")
print("SINGLE_MEM_DELTA:\(memAfter &- memBefore)")
let singleOutput = output.replacingOccurrences(of: "\n", with: "\\n")
print("SINGLE_OUTPUT:\(singleOutput)")

// --- Turnwise benchmark ---
print("BEGIN_TURNWISE")
let t1 = mach_absolute_time()
let memBefore2 = getResidentMemoryBytes()

"user".withCString { rolePtr in
    prompt.withCString { contentPtr in
        var msg = ChatMessageC(role: rolePtr, content: contentPtr)
        guard let result2 = generateConversationTurnwise(&msg, 1, systemPrompt, 512, 0.3, 4096) else {
            print("Error: generate_turnwise_conversation returned null")
            exit(1)
        }
        let memAfter2 = getResidentMemoryBytes()
        let genTime2 = elapsedMs(t1)
        let output2 = String(cString: result2)
        freeString(result2)

        print("TURN_TIME_MS:\(String(format: "%.1f", genTime2))")
        print("TURN_MEM_BEFORE:\(memBefore2)")
        print("TURN_MEM_AFTER:\(memAfter2)")
        print("TURN_MEM_DELTA:\(memAfter2 &- memBefore2)")
        let turnOutput = output2.replacingOccurrences(of: "\n", with: "\\n")
        print("TURN_OUTPUT:\(turnOutput)")
    }
}

print("END")
