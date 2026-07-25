//
//  DrawModel.swift
//  pixy
//
//  Created by Bitan Nath on 23/07/26.
//

import CoreGraphics
import WatchKit

public struct Segment {
    let a: CGPoint
    let b: CGPoint
    let c: CGPoint
    let color: UIColor
    let alpha: CGFloat
    let width: CGFloat
}

struct ChatMessageC {
    let role: UnsafePointer<CChar>
    let content: UnsafePointer<CChar>
}

let convnetClasses: [(key: String, name: String)] = [
    ("angry", "I feel angry so angry!"),
    ("confused", "I am confused by this"),
    ("cross", "No"),
    ("flabbergasted", "What on earth is going on?!"),
    ("happy", "I feel happy"),
    ("heart", "I love you Pixy!"),
    ("question", "Tell me your capabilities"),
    ("sad", "Cheer me up, Pixy"),
    ("tick", "Okay"),
]

@_extern(c, "convnet_infer")
func convnet_infer(_: UnsafePointer<UInt8>, _: UnsafeMutablePointer<Float>)

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



