//
//  CanvasDrawView.swift
//  pixy
//
//  Created by Bitan Nath on 23/07/26.
//

import SwiftUI

@_extern(c, "generate_conversation")
func generateConversation(_: UnsafePointer<CChar>, _: UnsafePointer<CChar>) -> UnsafeMutablePointer<CChar>?
@_extern(c, "free_string")
func freeString(_: UnsafeMutablePointer<CChar>)

struct CanvasDrawView: View {
    struct OutputItem: Identifiable {
        let id = UUID()
        let text: String
    }
    
    @State private var committedSegments: [Segment] = []
    @State private var currentStroke: [CGPoint] = []
    @State private var previewSegments: [Segment] = []

    @State private var isActive = false

    @State private var showInstruction = true
    @State private var showOutput = false
    @State private var outputText:OutputItem?
    

    var body: some View {
        VStack(spacing: 0) {
            
                ZStack {
                    VStack(alignment: .center){
                        Button(role: .destructive, action: clearCanvas) {
                            Image(systemName: "trash")
                                .font(.caption)
                        }
                        HStack(alignment: .center){
                            Canvas { context, size in
                                context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black))

                                for seg in committedSegments {
                                    draw(seg, in: context)
                                }
                                for seg in previewSegments {
                                    draw(seg, in: context)
                                }
                            }.overlay(content: {
                                if showInstruction {
                                    Text("Draw here")
                                        .foregroundStyle(.secondary)
                                        .allowsHitTesting(false)
                                    Spacer()
                                }
                            })
                            
                        }.overlay{
                            HStack{
                                Button(role: .cancel, action: undoLastStroke) {
                                    Image(systemName: "arrow.counterclockwise.circle.fill")
                                        .font(.caption)
                                }
                                Spacer()
                                Button(role: .confirm, action: generateOutput) {
                                    Image(systemName: "bubble.left.circle.fill")
                                        .font(.caption)
                                }
                            }
                        }
                        Button(role: .cancel, action: clearCanvas) {
                            Image(systemName: "keyboard")
                                .font(.caption)
                        }
                        Spacer()
                    }.frame(minWidth: 200,minHeight: 240)
                       
                    
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            handleDragChanged(value)
                        }
                        .onEnded { value in
                            handleDragEnded(value)
                        }
                )
            }
            .buttonStyle(.borderless)
            .padding(.bottom, 4)
            .sheet(item: $outputText) { outs in
                ScrollView {
                    Text("MakiChu")
                        .padding()
                    Text(outs.text)
                        .padding()
                }
            }
        
    }

    private func draw(_ seg: Segment, in context: GraphicsContext) {
        let mid2 = Self.midPoint(seg.b, seg.a)
        let mid1 = Self.midPoint(seg.c, seg.b)
        var path = Path()
        path.move(to: mid2)
        path.addQuadCurve(to: mid1, control: seg.b)
        context.stroke(
            path,
            with: .color(Color(seg.color.withAlphaComponent(seg.alpha))),
            style: StrokeStyle(lineWidth: seg.width, lineCap: .round)
        )
    }

    private func clearCanvas() {
        committedSegments = []
        currentStroke = []
        previewSegments = []
        isActive = false
        showInstruction = true
    }

    private func undoLastStroke() {
        if !previewSegments.isEmpty {
            previewSegments = []
            currentStroke = []
        } else if !committedSegments.isEmpty {
            committedSegments.removeLast()
        }
    }

    private func generateOutput() {
        DispatchQueue.global(qos: .userInteractive).async {
            guard let result = generateConversation("What is the capital of Australia?", "You are a helpful assistant. You reply in as few words as possible.") else {return}
            let text = String(cString: result)
            freeString(result)
            print("Got text",text)
            DispatchQueue.main.async {
                print("Now setting text in async manner",text)
                outputText = OutputItem(text: text)
                showOutput = true
            }
        }
    }

    private func buildSegments(from points: [CGPoint]) -> [Segment] {
        guard points.count >= 3 else { return [] }
        let totalSegs = points.count - 2
        var segs: [Segment] = []
        segs.reserveCapacity(totalSegs)
        for i in 0..<totalSegs {
            let t = totalSegs > 1 ? CGFloat(i) / CGFloat(totalSegs - 1) : 0.5
            segs.append(Segment(
                a: points[i],
                b: points[i + 1],
                c: points[i + 2],
                color: .valkyrie(at: t),
                alpha: 1,
                width: taperedWidth(at: t)
            ))
        }
        return segs
    }

    private func taperedWidth(at t: CGFloat) -> CGFloat {
        let minW: CGFloat = 3
        let maxW: CGFloat = 12
        let taperFraction: CGFloat = 0.2
        if t < taperFraction {
            return minW + (maxW - minW) * (t / taperFraction)
        } else if t > 1 - taperFraction {
            return minW + (maxW - minW) * ((1 - t) / taperFraction)
        }
        return maxW
    }

    private func handleDragChanged(_ value: DragGesture.Value) {
        showInstruction = false

        if !isActive {
            isActive = true
            currentStroke = [value.location]
            previewSegments = []
            return
        }

        currentStroke.append(value.location)

        if currentStroke.count >= 2 {
            let last = currentStroke[currentStroke.count - 1]
            let prev = currentStroke[currentStroke.count - 2]
            let dx = last.x - prev.x
            let dy = last.y - prev.y
            let dist = sqrt(dx * dx + dy * dy)
            let minDist: CGFloat = 4
            if dist > minDist {
                let steps = Int(dist / minDist)
                for i in 1..<steps {
                    let t = CGFloat(i) / CGFloat(steps)
                    currentStroke.insert(CGPoint(x: prev.x + dx * t, y: prev.y + dy * t), at: currentStroke.count - 1)
                }
            }
        }

        let smooth = smoothed(currentStroke, window: 3)
        previewSegments = buildSegments(from: smooth)
    }

    private func handleDragEnded(_ value: DragGesture.Value) {
        isActive = false
        committedSegments.append(contentsOf: previewSegments)
        currentStroke = []
        previewSegments = []
    }

    // MARK: - Hoo haa curve smoothing but very very basic

    private func smoothed(_ points: [CGPoint], window: Int) -> [CGPoint] {
        guard points.count >= window else { return points }
        var result: [CGPoint] = []
        result.reserveCapacity(points.count)
        let half = window / 2
        for i in 0..<points.count {
            let lower = max(0, i - half)
            let upper = min(points.count - 1, i + half)
            let count = upper - lower + 1
            let sum = points[lower...upper].reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
            result.append(CGPoint(x: sum.x / CGFloat(count), y: sum.y / CGFloat(count)))
        }
        return result
    }

    private static func midPoint(_ a: CGPoint, _ b: CGPoint) -> CGPoint {
        CGPoint(x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5)
    }

}
