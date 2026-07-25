import SwiftUI

struct ExplanationView: View {
    @Binding var isPresented: Bool
    @State private var selected: (emoji: String, name: String, explanation: String)? = nil

    let items: [(emoji: String, name: String, explanation: String)] = [
        ("😐", "Angry", "Draw an angry face to tell Pixy you're mad"),
        ("😖", "Confused", "Draw a confused face when you're not sure"),
        ("❌", "Nope", "Draw a cross to show disagreement"),
        ("😲", "Flabbergasted", "Draw a shocked face to amaze Pixy"),
        ("😊", "Happy", "Draw a happy face to share your joy"),
        ("❤️", "Love", "Draw a heart to tell Pixy you love them"),
        ("❓", "Question", "Draw a question mark to understand capabilities"),
        ("🙁", "Sad", "Draw a sad face when you're down"),
        ("✅", "Okay", "Draw a tick to say yes or confirm"),
    ]

    var body: some View {
        VStack {
            if let sel = selected {
                detailView(sel)
            } else {
                gridView
            }
        }
    }

    private var gridView: some View {
        VStack(spacing: 4) {
            Text("Try drawing these for Pixy")
                .font(.caption2)
                .padding(.top)
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3)) {
                ForEach(items, id: \.name) { item in
                    Button(action: { selected = item }) {
                        VStack(spacing: 2) {
                            Text(item.emoji).font(.title)
                            Text(item.name).font(.system(size: 5))
                        }
                        .padding(2)
                    }
                    .buttonStyle(.plain)
                }
            }
            Spacer()
        }
    }

    private func detailView(_ sel: (emoji: String, name: String, explanation: String)) -> some View {
        VStack(spacing: 12) {
            Spacer()
            Text(sel.emoji).font(.largeTitle).scaleEffect(1.25)
            Text(sel.name).font(.system(size: 16,weight: .semibold, design: .rounded))
            Text(sel.explanation)
                .font(.system(size: 8))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            Spacer()
            Button("Back") { selected = nil }
        }
    }
}
