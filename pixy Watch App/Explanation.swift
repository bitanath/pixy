import SwiftUI

struct ExplanationView: View {
    @Binding var isPresented: Bool
    @State private var selected: (key: String, name: String, emoji: String, item: String)? = nil

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
            HStack{
                Spacer()
                Text("Draw these for Pixy")
                    .font(.caption2)
                    .padding(.top,5)
                Spacer()
            }
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3)) {
                ForEach(convnetClasses, id: \.key) { entry in
                    Button(action: { selected = entry }) {
                        VStack(spacing: 2) {
                            Text(entry.emoji).font(.title)
                            Text(entry.item).font(.system(size: 5))
                        }
                        .padding(2)
                    }
                    .buttonStyle(.plain)
                }
            }
            Spacer()
        }
    }

    private func detailView(_ sel: (key: String, name: String, emoji: String, item: String)) -> some View {
        VStack(spacing: 12) {
            Spacer()
            Text(sel.emoji).font(.largeTitle).scaleEffect(1.25)
            Text(sel.item).font(.system(size: 16,weight: .semibold, design: .rounded))
            Text(sel.name)
                .font(.system(size: 8))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            Spacer()
            Button("Back") { selected = nil }
        }
    }
}
