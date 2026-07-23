import SwiftUI

struct ChattingView: View {
    @Binding var prompt: String
    @State private var messages: [(String, String)] = []
    @State private var isLoading = false

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(alignment: .leading) {
                    ForEach(messages.indices, id: \.self) { i in
                        Text(messages[i].0 + ": " + messages[i].1)
                            .padding(.vertical, 2)
                    }
                    if isLoading {
                        ProgressView()
                            .padding()
                    }
                }
                .padding()
            }
            Spacer()
            HStack {
                TextField("Ask anything...", text: $prompt).font(.caption)
                Spacer()
                Button(role: .confirm, action: {
                    sendMessage()
                }) {
                    Image(systemName: "arrow.forward.circle.fill")
                        .font(.caption)
                }
                .disabled(prompt.trimmingCharacters(in: .whitespaces).isEmpty || isLoading)
                .buttonStyle(.plain)
                    
            }
            .padding(.horizontal,10)
            .padding(.vertical,2)
            .ignoresSafeArea(.container)
        }
        .frame(maxHeight: .infinity)
    }

    private func sendMessage() {
        let text = prompt
        prompt = ""
        messages.append(("User", text))
        isLoading = true
        DispatchQueue.global(qos: .userInteractive).async {
            guard let result = generateConversation(text, "You are a helpful assistant.") else {
                DispatchQueue.main.async { isLoading = false }
                return
            }
            let out = String(cString: result)
            freeString(result)
            DispatchQueue.main.async {
                messages.append(("Assistant", out))
                isLoading = false
            }
        }
    }
}
