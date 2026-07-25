import SwiftUI

struct ChattingView: View {
    @Binding var prompt: String
    @Binding var messages: [(String, String)]
    @Binding var autoSend: Bool
    @State private var isLoading = false
    
    // MARK: - let's have these params purely as hardcoded vals for now
    private let maxTokens: Int32 = 512
    private let temperature: Double = 0.5
    private let contextSize: UInt = 2048

    private var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
    }

    private var osVersion: String {
        WKInterfaceDevice.current().systemVersion
    }
    // MARK: - Now for the actual view

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading) {
                        HStack {
                            Text("Pixy \(appVersion) • watchOS \(osVersion)")
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(Color.gray.opacity(0.3))
                                .foregroundColor(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 16))
                            Spacer()
                        }
                        .padding(.horizontal, 4)
                        .padding(.vertical, 2)
                        ForEach(messages.indices, id: \.self) { i in
                            let isUser = messages[i].0.lowercased() == "user"
                            HStack {
                                if isUser { Spacer() }
                                Text(messages[i].1)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(isUser ? Color.blue : Color.gray.opacity(0.3))
                                    .foregroundColor(.white)
                                    .clipShape(RoundedRectangle(cornerRadius: 16))
                                if !isUser { Spacer() }
                            }
                            .padding(.horizontal, 4)
                            .padding(.vertical, 2)
                        }
                        if isLoading {
                            ProgressView()
                                .padding()
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding()
                }
                .onAppear {
                    proxy.scrollTo("bottom", anchor: .bottom)
                    if autoSend {
                        autoSend = false
                        turnwiseMessaging()
                    }
                }
                .onChange(of: messages.count) { _, _ in
                    withAnimation {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }
            }
            Spacer()
            HStack {
                TextField("Ask anything...", text: $prompt).font(.caption)
                Spacer()
                Button(role: .confirm, action: {
turnwiseMessaging()
                }) {
                    Image(systemName: "arrow.forward.circle.fill")
                        .font(.caption)
                }
                .disabled(prompt.trimmingCharacters(in: .whitespaces).isEmpty || isLoading)
                .buttonStyle(.plain)
                    
            }
            .padding(.horizontal,20)
            .padding(.bottom,6)
            
        }
        .ignoresSafeArea(.container)
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

    private func turnwiseMessaging() {
        let text = prompt
        prompt = ""
        messages.append(("User", text))
        isLoading = true
        DispatchQueue.global(qos: .userInteractive).async {
            
            let recentMessages = self.messages.suffix(5)
            let contextMessages = recentMessages.first?.0 == "User" ? recentMessages : recentMessages.dropFirst()
            
            
            
            var cPointers: [UnsafeMutablePointer<CChar>] = []
            for (role, content) in contextMessages {
                cPointers.append(strdup(role))
                cPointers.append(strdup(content))
            }
            defer { for p in cPointers { free(p) } }

            var cMsgs: [ChatMessageC] = []
            for i in stride(from: 0, to: cPointers.count, by: 2) {
                cMsgs.append(.init(role: cPointers[i], content: cPointers[i + 1]))
            }

            let device = WKInterfaceDevice.current()
            let df = DateFormatter()
            df.locale = Locale(identifier: "en_US_POSIX")
            df.dateFormat = "EEEE, MMMM d h:mm a"
            let now = df.string(from: Date())
            let system = "Your name is Pixy and you are a friendly, helpful and succinct assistant. Reply in as few words as possible. \n ## CONTEXT: \n The current date and time is \(now). You are running on \(device.model) with \(device.systemName) \(device.systemVersion). \n ## CRITICAL: \n Use emoji wherever possible. Respond to the user's mood, if the user is feeling sad or angry tell a joke. If the user is feeling confused or flabbergasted try to clarify the earlier chat messages. If the user asks for your capabilities reply very briefly. Do not reply pointwise, instead write prose. Do not try to write code, or call tools."
            print("Got messages",contextMessages,system)
            guard let result = generateConversationTurnwise(&cMsgs, UInt(cMsgs.count), system, maxTokens, temperature, contextSize) else {
                DispatchQueue.main.async { self.isLoading = false }
                return
            }
            let out = String(cString: result)
            freeString(result)
            DispatchQueue.main.async {
                self.messages.append(("Assistant", out))
                self.isLoading = false
            }
        }
    }
}
