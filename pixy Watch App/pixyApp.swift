//
//  pixyApp.swift
//  pixy Watch App
//
//  Created by Bitan Nath on 23/07/26.
//

import SwiftUI

@main
struct pixy_Watch_AppApp: App {
    
    init() {
        monitorMemoryPressure()
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
    
    private func monitorMemoryPressure() {
            let source = DispatchSource.makeMemoryPressureSource(
                eventMask: [.warning, .critical],
                queue: .main
            )
            
            source.setEventHandler {
                let event = source.data
                if event.contains(.critical) {
                    print("CRITICAL watchOS memory pressure!")
                } else if event.contains(.warning) {
                    print("⚠️ WARNING watchOS memory pressure!")
                }
            }
            
            source.activate()
        }
}
