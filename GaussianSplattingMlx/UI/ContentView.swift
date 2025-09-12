//
//  ContentView.swift
//  GaussianSplattingMlX
//
//  Created by Tatsuya Ogawa on 2025/05/28.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        MainNavigationView()
    }
}

// MARK: - Main Navigation View

struct MainNavigationView: View {
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Data Collection Tab
            ARKitCaptureView()
                .tabItem {
                    Image(systemName: "camera.fill")
                    Text("Capture")
                }
                .tag(0)
            
            // Training Tab
            TrainView()
                .tabItem {
                    Image(systemName: "brain.head.profile")
                    Text("Training")
                }
                .tag(1)
        }
        .accentColor(.blue)
    }
}

#Preview {
    ContentView()
}
