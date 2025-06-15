import SwiftUI
import UniformTypeIdentifiers

struct SelectedDataSet: Equatable {
    var format: FileFormat
    var url: URL?
}
enum DemoKind: String, CaseIterable, Identifiable {
    case chair = "Chair"
    case lego = "Lego"
    var id: String { self.rawValue }
}

enum FileFormat: Equatable, Identifiable, Hashable {
    case colmap
    case nerfstudio
    case demo(DemoKind)

    var id: String {
        switch self {
        case .colmap: return "colmap"
        case .nerfstudio: return "nerfstudio"
        case .demo(let kind): return "demo_\(kind.rawValue)"
        }
    }

    var label: String {
        switch self {
        case .colmap: return "Colmap"
        case .nerfstudio: return "NerfStudio"
        case .demo(let kind): return "Demo: \(kind.rawValue)"
        }
    }
}

struct ZipDocumentPicker: UIViewControllerRepresentable {
    var onPick: (URL) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let types = [UTType.zip]
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: types, asCopy: true)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(
        _ uiViewController: UIDocumentPickerViewController, context: Context
    ) {}

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let parent: ZipDocumentPicker

        init(_ parent: ZipDocumentPicker) {
            self.parent = parent
        }

        func documentPicker(
            _ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]
        ) {
            guard let url = urls.first else { return }
            parent.onPick(url)
        }
    }
}

struct SelectDataSetView: View {
    @Binding var selected: SelectedDataSet
    private let allFormats: [FileFormat] =
        [
            .colmap,
            .nerfstudio,
        ] + DemoKind.allCases.map { .demo($0) }
    @State private var showPicker = false
    @State private var pickedFile: URL?
    @State private var demoSelected: DemoKind = .chair

    var body: some View {
        VStack(spacing: 24) {
            Text("Select Data Set")
                .font(.headline)

            Picker("File Format", selection: $selected.format) {
                ForEach(allFormats, id: \.self) { format in
                    Text(format.label).tag(format)
                }
            }
            .pickerStyle(.wheel)
            .padding(.horizontal)

            switch selected.format {
            case .colmap, .nerfstudio:
                Button("Select dataset(.zip") {
                    showPicker = true
                }
                .buttonStyle(.borderedProminent)
                if let url = pickedFile {
                    Text("Selected: \(url.lastPathComponent)")
                        .font(.caption)
                }
            case .demo(let kind):
                VStack {
                    Text("Use Demo Data: \(kind.rawValue)")
                        .font(.subheadline)
                }
            }

            Spacer()
        }
        .sheet(isPresented: $showPicker) {
            ZipDocumentPicker { url in
                pickedFile = url
                selected.url = url
                showPicker = false
            }
        }
        .padding()
    }
}
