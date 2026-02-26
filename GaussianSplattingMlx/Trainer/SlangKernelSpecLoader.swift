import Foundation
import MLXFast

struct SlangKernelSpec: Decodable {
    let kernelName: String
    let inputNames: [String]
    let outputNames: [String]
    let source: String
    let header: String
    let atomicOutputs: Bool?

    private enum CodingKeys: String, CodingKey {
        case kernelName = "kernel_name"
        case inputNames = "input_names"
        case outputNames = "output_names"
        case source
        case header
        case atomicOutputs = "atomic_outputs"
    }
}

enum SlangKernelSpecLoaderError: LocalizedError {
    case missingResource(String)
    case invalidSpec(String)

    var errorDescription: String? {
        switch self {
        case .missingResource(let name):
            return "Missing Slang kernel resource: \(name).json"
        case .invalidSpec(let detail):
            return "Invalid Slang kernel spec: \(detail)"
        }
    }
}

enum SlangKernelSpecLoader {
    static func loadKernel(
        named name: String,
        from bundle: Bundle = .main,
        subdirectory: String = "Slang"
    ) throws -> MLXFast.MLXFastKernel {
        let spec = try loadSpec(named: name, from: bundle, subdirectory: subdirectory)
        return MLXFast.metalKernel(
            name: spec.kernelName,
            inputNames: spec.inputNames,
            outputNames: spec.outputNames,
            source: spec.source,
            header: spec.header,
            atomicOutputs: spec.atomicOutputs ?? false
        )
    }

    static func loadSpec(
        named name: String,
        from bundle: Bundle = .main,
        subdirectory: String = "Slang"
    ) throws -> SlangKernelSpec {
        guard let url = bundle.url(forResource: name, withExtension: "json", subdirectory: subdirectory)
            ?? bundle.url(forResource: name, withExtension: "json")
        else {
            throw SlangKernelSpecLoaderError.missingResource(name)
        }

        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(SlangKernelSpec.self, from: data)
        } catch {
            throw SlangKernelSpecLoaderError.invalidSpec(
                "failed to decode \(name).json: \(error.localizedDescription)"
            )
        }
    }
}
