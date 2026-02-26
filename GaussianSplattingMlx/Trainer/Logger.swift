//
//  Logger.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/03.
//

class Logger {
    static let shared: Logger = Logger()
    let isDebug: Bool
    private init(isDebug:Bool=false) {
        self.isDebug = isDebug
    }
    func debug(_ message: @autoclosure () -> String) {
        #if DEBUG
        if self.isDebug{
            print(message())
        }
        #endif
    }
    func info(_ message: String) {
        print(message)
    }
    func error(_ message: String, error: Error?) {
        print("Error: \(message)")
        if let error = error {
            print(error.localizedDescription)
        }
    }
    func error(error: Error?) {
        if let error = error {
            print(error.localizedDescription)
        }
    }
}
