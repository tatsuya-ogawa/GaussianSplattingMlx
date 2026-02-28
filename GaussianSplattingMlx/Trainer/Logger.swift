//
//  Logger.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/03.
//

class Logger {
    static let shared: Logger = Logger()
    static let enableDebug: Bool = false
    private init() {

    }
    func debug(_ message: @autoclosure () -> String) {
        #if DEBUG
        if Logger.enableDebug{
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
