//
//  DataLoaderBase.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/07.
//

protocol DataLoaderProtocol {
    func load(resizeFactor: Double, whiteBackground: Bool) throws -> (
        TrainData, PointCloud, TILE_SIZE_H_W
    )
}
let WORKING_DIRECTORY = "work"
