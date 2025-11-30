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
    func getOriginalImageSize() throws -> (width: Int, height: Int)
}
let WORKING_DIRECTORY = "work"
