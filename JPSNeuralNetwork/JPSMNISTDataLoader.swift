//
//  JPSMNISTDataLoader.swift
//
//  Created by Jonathan Sullivan on 5/3/17.
//

import Foundation

public typealias MNISTLabel = [Float]
public typealias MNISTPixel = Float
public typealias MNISTImage = [MNISTPixel]

public class JPSMNISTDataLoader: JPSBinaryDataLoader
{
    private let labelSize = 1
    private let labelOffset = 8
    private let imageSize: Int!
    private let imageOffset = 16
    private let numberOfRows = 28
    private let numberOfColumns = 28
    private let numberOfItemsOffset = 4
    
    public override init() {
        self.imageSize = (numberOfRows * numberOfColumns)
    }
    
    public class func normalizePixel(_ value: UInt8) -> Float {
        return Float(Float(value) / 255.0)
    }
    
    private class func pixels(forImageData imageData: NSData) -> MNISTImage
    {
        var mnistImage = MNISTImage()
        
        for pixelLocation in 0..<imageData.length
        {
            let pixelRange = NSMakeRange(Int(pixelLocation), 1)
            
            var pixel: UInt8 = 0
            imageData.getBytes(&pixel, range: pixelRange)
            
            let normalizedPixel = JPSMNISTDataLoader.normalizePixel(pixel)
            mnistImage.append(normalizedPixel)
        }
        
        return mnistImage
    }
    
    private class func loadTrainingData(numberOfItemsOffset: Int, labelOffset: Int, labelSize: Int, imageOffset: Int, imageSize: Int) throws -> (labels: [MNISTLabel], images: [MNISTImage])
    {
        let bufferSize = 16384
        
        let labelData = (try JPSBinaryDataLoader.data(forResource: "train-labels", ofType: "idx1-ubyte", bufferSize: bufferSize) as NSData)
        
        let imageData = (try JPSBinaryDataLoader.data(forResource: "train-images", ofType: "idx3-ubyte", bufferSize: bufferSize) as NSData)
        
        var count: UInt32 = 0
        let dataRange = NSMakeRange(numberOfItemsOffset, labelOffset)
        
        labelData.getBytes(&count, range: dataRange)
        let numberOfLabels = UInt32(bigEndian: count)
        
        var labels = [MNISTLabel]()
        var images = [MNISTImage]()
        
        for location in 0..<(numberOfLabels)
        {
            let labelDataRange = NSMakeRange(labelOffset + (labelSize * Int(location)), labelSize)
            let imagePixelDataRange = NSMakeRange(imageOffset + (imageSize * Int(location)), imageSize)
            
            var labelVector = MNISTLabel(repeating: 0, count: 10)
            
            var label: Int = 0
            labelData.getBytes(&label, range: labelDataRange)
            
            labelVector[label] = 1
            labels.append(labelVector)
            
            let imagePixelData = imageData.subdata(with: imagePixelDataRange)
            let imagePixels = JPSMNISTDataLoader.pixels(forImageData: imagePixelData as NSData)
            images.append(imagePixels)
        }
        
        return(labels, images)
    }
    
    public func normalizePixel(_ value: MNISTPixel)
    {
        return
    }
    
    public func pixels(forImageData imageData: NSData) -> MNISTImage {
        return JPSMNISTDataLoader.pixels(forImageData: imageData)
    }
    
    public func loadTrainingData() throws -> (labels: [MNISTLabel], images: [MNISTImage])
    {
        return try JPSMNISTDataLoader.loadTrainingData(numberOfItemsOffset: self.numberOfItemsOffset, labelOffset: self.labelOffset, labelSize: self.labelSize, imageOffset: self.imageOffset, imageSize: self.imageSize)
    }
}
