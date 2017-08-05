//
//  UIImage+Pixel.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 5/11/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

internal extension CGRect
{
    internal func area() -> CGFloat {
        return self.width * self.height
    }
}

public protocol PixelData
{
    static func pixels(forCGImage cgImage: CGImage, colorSpace: CGColorSpace, bitmapInfo: UInt32) -> [UInt8]?
    static func flattenedPixels(forBytes bytes: [UInt8], imageSize: CGSize, numberOfComponents: Int, pixels: inout [UInt8])
    static func image(fromPixels pixels: [UInt8], imageSize: CGSize, bitsPerComponent: Int, colorSpace: CGColorSpace, bitmapInfo: CGBitmapInfo, decode: UnsafePointer<CGFloat>?, shouldInterpolate: Bool, intent: CGColorRenderingIntent) -> CGImage?
    
    func pixels() -> [UInt8]?
    func grayScalePixels() -> [UInt8]?
    func image(fromPixels pixels: [UInt8]) -> CGImage?
    func grayScaleImage(fromGrayScalePixels grayScalePixels: [UInt8]) -> CGImage?
}

extension CGImage: PixelData
{
    public static func flattenedPixels(forBytes bytes: [UInt8], imageSize: CGSize, numberOfComponents: Int, pixels: inout [UInt8])
    {
        let width = Int(imageSize.width)
        let height = Int(imageSize.height)
        
        for y in 0..<height
        {
            for x in 0..<width
            {
                let index = (width * y + x) * numberOfComponents
                
                for componentOffset in 0..<numberOfComponents
                {
                    let componentIndex = index + componentOffset
                    pixels[componentIndex] = bytes[componentIndex]
                }
            }
        }
        
    }
    
    public static func pixels(forCGImage cgImage: CGImage, colorSpace: CGColorSpace, bitmapInfo: UInt32) -> [UInt8]?
    {
        let width = cgImage.width
        let height = cgImage.height
        let maxComponentValue = UInt8.max
        let bitsPerComponent = cgImage.bitsPerComponent
        let repeatedValue = colorSpace.model == .monochrome ? maxComponentValue: 0
        let numberOfComponents = colorSpace.numberOfComponents + (bitmapInfo & CGBitmapInfo.alphaInfoMask.rawValue == 0 ? 0 : 1)
        
        let bytesPerRow = width * numberOfComponents
        let totalBytes =  height * bytesPerRow
        
        var bytes = [UInt8](repeating: repeatedValue, count: totalBytes)
        
        let didWriteBytes = bytes.withUnsafeMutableBufferPointer { (buffer) -> Bool in
            
            guard let imageContext = CGContext(data: buffer.baseAddress!, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else { return false }
            imageContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            
            return true
        }
        
        var pixels: [UInt8]?
        
        if didWriteBytes
        {
            pixels = [UInt8](repeating: repeatedValue, count: totalBytes)
            
            let imageSize = CGSize(width: width, height: height)
            CGImage.flattenedPixels(forBytes: bytes, imageSize: imageSize, numberOfComponents: numberOfComponents, pixels: &pixels!)
        }
        
        return pixels
    }
    
    public static func image(fromPixels pixels: [UInt8], imageSize: CGSize, bitsPerComponent: Int, colorSpace: CGColorSpace, bitmapInfo: CGBitmapInfo, decode: UnsafePointer<CGFloat>?, shouldInterpolate: Bool, intent: CGColorRenderingIntent) -> CGImage?
    {
        let width = Int(imageSize.width)
        let height = Int(imageSize.height)
        let numberOfComponents = colorSpace.numberOfComponents + (bitmapInfo.rawValue & CGBitmapInfo.alphaInfoMask.rawValue == 0 ? 0 : 1)
        
        let bytesPerRow = width * numberOfComponents
        let bitsPerPixel = bitsPerComponent * numberOfComponents
        
        let data = Data(bytes: pixels)
        guard let dataProvider = CGDataProvider(data: data as CFData) else { return nil }
        
        guard let cgImage = CGImage(width: width, height: height, bitsPerComponent: bitsPerComponent, bitsPerPixel: bitsPerPixel, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo, provider: dataProvider, decode: decode, shouldInterpolate: shouldInterpolate, intent: intent) else { return nil }
        
        return cgImage
    }
    
    public func grayScalePixels() -> [UInt8]?
    {
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        return CGImage.pixels(forCGImage: self, colorSpace: colorSpace, bitmapInfo: bitmapInfo)
    }
    
    public func grayScaleImage(fromGrayScalePixels grayScalePixels: [UInt8]) -> CGImage?
    {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let imageSize = CGSize(width: self.width, height: self.height)
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        return CGImage.image(fromPixels: grayScalePixels, imageSize: imageSize, bitsPerComponent: self.bitsPerComponent, colorSpace: colorSpace, bitmapInfo: bitmapInfo, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
    
    public func pixels() -> [UInt8]?
    {
        guard let colorSpace = self.colorSpace else { return nil }
        
        return CGImage.pixels(forCGImage: self, colorSpace: colorSpace, bitmapInfo: self.bitmapInfo.rawValue)
    }
    
    public func image(fromPixels pixels: [UInt8]) -> CGImage?
    {
        guard let colorSpace = self.colorSpace else { return nil }
        
        let imageSize = CGSize(width: self.width, height: self.height)
        return CGImage.image(fromPixels: pixels, imageSize: imageSize, bitsPerComponent: self.bitsPerComponent, colorSpace: colorSpace, bitmapInfo: self.bitmapInfo, decode: self.decode, shouldInterpolate: self.shouldInterpolate, intent: self.renderingIntent)
    }
}
