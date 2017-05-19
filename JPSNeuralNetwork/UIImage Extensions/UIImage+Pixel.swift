//
//  UIImage+Pixel.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 5/11/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

public extension UIImage
{
    public func getPixelColor(pos: CGPoint) -> Float
    {
        let pixelData = self.cgImage!.dataProvider!.data
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y)) + Int(pos.x)) * 4
        
        let r = (Float(data[pixelInfo]) / Float(255.0))
        let g = (Float(data[pixelInfo+1]) / Float(255.0))
        let b = (Float(data[pixelInfo+2]) / Float(255.0))
        let a = (Float(data[pixelInfo+3]) / Float(255.0))
        
        return (r + g + b + a)
    }
    
    public func pixelColors() -> [MNISTPixel]
    {
        var result = [MNISTPixel]()
        
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        
        for y in 0..<height
        {
            for x in 0..<width {
                result.append(self.getPixelColor(pos: CGPoint(x: x, y: y)))
            }
        }
        
        return result
    }
}
