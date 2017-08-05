//
//  CanvasView.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 5/23/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

import Foundation


public protocol JPSCanvasViewDelegate: class {
    func canvasViewTouchesBegan(canvasView: JPSCanvasView)
    func canvasViewTouchesEnded(canvasView: JPSCanvasView)
}

public class JPSCanvasView: UIView
{
    public var bufferImage: UIImage?
    public weak var delegate: JPSCanvasViewDelegate?
    
    private var brushWidth: CGFloat = 5.0
    private var lastPoint = CGPoint.zero
    private var swiped = false
    
    public func getImage() -> UIImage?
    {
        guard let bufferImage = self.bufferImage else { return nil }
        
        UIGraphicsBeginImageContextWithOptions(self.bounds.size, true, 1)
        defer { UIGraphicsEndImageContext() }
        
        self.bufferImage?.draw(at: CGPoint.zero)
        
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    public func clear()
    {
        self.layer.contents = nil
        self.bufferImage = nil
    }
    
    public func drawLineFrom(fromPoint: CGPoint, toPoint: CGPoint)
    {
        UIGraphicsBeginImageContextWithOptions(self.bounds.size, false, 0)
        defer { UIGraphicsEndImageContext() }
        
        guard let context = UIGraphicsGetCurrentContext() else { return }
        
        if let image = self.bufferImage {
            image.draw(at: CGPoint.zero)
        }
        
        context.move(to: fromPoint)
        context.addLine(to: toPoint)
        context.setLineCap(.round)
        context.setLineWidth(self.brushWidth)
        context.setStrokeColor(UIColor.white.cgColor)
        context.strokePath()
        
        self.bufferImage = UIGraphicsGetImageFromCurrentImageContext()
        self.layer.contents = self.bufferImage?.cgImage
    }
    
    override public func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?)
    {
        self.swiped = false
        
        if let touch = touches.first {
            self.lastPoint = touch.location(in: self)
        }
        
        self.delegate?.canvasViewTouchesBegan(canvasView: self)
    }
    
    override public func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?)
    {
        self.swiped = true
        
        if let touch = touches.first
        {
            let currentPoint = touch.location(in: self)
            self.drawLineFrom(fromPoint: self.lastPoint, toPoint: currentPoint)
            
            self.lastPoint = currentPoint
        }
    }
    
    override public func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?)
    {
        if !swiped {
            self.drawLineFrom(fromPoint: self.lastPoint, toPoint: self.lastPoint)
        }
        
        self.delegate?.canvasViewTouchesEnded(canvasView: self)
    }
}
