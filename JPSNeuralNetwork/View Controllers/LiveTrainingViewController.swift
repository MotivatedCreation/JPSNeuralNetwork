//
//  LiveTrainingViewController.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 5/11/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

import Foundation

class LiveTrainingViewController: UIViewController
{
    @IBOutlet weak var predictionLabel: UILabel!
    @IBOutlet weak var tempImageView: UIImageView!
    
    var neuralNetwork: JPSNeuralNetwork?
    
    var lastPoint = CGPoint.zero
    var red: CGFloat = 0.0
    var green: CGFloat = 0.0
    var blue: CGFloat = 0.0
    var brushWidth: CGFloat = 1.0
    var opacity: CGFloat = 1.0
    var swiped = false
    
    public func prediction() -> Scalar
    {
        let inputs = self.tempImageView.image!.pixelColors()
        let outputs = self.neuralNetwork!.feedForward(inputs: inputs)
        
        return Scalar(outputs.max()!)
    }
    
    func drawLineFrom(fromPoint: CGPoint, toPoint: CGPoint)
    {
        UIGraphicsBeginImageContext(self.view.bounds.size)
        let context = UIGraphicsGetCurrentContext()
        self.tempImageView.image?.draw(in: CGRect(x: 0, y: 0, width: self.view.frame.size.width, height: self.view.frame.size.height))
        
        context?.move(to: fromPoint)
        context?.addLine(to: toPoint)
        context?.setLineCap(.round)
        context?.setLineWidth(self.brushWidth)
        context?.setStrokeColor(red: self.red, green: self.green, blue: self.blue, alpha: 1.0)
        context?.setBlendMode(.normal)
        context?.strokePath()
        
        self.tempImageView.image = UIGraphicsGetImageFromCurrentImageContext()
        self.tempImageView.alpha = self.opacity
        UIGraphicsEndImageContext()
        
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?)
    {
        self.swiped = false
        
        if let touch = touches.first {
            self.lastPoint = touch.location(in: self.view)
        }
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?)
    {
        self.swiped = true
        
        if let touch = touches.first
        {
            let currentPoint = touch.location(in: self.view)
            self.drawLineFrom(fromPoint: self.lastPoint, toPoint: currentPoint)
            
            self.lastPoint = currentPoint
        }
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?)
    {
        if !swiped {
            self.drawLineFrom(fromPoint: self.lastPoint, toPoint: self.lastPoint)
        }
        
        //self.predictionLabel.text = "Predication: \(self.prediction())"
        
        self.tempImageView.image = nil
    }
}
