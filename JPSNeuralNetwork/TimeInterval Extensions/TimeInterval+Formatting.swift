//
//  TimeInterval+Formatting.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 5/11/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

public extension TimeInterval
{
    func stringFromTimeInterval() -> NSString
    {
        let timeInterval = NSInteger(self)
        
        let seconds = (timeInterval % 60)
        let minutes = (timeInterval / 60) % 60
        let hours = (timeInterval / 3600)
        
        return NSString(format: "%0.2d:%0.2d:%0.2d", hours, minutes, seconds)
    }
}
