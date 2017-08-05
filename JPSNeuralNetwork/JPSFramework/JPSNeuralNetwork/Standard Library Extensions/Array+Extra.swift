//
//  Array+Extra.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 8/5/17.
//

import Foundation
import GameKit

public extension Array
{
    public func shuffled() -> [Element] {
        return (self as NSArray).shuffled() as! [Element]
    }
}
