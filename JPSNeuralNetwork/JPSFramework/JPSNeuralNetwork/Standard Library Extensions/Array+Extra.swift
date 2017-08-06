//
//  Array+Extra.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 8/5/17.
//

import Foundation
import GameKit

extension Array
{
    func shuffled(using source: GKRandomSource) -> [Element] {
        return (self as NSArray).shuffled(using: source) as! [Element]
    }
    
    func shuffled() -> [Element] {
        return (self as NSArray).shuffled() as! [Element]
    }
}
