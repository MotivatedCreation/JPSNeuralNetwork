//
//  JPSNeuralNetworkNeuron.swift
//
//  Created by Jonathan Sullivan on 4/4/17.
//

import Foundation

public typealias Scalar = Float
public typealias Vector = [Scalar]
public typealias Matrix = [Vector]

public class JPSNeuralNetworkNeuron
{
    /**
        Used to generate a single random weight.
    */
    private class func randomWeight(inputCount: Int) -> Scalar
    {
        let range = (1 / sqrt(Scalar(inputCount)))
        let rangeInt = UInt32(2000000 * range)
        let randomDouble = Scalar(arc4random_uniform(rangeInt)) - Scalar(rangeInt / 2)
        return (randomDouble / 1000000)
    }
    
    /**
     Used to generate a vector of random weights.
    */
    public class func randomWeights(inputCount: Int) -> Vector
    {
        return (0..<inputCount).map({ _ in
            return JPSNeuralNetworkNeuron.randomWeight(inputCount: inputCount)
        })
    }
}
