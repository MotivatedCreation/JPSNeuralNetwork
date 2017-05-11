//
//  JPSNeuralNetworkNeuron.swift
//
//  Created by Jonathan Sullivan on 4/4/17.
//

import Foundation

public typealias Scalar = Float
public typealias Vector = [Scalar]
public typealias Matrix = [Vector]

public enum JPSNeuralNetworkCostFunction: Int
{
    case meanSquared = 0
    case crossEntropy = 1
    
    func derivative(OfOutput output: Scalar, targetOutput: Scalar) -> Scalar
    {
        switch self
        {
        case .crossEntropy:
            return (output - targetOutput) / ((1 - output) * output)
            
        case .meanSquared:
            fallthrough
            
        default:
            return (output - targetOutput)
        }
    }
    
    func cost(forOutputs outputs: Vector, targetOutputs: Vector) -> Scalar
    {
        switch self
        {
        case .crossEntropy:
            return -zip(outputs, targetOutputs).reduce(0, { (sum, pair) -> Scalar in
                let temp = pair.1 * log(pair.0)
                return sum + temp + (1 - pair.1) * log(1 - pair.0)
            })
            
        case .meanSquared:
            fallthrough
            
        default:
            return 0.5 * zip(outputs, targetOutputs).reduce(0, { (sum, pair) -> Scalar in
                return pow(pair.1 - pair.0, 2)
            })
        }
    }
}

public enum JPSNeuralNetworkActivationFunction: Int
{
    case sigmoid = 0
    case hyperbolicTangent = 1
    
    func derivative(_ activation: Scalar) -> Scalar
    {
        switch self
        {
        case .hyperbolicTangent:
            return (1 - pow(activation, 2))
            
        case .sigmoid:
            fallthrough
            
        default:
            return (activation * (1 - activation))
        }
    }
    
    func activation(_ weightedInput: Scalar) -> Scalar
    {
        switch self
        {
        case .hyperbolicTangent:
            return tanh(weightedInput)
            
        case .sigmoid:
            fallthrough
            
        default:
            return (1 / (1 + exp(-weightedInput)))
        }
    }
}

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
        var weights = Vector()
        
        for _ in 0..<inputCount
        {
            let weight = JPSNeuralNetworkNeuron.randomWeight(inputCount: inputCount)
            weights.append(weight)
        }
        
        return weights
    }
}
