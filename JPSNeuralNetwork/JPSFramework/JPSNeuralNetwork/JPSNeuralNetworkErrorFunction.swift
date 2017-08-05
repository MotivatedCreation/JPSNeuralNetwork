//
//  JPSNeuralNetworkErrorFunction.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 8/5/17.
//

import Foundation

public enum JPSNeuralNetworErrorFunction: Int
{
    case sumOfSquared = 0
    case meanSquared = 1
    case crossEntropy = 2
    
    public func derivative(OfOutput output: Scalar, targetOutput: Scalar) -> Scalar
    {
        switch self
        {
        case .crossEntropy:
            return ((output - targetOutput) / ((1 - output) * output))
            
        case .sumOfSquared:
            fallthrough
            
        default:
            return -(targetOutput - output)
        }
    }
    
    public func gradient(OfOutputs outputs: Vector, targetOutputs: Vector) -> Vector
    {
        return zip(outputs, targetOutputs).map({
            return self.derivative(OfOutput: $0, targetOutput: $1)
        })
    }
    
    public func error(forOutputs outputs: Matrix, targetOutputs: Matrix) -> Scalar
    {
        switch self
        {
        case .crossEntropy:
            return zip(outputs.joined(), targetOutputs.joined()).reduce(0, { (sum, pair) -> Scalar in
                return (sum + (log(pair.0) + (1 - pair.1) * log(1 - pair.0)))
            })
            
        case .meanSquared:
            return (2 * JPSNeuralNetworErrorFunction.sumOfSquared.error(forOutputs: outputs, targetOutputs: targetOutputs) / Scalar(outputs.count))
            
        case .sumOfSquared:
            return 0.5 * zip(outputs.joined(), targetOutputs.joined()).reduce(0, { (sum, pair) -> Scalar in
                return sum + (pair.1 - pair.0) * (pair.1 - pair.0)
            })
        }
    }
}
