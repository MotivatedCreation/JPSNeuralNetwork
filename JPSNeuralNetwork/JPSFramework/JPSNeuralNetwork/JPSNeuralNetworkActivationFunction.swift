//
//  JPSNeuralNetworkActivationFunction.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 8/5/17.
//

import Foundation

public enum JPSNeuralNetworkActivationFunction: Int
{
    case sigmoid = 0
    case hyperbolicTangent = 1
    case reLU = 2
    case softplus = 3
    
    public func derivative(_ activation: Scalar) -> Scalar
    {
        switch self
        {
        case .hyperbolicTangent:
            return (1 - activation * activation)
            
        case .sigmoid:
            return (activation * (1 - activation))
            
        case .softplus:
            return JPSNeuralNetworkActivationFunction.sigmoid.activation(activation)
            
        case .reLU:
            return (activation > 0 ? 1 : 0)
        }
    }
    
    public func gradient(_ activations: Vector) -> Vector
    {
        return activations.map({
            return self.derivative($0)
        })
    }
    
    public func activation(_ weightedInput: Scalar) -> Scalar
    {
        switch self
        {
        case .hyperbolicTangent:
            return tanh(weightedInput)
            
        case .sigmoid:
            return (1 / (1 + exp(-weightedInput)))
            
        case .softplus:
            return log(1 + exp(weightedInput))
            
        case .reLU:
            return max(0, weightedInput)
        }
    }
    
    public func activations(_ weightedInputs: Vector) -> Vector
    {
        return weightedInputs.map({
            return self.activation($0)
        })
    }
}
