//
//  JPSNeuralNetwork.swift
//
//  Created by Jonathan Sullivan on 4/4/17.
//

import Foundation
import Accelerate

public protocol JPSNeuralNetworkDelegate
{
    func network(costDidChange cost: Scalar)
    func network(progressDidChange progress: Scalar)
    func network(overallProgressDidChange progress: Scalar)
}

public enum JPSNeuralNetworkCostFunction: Int
{
    case sumOfSquared = 0
    case meanSquared = 1
    case crossEntropy = 2
    
    func derivative(OfOutput output: Scalar, targetOutput: Scalar) -> Scalar
    {
        switch self
        {
        case .crossEntropy:
            return ((output - targetOutput) / ((1 - output) * output))
            
        case .meanSquared:
            fallthrough
            
        default:
            return (output - targetOutput)
        }
    }
    
    func gradient(OfOutputs outputs: Vector, targetOutputs: Vector) -> Vector
    {
        return zip(outputs, targetOutputs).map({
            return self.derivative(OfOutput: $0, targetOutput: $1)
        })
    }
    
    func cost(forOutputs outputs: Matrix, targetOutputs: Matrix) -> Scalar
    {
        switch self
        {
        case .crossEntropy:
            return zip(outputs.joined(), targetOutputs.joined()).reduce(0, { (sum, pair) -> Scalar in
                return (sum + (log(pair.0) + (1 - pair.1) * log(1 - pair.0)))
            })
            
        case .meanSquared:
            return (2 * JPSNeuralNetworkCostFunction.sumOfSquared.cost(forOutputs: outputs, targetOutputs: targetOutputs) / Scalar(outputs.count))
            
        case .sumOfSquared:
            return zip(outputs.joined(), targetOutputs.joined()).reduce(0, { (sum, pair) -> Scalar in
                return (sum + pow(pair.1 - pair.0, 2))
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
            return (activation * (1 - activation))
        }
    }
    
    func gradient(_ activations: Vector) -> Vector
    {
        return activations.map({
            return self.derivative($0)
        })
    }
    
    func activation(_ weightedInput: Scalar) -> Scalar
    {
        switch self
        {
        case .hyperbolicTangent:
            return tanh(weightedInput)
            
        case .sigmoid:
            return (1 / (1 + exp(-weightedInput)))
        }
    }
    
    func activations(_ weightedInputs: Vector) -> Vector
    {
        return weightedInputs.map({
            return self.activation($0)
        })
    }
}

public class JPSNeuralNetwork
{
    private var isTraining = false
    
    private typealias FeedForwardResult = (inputs: Matrix, activations: Matrix, activationRates: Matrix)
    
    public let architecture: [Int]!
    public let activationFunctions: [JPSNeuralNetworkActivationFunction]!
    
    public var weights: Matrix!
    public var bias: Scalar = 1
    public var delegate: JPSNeuralNetworkDelegate?
    
    public init(architecture: [Int], activationFunctions: [JPSNeuralNetworkActivationFunction])
    {
        self.architecture = architecture
        self.activationFunctions = activationFunctions
        self.weights = JPSNeuralNetwork.weights(forArchitecture: self.architecture)
    }

    private class func weights(forArchitecture architecture: [Int]) -> Matrix
    {
        var weights = Matrix()
        
        var previousNumberOfInputs = architecture[0]
        
        for neuronCount in architecture[1..<architecture.count]
        {
            // Plus one for the bias weight.
            
            let neuronWeights = JPSNeuralNetworkLayer.randomWeights(neuronCount: neuronCount, inputCount: previousNumberOfInputs + 1)
            weights.append(neuronWeights)
            
            previousNumberOfInputs = neuronCount
        }
        
        return weights
    }
    
    public func feedForward(inputs: Vector) -> Vector {
        return self.feedForward(inputs: inputs).activations.last!
    }
    
    private func feedForward(inputs: Vector) -> FeedForwardResult
    {
        var previousActivations = inputs
        
        var networkInputs = Matrix()
        var networkActivations = Matrix()
        var networkActivationRates = Matrix()
        
        // Ignore the input layer as it's just a place holder.
        
        for (activationFunction, (neuronCount, layerWeights)) in zip(self.activationFunctions, zip(self.architecture[1..<self.architecture.count], self.weights))
        {
            // Append one for the bias node.
            
            var layerInputs = previousActivations
            layerInputs.append(self.bias)
            networkInputs.append(layerInputs)
            
            let feedForward = JPSNeuralNetworkLayer.feedForward(neuronCount: neuronCount, activationFunction: activationFunction, inputs: layerInputs, weights: layerWeights)
            
            previousActivations = feedForward.activations
            
            networkActivations.append(previousActivations)
            networkActivationRates.append(feedForward.activationRates)
        }
        
        return (networkInputs, networkActivations, networkActivationRates)
    }
    
    private func outputGradient(costFunction: JPSNeuralNetworkCostFunction, activations: Vector, activationRates: Vector, targetOutputs: Vector) -> Vector
    {
        var gradient = Vector(repeating: 0, count: activationRates.count)
        let costRates = costFunction.gradient(OfOutputs: activations, targetOutputs: targetOutputs)
        
        vDSP_vmul(activationRates, 1,
                  costRates, 1,
                  &gradient, 1,
                  vDSP_Length(gradient.count))
        
        return gradient
    }
    
    private func gradient(costFunction: JPSNeuralNetworkCostFunction, activations: Matrix, activationRates: Matrix, targetOutputs: Vector) -> Matrix
    {
        let reversedWeights = self.weights.reversed()
        var reversedActivations = (activations.reversed() as Matrix)
        var reversedActivationRates = (activationRates.reversed() as Matrix)
        
        let outputLayerActivations = reversedActivations.removeFirst()
        let outputLayerActivationRates = reversedActivationRates.removeFirst()
        var previousGradient = self.outputGradient(costFunction: costFunction, activations: outputLayerActivations, activationRates: outputLayerActivationRates, targetOutputs: targetOutputs)
        
        var gradient = Matrix()
        gradient.append(previousGradient)
        
        for (layerActivationRates, (layerActivations, layerWeights)) in zip(reversedActivationRates, zip(reversedActivations, reversedWeights))
        {
            previousGradient = JPSNeuralNetworkLayer.gradient(forActivations: layerActivations, activationRates: layerActivationRates, weights: layerWeights, gradient: previousGradient)
            
            gradient.append(previousGradient)
        }
        
        return gradient.reversed()
    }
    
    private func update(weights: Matrix, learningRate: Scalar, gradient: Matrix, inputs: Matrix) -> Matrix
    {
        return zip(zip(inputs, weights), gradient).map({ layerInputsAndWeights, layerGradient in
            print(layerInputsAndWeights.1.count)
            print(layerInputsAndWeights.0.count)
            print(layerGradient.count)
            return JPSNeuralNetworkLayer.update(weights: layerInputsAndWeights.1, learningRate: learningRate, gradient: layerGradient, inputs: layerInputsAndWeights.0)
        })
    }
    
    private func backpropagate(costFunction: JPSNeuralNetworkCostFunction, learningRate: Scalar, inputs: Matrix, activations: Matrix, activationRates: Matrix, targetOutput: Vector) -> Matrix
    {
        let gradient = self.gradient(costFunction: costFunction, activations: activations, activationRates: activationRates, targetOutputs: targetOutput)
        
        return self.update(weights: weights, learningRate: learningRate, gradient: gradient, inputs: inputs)
    }
    
    public func train(epochs: Int, costFunction: JPSNeuralNetworkCostFunction, learningRate: Scalar, trainingInputs: Matrix, targetOutputs: Matrix) -> Matrix
    {
        self.isTraining = true
        
        for epoch in 0..<epochs
        {
            var activations = Matrix()
            
            for (index, (inputs, targetOutput)) in zip(trainingInputs, targetOutputs).enumerated()
            {
                let progress = (Scalar(index + 1) / Scalar(targetOutputs.count))
                self.delegate?.network(progressDidChange: progress)
                
                let overallProgress = ((Scalar(epoch) + progress) / Scalar(epochs))
                self.delegate?.network(overallProgressDidChange: overallProgress)
                
                let feedForward: FeedForwardResult = self.feedForward(inputs: inputs)
                activations.append(feedForward.activations.last!)
                
                self.weights = self.backpropagate(costFunction: costFunction, learningRate: learningRate, inputs: feedForward.inputs, activations: feedForward.activations, activationRates: feedForward.activationRates, targetOutput: targetOutput)
                
                if (!self.isTraining) { break }
            }
            
            if (!self.isTraining) { break }
            
            let cost = costFunction.cost(forOutputs: activations, targetOutputs: targetOutputs)
            delegate?.network(costDidChange: cost)
        }
        
        return self.weights
    }
    
    public func cancelTraining() {
        self.isTraining = false
    }
}
