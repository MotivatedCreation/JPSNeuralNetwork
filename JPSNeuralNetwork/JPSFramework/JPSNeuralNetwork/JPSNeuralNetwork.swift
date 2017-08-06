//
//  JPSNeuralNetwork.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 4/4/17.
//

import Foundation
import Accelerate

public protocol JPSNeuralNetworkDelegate: class
{
    func network(_ network: JPSNeuralNetwork, errorDidChange error: Scalar)
    func network(_ network: JPSNeuralNetwork, progressDidChange progress: Scalar)
    func network(_ network: JPSNeuralNetwork, overallProgressDidChange progress: Scalar)
}

public class JPSNeuralNetwork
{
    public enum JPSNeuralNetworkError: Error {
        case TrainingError(String)
    }
    
    fileprivate var isTraining = false
    
    public typealias FeedForwardResult = (inputs: Matrix, activations: Matrix)
    
    public var bias: Scalar = 1
    public var previousWeights: Matrix!
    
    public var weights: Matrix! {
        didSet { self.previousWeights = self.weights }
    }
    
    public weak var delegate: JPSNeuralNetworkDelegate?
}

/*
    Feed Forward Functions
*/
extension JPSNeuralNetwork
{
    private class func feed(architecture: [Int], inputs: Vector, weights: Matrix, bias: Scalar, activationFunctions: [JPSNeuralNetworkActivationFunction]) -> FeedForwardResult
    {
        var previousActivations = inputs
        
        var networkInputs = Matrix()
        var networkActivations = Matrix()
        
        // Ignore the input layer as it's just a placeholder.
        for (activationFunction, (neuronCount, layerWeights)) in zip(activationFunctions, zip(architecture[1..<architecture.count], weights))
        {
            var layerInputs = previousActivations
            layerInputs.append(bias)
            networkInputs.append(layerInputs)
            
            previousActivations = JPSNeuralNetworkLayer.feedForward(neuronCount: neuronCount, activationFunction: activationFunction, inputs: layerInputs, weights: layerWeights)
            networkActivations.append(previousActivations)
        }
        
        return (networkInputs, networkActivations)
    }
    
    fileprivate func feedForward(architecture: [Int], activationFunctions: [JPSNeuralNetworkActivationFunction], inputs: Vector) -> FeedForwardResult {
        return JPSNeuralNetwork.feed(architecture: architecture, inputs: inputs, weights: self.weights, bias: self.bias, activationFunctions: activationFunctions)
    }
    
    public func feedForward(architecture: [Int], activationFunctions: [JPSNeuralNetworkActivationFunction], inputs: Vector) -> Vector {
        return self.feedForward(architecture: architecture, activationFunctions: activationFunctions, inputs: inputs).activations.last!
    }
}

/*
    Helper Functions
*/
extension JPSNeuralNetwork
{
    public class func weights(forArchitecture architecture: [Int]) -> Matrix
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
    
    public func preprocess(inputs: Vector) -> Vector
    {
        let average = inputs.reduce(0, +) / Scalar(inputs.count)
        //let variance = inputs.map({ return $0 * $0 }).reduce(0, +) / Scalar(inputs.count)
        
        let preprocessedInputs = inputs.map({
            return ($0 - average)
        })
        
        return preprocessedInputs
    }
    
    fileprivate class func validateParameters(architecture: [Int], activationFunctions: [JPSNeuralNetworkActivationFunction], inputs: Matrix, targetOutputs: Matrix) throws
    {
        if architecture.count - 1 != activationFunctions.count {
            throw NSError(domain: "InvalidTopology", code: 1, userInfo: [NSLocalizedFailureReasonErrorKey: "The number of layers do not match the number of activation functions."])
        }
        
        let inputNeuronsCount = architecture[0]
        
        for (i, input) in inputs.enumerated()
        {
            if input.count != inputNeuronsCount {
                throw NSError(domain: "InvalidArchitecture", code: 2, userInfo: [NSLocalizedFailureReasonErrorKey: "The dimension of the input (\(input.count)) at index \(i) does not equal the dimension of the input layer (\(inputNeuronsCount)) in the architecture."])
            }
        }
        
        let outputNeuronsCount = architecture.last!
        
        for (i, targetOutput) in targetOutputs.enumerated()
        {
            if targetOutput.count != outputNeuronsCount {
                throw NSError(domain: "InvalidArchitecture", code: 3, userInfo: [NSLocalizedFailureReasonErrorKey: "The dimension of the label (\(targetOutput.count)) at index \(i) does not equal the dimension of the output layer (\(outputNeuronsCount)) in the architecture."])
            }
        }
    }
}

/*
    Training Functions
*/
extension JPSNeuralNetwork
{
    private func outputGradient(activationFunctions: [JPSNeuralNetworkActivationFunction], errorFunction: JPSNeuralNetworErrorFunction, activations: Vector, targetOutputs: Vector) -> Vector
    {
        var gradient = Vector(repeating: 0, count: activations.count)
        let activationGradient = activationFunctions.last!.gradient(activations)
        let errorGradient = errorFunction.gradient(OfOutputs: activations, targetOutputs: targetOutputs)
        
        vDSP_vmul(errorGradient, 1,
                  activationGradient, 1,
                  &gradient, 1,
                  vDSP_Length(gradient.count))
        
        return gradient
    }
    
    private func gradient(activationFunctions: [JPSNeuralNetworkActivationFunction], errorFunction: JPSNeuralNetworErrorFunction, activations: Matrix, targetOutputs: Vector) -> Matrix
    {
        let reversedWeights = self.weights.reversed()
        var reversedActivations = (activations.reversed() as Matrix)
        
        let outputLayerActivations = reversedActivations.removeFirst()
        var previousGradient = self.outputGradient(activationFunctions: activationFunctions, errorFunction: errorFunction, activations: outputLayerActivations, targetOutputs: targetOutputs)
        
        var gradient = Matrix()
        gradient.append(previousGradient)
        
        for (layerActivationFunction, (layerActivations, layerWeights)) in zip(activationFunctions, zip(reversedActivations, reversedWeights))
        {
            let activationGradient = layerActivationFunction.gradient(layerActivations)
            previousGradient = JPSNeuralNetworkLayer.gradient(forActivationGradient: activationGradient, weights: layerWeights, gradient: previousGradient)
            gradient.append(previousGradient)
        }
        
        return gradient.reversed()
    }
    
    private func updateWeights(learningRate: Scalar, momentum: Scalar, gradient: Matrix, inputs: Matrix) -> Matrix
    {
        return zip(zip(inputs, zip(self.weights, self.previousWeights)), gradient).map({ factors, layerGradient in
            return JPSNeuralNetworkLayer.update(weights: factors.1.0, learningRate: learningRate, momentum: momentum, gradient: layerGradient, inputs: factors.0, previousWeights: factors.1.1)
        })
    }
    
    public func backpropagate(activationFunctions: [JPSNeuralNetworkActivationFunction], errorFunction: JPSNeuralNetworErrorFunction, learningRate: Scalar, momentum: Scalar, inputs: Matrix, activations: Matrix, targetOutput: Vector) -> Matrix
    {
        let gradient = self.gradient(activationFunctions: activationFunctions, errorFunction: errorFunction, activations: activations, targetOutputs: targetOutput)
        
        return self.updateWeights(learningRate: learningRate, momentum: momentum, gradient: gradient, inputs: inputs)
    }
    
    public func train(epochs: Int, architecture: [Int], activationFunctions: [JPSNeuralNetworkActivationFunction], errorFunction: JPSNeuralNetworErrorFunction, learningRate: Scalar, momentum: Scalar, trainingInputs: Matrix, targetOutputs: Matrix) throws
    {
        try JPSNeuralNetwork.validateParameters(architecture: architecture, activationFunctions: activationFunctions, inputs: trainingInputs, targetOutputs: targetOutputs)
        
        self.isTraining = true
        
        for epoch in 0..<epochs
        {
            var activations = Matrix()
            
            for (index, (inputs, targetOutput)) in zip(trainingInputs, targetOutputs).enumerated()
            {
                let progress = (Scalar(index + 1) / Scalar(targetOutputs.count))
                self.delegate?.network(self, progressDidChange: progress)
                
                let overallProgress = ((Scalar(epoch) + progress) / Scalar(epochs))
                self.delegate?.network(self, overallProgressDidChange: overallProgress)
                
                let feedForward: FeedForwardResult = self.feedForward(architecture: architecture, activationFunctions: activationFunctions, inputs: inputs)
                activations.append(feedForward.activations.last!)
                
                let newWeights = self.backpropagate(activationFunctions: activationFunctions, errorFunction: errorFunction, learningRate: learningRate, momentum: momentum, inputs: feedForward.inputs, activations: feedForward.activations, targetOutput: targetOutput)
                self.previousWeights = self.weights
                self.weights = newWeights
                
                if (!self.isTraining) { break }
            }
            
            if (!self.isTraining) { break }
            
            let error = errorFunction.error(forOutputs: activations, targetOutputs: targetOutputs)
            self.delegate?.network(self, errorDidChange: error)
        }
    }
    
    public func cancelTraining() {
        self.isTraining = false
    }
}
