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

public class JPSNeuralNetwork
{
    private typealias FeedForwardResult = (inputs: Matrix, activations: Matrix, activationRates: Matrix)
    
    private class func cost(costFunction: JPSNeuralNetworkCostFunction, activations: Matrix, targetOutputs: Matrix) -> Scalar
    {
        var cost: Scalar = 0
        
        for (activation, targetOutput) in zip(activations, targetOutputs) {
            cost += costFunction.cost(forOutputs: activation, targetOutputs: targetOutput)
        }
        
        cost /= Scalar(targetOutputs.count)
        
        return cost
    }
    
    private class func buffers(forTopology topology: [Int]) -> Matrix
    {
        var buffers = Matrix()
        
        for neuronCount in topology[1..<topology.count]
        {
            let buffer = Vector(repeating: 0, count: neuronCount)
            buffers.append(buffer)
        }
        
        return buffers
    }
    
    private class func weights(forTopology topology: [Int]) -> Matrix
    {
        var weights = Matrix()
        
        var previousNumberOfInputs = topology[0]
        
        for neuronCount in topology[1..<topology.count]
        {
            // Plus one for the bias weight.
            
            let neuronWeights = JPSNeuralNetworkLayer.randomWeights(neuronCount: neuronCount, inputCount: previousNumberOfInputs + 1)
            weights.append(neuronWeights)
            
            previousNumberOfInputs = neuronCount
        }
        
        return weights
    }
    
    public class func feedForward(topology: [Int], activationFunction: JPSNeuralNetworkActivationFunction, inputs: Vector, weights: Matrix) -> Vector {
        return JPSNeuralNetwork.feedForward(topology: topology, activationFunction: activationFunction, inputs: inputs, weights: weights).activations.last!
    }
    
    private class func feedForward(topology: [Int], activationFunction: JPSNeuralNetworkActivationFunction, inputs: Vector, weights: Matrix) -> FeedForwardResult
    {
        var previousActivations = inputs
        
        var networkInputs = Matrix()
        var networkActivations = Matrix()
        var networkActivationRates = Matrix()
        
        // Ignore the input layer as it's just a place holder.
        
        for (neuronCount, layerWeights) in zip(topology[1..<topology.count], weights)
        {
            // Append one for the bias input.
            
            var layerInputs = previousActivations
            layerInputs.append(1)
            networkInputs.append(layerInputs)
            
            let feedForward = JPSNeuralNetworkLayer.feedForward(neuronCount: neuronCount, activationFunction: activationFunction, inputs: layerInputs, weights: layerWeights)
            
            previousActivations = feedForward.activations
            
            networkActivations.append(previousActivations)
            networkActivationRates.append(feedForward.activationRates)
        }
        
        return (networkInputs, networkActivations, networkActivationRates)
    }
    
    private class func outputGradient(forCostFunction costFunction: JPSNeuralNetworkCostFunction, activations: Vector, activationRates: Vector, targetOutputs: Vector) -> Vector
    {
        var gradient = Vector()
        
        for (activationRate, (activation, targetOutput)) in zip(activationRates, zip(activations, targetOutputs))
        {
            let costRate = costFunction.derivative(OfOutput: activation, targetOutput: targetOutput)
            let error = (costRate * activationRate)
            gradient.append(error)
        }
        
        return gradient
    }
    
    private class func gradient(forCostFunction costFunction: JPSNeuralNetworkCostFunction, activations: Matrix, activationRates: Matrix, weights: Matrix, targetOutputs: Vector) -> Matrix
    {
        let reversedWeights = weights.reversed()
        var reversedActivations = (activations.reversed() as Matrix)
        var reversedActivationRates = (activationRates.reversed() as Matrix)
        
        let outputLayerActivations = reversedActivations.removeFirst()
        let outputLayerActivationRates = reversedActivationRates.removeFirst()
        var previousGradient = JPSNeuralNetwork.outputGradient(forCostFunction: costFunction, activations: outputLayerActivations, activationRates: outputLayerActivationRates, targetOutputs: targetOutputs)
        
        var gradient = Matrix()
        gradient.append(previousGradient)
        
        for (layerActivationRates, (layerActivations, layerWeights)) in zip(reversedActivationRates, zip(reversedActivations, reversedWeights))
        {
            previousGradient = JPSNeuralNetworkLayer.gradient(forActivations: layerActivations, activationRates: layerActivationRates, weights: layerWeights, gradient: previousGradient)
            
            gradient.append(previousGradient)
        }
        
        return gradient.reversed()
    }
    
    private class func update(weights: Matrix, learningRate: Float, gradient: Matrix, inputs: Matrix) -> Matrix
    {
        var newWeights = Matrix()
        
        for ((layerInputs, layerWeights), layerGradient) in zip(zip(inputs, weights), gradient)
        {
            let newLayerWeights = JPSNeuralNetworkLayer.update(weights: layerWeights, learningRate: learningRate, gradient: layerGradient, inputs: layerInputs)
            newWeights.append(newLayerWeights)
        }
        
        return newWeights
    }
    
    private class func backpropagate(learningRate: Float, costFunction: JPSNeuralNetworkCostFunction, inputs: Matrix, weights: Matrix, activations: Matrix, activationRates: Matrix, targetOutput: Vector) -> Matrix
    {
        let gradient = JPSNeuralNetwork.gradient(forCostFunction: costFunction, activations: activations, activationRates: activationRates, weights: weights, targetOutputs: targetOutput)
        
        return JPSNeuralNetwork.update(weights: weights, learningRate: learningRate, gradient: gradient, inputs: inputs)
    }
    
    public class func train(delegate: JPSNeuralNetworkDelegate?, topology: [Int], epochs: Int, learningRate: Float, activationFunction: JPSNeuralNetworkActivationFunction, costFunction: JPSNeuralNetworkCostFunction, trainingInputs: Matrix, targetOutputs: Matrix) -> Matrix
    {
        var weights = JPSNeuralNetwork.weights(forTopology: topology)
        
        for epoch in 0..<epochs
        {
            var activations = Matrix()
            
            for (index, (inputs, targetOutput)) in zip(trainingInputs, targetOutputs).enumerated()
            {
                let progress = (Scalar(index + 1) / Scalar(targetOutputs.count))
                delegate?.network(progressDidChange: progress)
                
                let overallProgress = ((Scalar(epoch) + progress) / Scalar(epochs))
                delegate?.network(overallProgressDidChange: overallProgress)
                
                let feedForward: FeedForwardResult = JPSNeuralNetwork.feedForward(topology: topology, activationFunction: activationFunction, inputs: inputs, weights: weights)
                activations.append(feedForward.activations.last!)
                
                weights = JPSNeuralNetwork.backpropagate(learningRate: learningRate, costFunction: costFunction, inputs: feedForward.inputs, weights: weights, activations: feedForward.activations, activationRates: feedForward.activationRates, targetOutput: targetOutput)
            }
            
            let cost = JPSNeuralNetwork.cost(costFunction: costFunction, activations: activations, targetOutputs: targetOutputs)
            delegate?.network(costDidChange: cost)
        }
        
        return weights
    }
}
