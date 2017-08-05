//
//  JPSNeuralNetworkLayer.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 4/4/17.
//

import Foundation
import Accelerate


/*
    NOTE:
 
    The majority of these functions are class (static) functions to retain the notation of "pure" functions. This is useful for parallel processing
    and is beneficial by knowing the exact state at each step.
*/

public class JPSNeuralNetworkLayer { }

/*
    Helper Functions
*/
extension JPSNeuralNetworkLayer
{
    /**
        Used to generate a random weights for all neurons.
    */
    public class func randomWeights(neuronCount: Int, inputCount: Int) -> Vector
    {
        let layerWeights = (0..<neuronCount).map({ _ in
            return JPSNeuralNetworkNeuron.randomWeights(inputCount: inputCount)
        })
        
        return Vector(layerWeights.joined())
    }
}

/*
    Feed Forward Functions
*/
extension JPSNeuralNetworkLayer
{
    /**
        Used to calculate the weighted input.
     
        Equations:
        weightedInput = sum(x[i] * w[i])
    */
    public class func weightedInputs(neuronCount: Int, inputs: Vector, weights: Vector) -> Vector
    {
        var weightedInputs = Vector(repeating: 0, count: neuronCount)
        
        vDSP_mmul(weights, 1,
                  inputs, 1,
                  &weightedInputs, 1,
                  vDSP_Length(neuronCount), 1,
                  vDSP_Length(inputs.count))
        
        return weightedInputs
    }
    
    /**
        Used to feed the inputs and weights forward and calculate the weighted input and activation.
        This method also precalculates the activation rate for use later on and to reduce the number of
        calculations.
     
        Equations:
        weightedInput = sum(x[i] * w[i])
        activation = sigma(weightedInput[j])
        activationRate = sigma'(activation[j])
    */
    public class func feedForward(neuronCount: Int, activationFunction: JPSNeuralNetworkActivationFunction, inputs: Vector, weights: Vector) -> Vector
    {
        let weightedInputs = JPSNeuralNetworkLayer.weightedInputs(neuronCount: neuronCount, inputs: inputs, weights: weights)
        let activations = activationFunction.activations(weightedInputs)
        
        return activations
    }
}

/*
    Training Functions
*/
extension JPSNeuralNetworkLayer
{
    /**
        Used to calculate the error gradient for each neuron.
    */
    public class func gradient(forActivationGradient activationGradient: Vector, weights: Vector, gradient: Vector) -> Vector
    {
        var layerGradient = Vector(repeating: 0, count: activationGradient.count)
        
        var weights = weights
        weights.removeLast()
        
        vDSP_mmul(gradient, 1,
                  weights, 1,
                  &layerGradient, 1,
                  1, vDSP_Length(layerGradient.count),
                  vDSP_Length(gradient.count))
        
        vDSP_vmul(layerGradient, 1,
                  activationGradient, 1,
                  &layerGradient, 1,
                  vDSP_Length(layerGradient.count))
        
        return layerGradient
    }
    
    /**
        Used to update each neurons weights on a per neuron error basis given the input.
    */
    public class func update(weights: Vector, learningRate: Float, momentum: Float, gradient: Vector, inputs: Vector, previousWeights: Vector) -> Vector
    {
        var deltaWeights = Vector(repeating: 0, count: weights.count)
        vDSP_vsub(weights, 1,
                  previousWeights, 1,
                  &deltaWeights, 1,
                  vDSP_Length(weights.count))
        
        var momentum = momentum
        var scaledDeltaWeights = Vector(repeating: 0, count: deltaWeights.count)
        
        vDSP_vsmul(deltaWeights, 1,
                   &momentum,
                   &scaledDeltaWeights, 1,
                   vDSP_Length(deltaWeights.count))
        
        var negativeLearningRate = -learningRate
        var scaledGradient = Vector(repeating: 0, count: gradient.count)
        
        vDSP_vsmul(gradient, 1,
                   &negativeLearningRate,
                   &scaledGradient, 1,
                   vDSP_Length(gradient.count))
        
        var scaledInputs = Vector(repeating: 0, count: weights.count)
        
        vDSP_mmul(scaledGradient, 1,
                  inputs, 1,
                  &scaledInputs, 1,
                  vDSP_Length(scaledGradient.count), vDSP_Length(inputs.count),
                  1)
        
        var layerWeights = Vector(repeating: 0, count: weights.count)
        
        vDSP_vadd(weights, 1,
                  scaledInputs, 1,
                  &layerWeights, 1,
                  vDSP_Length(weights.count))
        
        vDSP_vadd(scaledDeltaWeights, 1,
                  layerWeights, 1,
                  &layerWeights, 1,
                  vDSP_Length(layerWeights.count))
        
        return layerWeights
    }
}
