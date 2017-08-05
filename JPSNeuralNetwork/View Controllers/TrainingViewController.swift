//
//  TrainingViewController.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 4/4/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

import UIKit


class TrainingViewController: UIViewController
{
    @IBOutlet weak var trainButton: UIButton!
    @IBOutlet weak var progressLabel: UILabel!
    @IBOutlet weak var elapsedTimeLabel: UILabel!
    @IBOutlet weak var currentCostLabel: UILabel!
    @IBOutlet weak var costGraphButton: UIButton!
    @IBOutlet weak var currentEpochLabel: UILabel!
    @IBOutlet weak var biasTextField: UITextField!
    @IBOutlet weak var progressView: UIProgressView!
    @IBOutlet weak var liveTrainingButton: UIButton!
    @IBOutlet weak var epochsTextField: UITextField!
    @IBOutlet weak var overallProgressLabel: UILabel!
    @IBOutlet weak var resetTrainingButton: UIButton!
    @IBOutlet weak var cancelTrainingButton: UIButton!
    @IBOutlet weak var momentumTextField: UITextField!
    @IBOutlet weak var learningRateTextField: UITextField!
    @IBOutlet weak var overallProgressView: UIProgressView!
    
    var inputs: Matrix = [[0, 0], [1, 0], [0, 1], [1, 1]]
    var labels: Matrix = [[0], [1], [1], [0]]
    
    let costGraphViewControllerSegueIdentifier = "CostGraphViewControllerSegue"
    let liveTrainingViewControllerSegueIdentifier = "LiveTrainingViewControllerSegue"
    
    let architecture = [784, 20, 10]
    let errorFunction = JPSNeuralNetworErrorFunction.meanSquared
    let activationFunctions: [JPSNeuralNetworkActivationFunction] = [.sigmoid, .sigmoid]
    
    var epochs = 10
    var momentum: Scalar = 0.1
    var learningRate: Scalar = 0.9
    var neuralNetwork: JPSNeuralNetwork?
    var neuralNetwork2: JPSNeuralNetwork?
    var testingData: (labels: [MNISTLabel], images: [MNISTImage])?
    var trainingData: (labels: [MNISTLabel], images: [MNISTImage])?
    
    var currentEpoch = 0
    var costs = Vector()
    var currentError: Float = 0
    var previousError: Float = 0
    var startTime: TimeInterval?
    
    override func viewDidLoad()
    {
        super.viewDidLoad()
        
        self.neuralNetwork = JPSNeuralNetwork(architecture: self.architecture, activationFunctions: self.activationFunctions)
        self.neuralNetwork!.delegate = self
        self.neuralNetwork!.bias = 1
        
        self.neuralNetwork2 = JPSNeuralNetwork(architecture: self.architecture.reversed(), activationFunctions: self.activationFunctions.reversed())
        self.neuralNetwork2!.delegate = self
        self.neuralNetwork2!.bias = 1
        
        self.costGraphButton.isEnabled = false
        self.resetTrainingButton.isEnabled = false
        self.cancelTrainingButton.isEnabled = false
        
        self.loadTrainingData()
    }
    
    @IBAction func beginTraining(_ sender: Any)
    {
        self.preTrainingUIUpdate()
        
        if (self.epochsTextField.text?.characters.count != 0) {
            self.epochs = Int((self.epochsTextField.text! as NSString).intValue)
        }
        
        if (self.learningRateTextField.text?.characters.count != 0) {
            self.learningRate = Scalar((self.learningRateTextField.text! as NSString).floatValue)
        }
        
        if (self.momentumTextField.text?.characters.count != 0) {
            self.momentum = Scalar((self.momentumTextField.text! as NSString).floatValue)
        }
        
        if (self.biasTextField.text?.characters.count != 0) {
            self.neuralNetwork!.bias = Scalar((self.biasTextField.text! as NSString).floatValue)
            self.neuralNetwork2!.bias = Scalar((self.biasTextField.text! as NSString).floatValue)
        }
        
        self.startTime = Date.timeIntervalSinceReferenceDate
        
        DispatchQueue.global(qos: .background).async
        {
            do
            {
                let inputs = Matrix(self.trainingData!.images[0..<30000])
                let labels = Matrix(self.trainingData!.labels[0..<30000])
                
                try self.neuralNetwork!.train(epochs: self.epochs, errorFunction: self.errorFunction, learningRate: self.learningRate, momentum: self.momentum, trainingInputs: inputs, targetOutputs: labels)
                try self.neuralNetwork2!.train(epochs: self.epochs, errorFunction: self.errorFunction, learningRate: self.learningRate, momentum: self.momentum, trainingInputs: labels, targetOutputs: inputs)
                self.testNetwork()
            }
            catch { print(error) }
            
            DispatchQueue.main.async {
                self.postTrainingUIUpdate()
            }
        }
    }
    
    @IBAction func resetTraining(_ sender: Any)
    {
        self.neuralNetwork!.bias = 1
        self.neuralNetwork2!.bias = 1
        self.biasTextField.text = ""
        
        self.costs = Vector()
        self.currentError = 0
        self.previousError = self.currentError
        self.currentCostLabel.text = "Error: ???"
        self.currentCostLabel.textColor = UIColor.black
        
        self.epochs = 10
        self.epochsTextField.text = ""
        
        self.currentEpoch = 0
        self.currentEpochLabel.text = "Epoch: 0"
        
        self.learningRate = 0.9
        self.learningRateTextField.text = ""
        
        self.momentum = 0.1
        self.momentumTextField.text = ""
        
        self.progressView.progress = 0
        self.progressLabel.text = "0 %"
        
        self.overallProgressView.progress = 0
        self.overallProgressLabel.text = "0 %"
        
        self.elapsedTimeLabel.text = "Elapsed Time: 00:00:00"
    }
    
    @IBAction func cancelTraining(_ sender: Any) {
        self.neuralNetwork!.cancelTraining()
        self.neuralNetwork2!.cancelTraining()
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?)
    {
        if (segue.identifier == self.costGraphViewControllerSegueIdentifier)
        {
            let viewController = (segue.destination as! CostGraphViewController)
            
            viewController.independentVariables = self.costs
            viewController.dependentVariables = Array<Int>(0..<self.currentEpoch).map({ epoch -> Scalar in return Scalar(epoch) / Scalar(self.epochs) })
        }
        else if (segue.identifier == self.liveTrainingViewControllerSegueIdentifier)
        {
            let viewController = (segue.destination as! LiveTrainingViewController)
            viewController.momentum = self.momentum
            viewController.learningRate = self.learningRate
            viewController.architecture = self.architecture
            viewController.errorFunction = self.errorFunction
            viewController.weights = self.neuralNetwork?.weights
            viewController.weights2 = self.neuralNetwork2?.weights
            viewController.activationFunctions = self.activationFunctions
        }
    }
    
    private func preTrainingUIUpdate()
    {
        self.biasTextField.isEnabled = false
        self.epochsTextField.isEnabled = false
        self.momentumTextField.isEnabled = false
        self.learningRateTextField.isEnabled = false
        
        self.trainButton.isEnabled = false
        self.costGraphButton.isEnabled = false
        self.liveTrainingButton.isEnabled = false
        self.resetTrainingButton.isEnabled = false
        self.cancelTrainingButton.isEnabled = true
    }
    
    private func loadTrainingData()
    {
        let dataLoader = JPSMNISTDataLoader()
        
        do
        {
            self.testingData = try dataLoader.loadTestingData()
            self.trainingData = try dataLoader.loadTrainingData()
        }
        catch { print(error) }
    }
    
    private func postTrainingUIUpdate()
    {
        self.biasTextField.isEnabled = true
        self.epochsTextField.isEnabled = true
        self.momentumTextField.isEnabled = true
        self.learningRateTextField.isEnabled = true
        
        self.trainButton.isEnabled = true
        self.costGraphButton.isEnabled = true
        self.liveTrainingButton.isEnabled = true
        self.resetTrainingButton.isEnabled = true
        self.cancelTrainingButton.isEnabled = false
    }
    
    private func testNetwork()
    {
        let testingImages = self.trainingData!.images[30000..<60000]
        let testingLabels = self.trainingData!.labels[30000..<60000]
        
        var outputs = Matrix()
        
        for testingImage in testingImages
        {
            let output: Vector = self.neuralNetwork!.feedForward(inputs: testingImage)
            outputs.append(output)
        }
        
        var correctPredictions = 0
        
        for (output, label) in zip(outputs, testingLabels)
        {
            let maxOutputIndex = output.index(of: output.max()!)
            let maxLabelIndex = label.index(of: label.max()!)
            
            correctPredictions += (maxOutputIndex == maxLabelIndex ? 1 : 0)
        }
        
        print("Accuracy: \(Scalar(correctPredictions) / Scalar(testingLabels.count) * 100) %")
        print("Number of Labels: \(testingLabels.count)")
        print("Number of Correct Predictions: \(correctPredictions)")
    }
}

extension TrainingViewController: UITextFieldDelegate
{
    public func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        return textField.resignFirstResponder()
    }
}

extension TrainingViewController: JPSNeuralNetworkDelegate
{
    func network(_ network: JPSNeuralNetwork, errorDidChange error: Scalar)
    {
        self.costs.append(error)
        
        self.currentEpoch += 1
        
        self.previousError = self.currentError
        self.currentError = error
        
        let deltaError = (self.previousError - self.currentError)
        let sign = (deltaError > 0 ? "-" : "+")
        
        DispatchQueue.main.async
        {
            self.currentCostLabel.text = "Error: \(self.currentError) (\(sign)\(abs(deltaError)))"
            self.currentCostLabel.textColor = (self.currentError > self.previousError ?  UIColor.red :  UIColor(red: 0, green: (230.0 / 255.0), blue: 0, alpha: 1))
            
            self.currentEpochLabel.text = "Epoch: \(self.currentEpoch)"
        }
    }
    
    func network(_ network: JPSNeuralNetwork, progressDidChange progress: Float)
    {
        DispatchQueue.main.async
        {
            self.progressView.progress = progress
            self.progressLabel.text = "\(progress * 100) %"
        }
    }
    
    func network(_ network: JPSNeuralNetwork, overallProgressDidChange progress: Float)
    {
        DispatchQueue.main.async
        {
            self.overallProgressView.progress = progress
            self.overallProgressLabel.text = "\(progress * 100) %"
            
            let endTime = Date.timeIntervalSinceReferenceDate
            let elapsedTime = (endTime - self.startTime!)
            
            self.elapsedTimeLabel.text = "Elapsed Time: \(elapsedTime.stringFromTimeInterval())"
        }
    }
}
