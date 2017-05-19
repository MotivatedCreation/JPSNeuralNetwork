//
//  TrainingViewController.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 4/4/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

import UIKit
import CorePlot


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
    @IBOutlet weak var learningRateTextField: UITextField!
    @IBOutlet weak var overallProgressView: UIProgressView!
    
    let costGraphViewControllerSegueIdentifier = "CostGraphViewControllerSegue"
    let liveTrainingViewControllerSegueIdentifier = "liveTrainingViewControllerSegue"
    
    var neuralNetwork: JPSNeuralNetwork?
    
    var epochs = 10
    let architecture = [784, 20, 10]
    var learningRate: Scalar = 0.4
    let costFunction = JPSNeuralNetworkCostFunction.meanSquared
    let activationFunctions: [JPSNeuralNetworkActivationFunction] = [.sigmoid, .hyperbolicTangent]
    
    var inputs: Matrix = [[0, 0], [1, 0], [0, 1], [1, 1]]
    var labels: Matrix = [[0], [1], [1], [0]]
    
    var testingData: (labels: [MNISTLabel], images: [MNISTImage])?
    var trainingData: (labels: [MNISTLabel], images: [MNISTImage])?
    
    var currentEpoch = 0
    var costs = Vector()
    var weights = Matrix()
    var currentCost: Float = 0
    var previousCost: Float = 0
    var startTime: TimeInterval?
    
    override func viewDidLoad()
    {
        super.viewDidLoad()
        
        self.neuralNetwork = JPSNeuralNetwork(architecture: self.architecture, activationFunctions: self.activationFunctions)
        self.neuralNetwork!.delegate = self
        self.neuralNetwork!.bias = 1
        
        self.costGraphButton.isEnabled = false
        self.liveTrainingButton.isEnabled = false
        self.resetTrainingButton.isEnabled = false
        self.cancelTrainingButton.isEnabled = false
        
        self.loadTrainingData()
    }
    
    @IBAction func beginTraining(_ sender: Any)
    {
        self.preTrainingUIUpdate()
        
        DispatchQueue.global(qos: .background).async
        {
            if (self.epochsTextField.text?.characters.count != 0) {
                self.epochs = Int((self.epochsTextField.text! as NSString).intValue)
            }
            
            if (self.learningRateTextField.text?.characters.count != 0) {
                self.learningRate = Float((self.learningRateTextField.text! as NSString).floatValue)
            }
            
            if (self.biasTextField.text?.characters.count != 0) {
                self.neuralNetwork!.bias = Float((self.biasTextField.text! as NSString).floatValue)
            }

            self.startTime = Date.timeIntervalSinceReferenceDate
            
            self.weights = self.neuralNetwork!.train(epochs: self.epochs, costFunction: self.costFunction, learningRate: self.learningRate, trainingInputs: self.trainingData!.images, targetOutputs: self.trainingData!.labels)
            self.testNetwork()
            
            DispatchQueue.main.async {
                self.postTrainingUIUpdate()
            }
        }
    }
    
    @IBAction func resetTraining(_ sender: Any)
    {
        self.neuralNetwork!.bias = 1
        self.biasTextField.text = ""
        
        self.costs = Vector()
        self.currentCost = 0
        self.previousCost = self.currentCost
        self.currentCostLabel.text = "???"
        self.currentCostLabel.textColor = UIColor.black
        
        self.epochs = 10
        self.epochsTextField.text = ""
        
        self.currentEpoch = 0
        self.currentEpochLabel.text = "Epoch: 0"
        
        self.learningRate = 0.4
        self.learningRateTextField.text = ""
        
        self.progressView.progress = 0
        self.progressLabel.text = "0 %"
        
        self.overallProgressView.progress = 0
        self.overallProgressLabel.text = "0 %"
        
        self.elapsedTimeLabel.text = "Elapsed Time: 00:00:00"
    }
    
    @IBAction func cancelTraining(_ sender: Any) {
        self.neuralNetwork!.cancelTraining()
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?)
    {
        if (segue.identifier == self.costGraphViewControllerSegueIdentifier)
        {
            let viewController = (segue.destination as! CostGraphViewController)
            viewController.epochs = self.epochs
            viewController.costs = self.costs
        }
        else if (segue.identifier == self.liveTrainingViewControllerSegueIdentifier)
        {
            let viewController = (segue.destination as! LiveTrainingViewController)
            viewController.neuralNetwork = self.neuralNetwork
        }
    }
    
    private func preTrainingUIUpdate()
    {
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
        self.trainButton.isEnabled = true
        self.costGraphButton.isEnabled = true
        self.liveTrainingButton.isEnabled = true
        self.resetTrainingButton.isEnabled = true
        self.cancelTrainingButton.isEnabled = false
    }
    
    private func testNetwork()
    {
        var outputs = Matrix()
        
        for testingImage in self.testingData!.images
        {
            let output = self.neuralNetwork!.feedForward(inputs: testingImage)
            outputs.append(output)
        }
        
        var correctPredictions = 0
        
        for (output, label) in zip(outputs, self.testingData!.labels)
        {
            let maxOutputIndex = output.index(of: output.max()!)
            let maxLabelIndex = label.index(of: label.max()!)
            
            correctPredictions += (maxOutputIndex == maxLabelIndex ? 1 : 0)
        }
        
        print(correctPredictions)
    }
}

extension TrainingViewController: JPSNeuralNetworkDelegate
{
    func network(costDidChange cost: Float)
    {
        self.costs.append(cost)
        
        self.currentEpoch += 1
        
        self.previousCost = self.currentCost
        self.currentCost = cost
        
        let deltaCost = (self.previousCost - self.currentCost)
        let sign = (deltaCost > 0 ? "-" : "+")
        
        DispatchQueue.main.async
        {
            self.currentCostLabel.text = "Cost: \(self.currentCost) (\(sign)\(abs(deltaCost)))"
            self.currentCostLabel.textColor = (self.currentCost > self.previousCost ?  UIColor.red :  UIColor(red: 0, green: (230.0 / 255.0), blue: 0, alpha: 1))
            
            self.currentEpochLabel.text = "Epoch: \(self.currentEpoch)"
        }
    }
    
    func network(progressDidChange progress: Float)
    {
        DispatchQueue.main.async
        {
            self.progressView.progress = progress
            self.progressLabel.text = "\(progress * 100) %"
        }
    }
    
    func network(overallProgressDidChange progress: Float)
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
