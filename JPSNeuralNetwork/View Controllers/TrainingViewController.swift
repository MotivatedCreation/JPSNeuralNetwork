//
//  TrainingViewController.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 4/4/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

import UIKit
import GameKit

private enum State
{
    case initial
    case loadingData
    case dataLoaded
    case training
    case finishedTraining
}

public class TrainingViewController: UIViewController
{
    @IBOutlet weak var progressLabel: UILabel!
    @IBOutlet weak var elapsedTimeLabel: UILabel!
    @IBOutlet weak var currentCostLabel: UILabel!
    @IBOutlet weak var currentEpochLabel: UILabel!
    @IBOutlet weak var biasTextField: UITextField!
    @IBOutlet weak var progressView: UIProgressView!
    @IBOutlet weak var liveTrainingButton: UIButton!
    @IBOutlet weak var epochsTextField: UITextField!
    @IBOutlet weak var overallProgressLabel: UILabel!
    @IBOutlet weak var momentumTextField: UITextField!
    @IBOutlet weak var loadTrainingDataButton: UIButton!
    @IBOutlet weak var learningRateTextField: UITextField!
    @IBOutlet weak var architectureTextField: UITextField!
    @IBOutlet weak var overallProgressView: UIProgressView!
    
    fileprivate let graphViewControllerSegueIdentifier = "GraphViewControllerSegue"
    fileprivate let customDataViewControllerSegueIdentifier = "CustomDataViewControllerSegue"
    fileprivate let liveTrainingViewControllerSegueIdentifier = "LiveTrainingViewControllerSegue"
    
    fileprivate let errorFunction = JPSNeuralNetworErrorFunction.meanSquared
    fileprivate let activationFunctions: [JPSNeuralNetworkActivationFunction] = [.sigmoid, .sigmoid]
    
    fileprivate var state: State = .initial {
        didSet { self.setUI(forState: self.state) }
    }
    
    fileprivate var epochs = 10
    fileprivate var momentum: Scalar = 0.1
    fileprivate var architecture = [784, 20, 10]
    fileprivate var learningRate: Scalar = 0.9
    fileprivate var recreationNetwork: JPSNeuralNetwork?
    fileprivate var classificationNetwork: JPSNeuralNetwork?
    
    fileprivate var testingInputDataset: Matrix?
    fileprivate var testingLabelDataset: Matrix?
    fileprivate var trainingInputDataset: Matrix?
    fileprivate var trainingLabelDataset: Matrix?
    
    fileprivate var currentEpoch = 0
    fileprivate var costs = Vector()
    fileprivate var currentError: Float = 0
    fileprivate var previousError: Float = 0
    fileprivate var startTime: TimeInterval?
    
    fileprivate var accuracy = 0.0
    fileprivate var numberOfLabels = 0
    fileprivate var numberOfCorrectPredictions = 0
    
    fileprivate var beginTrainingBarButton: UIBarButtonItem!
    fileprivate var trainingGraphBarButton: UIBarButtonItem!
    fileprivate var resetTrainingBarButton: UIBarButtonItem!
    fileprivate var cancelTrainingBarButton: UIBarButtonItem!
    fileprivate var trainingResultsBarButton: UIBarButtonItem!
    
    override public func viewDidLoad()
    {
        super.viewDidLoad()
        
        self.setupToolbar()
        self.setupProperties()
    }
    
    override public func prepare(for segue: UIStoryboardSegue, sender: Any?)
    {
        if segue.identifier == self.graphViewControllerSegueIdentifier
        {
            let viewController = segue.destination as! CostGraphViewController
            viewController.independentVariables = self.costs
            viewController.dependentVariables = Array<Int>(0..<self.currentEpoch).map({ epoch -> Scalar in return Scalar(epoch) / Scalar(self.epochs) })
        }
        else if segue.identifier == self.liveTrainingViewControllerSegueIdentifier
        {
            let viewController = segue.destination as! LiveTrainingViewController
            viewController.momentum = self.momentum
            viewController.learningRate = self.learningRate
            viewController.architecture = self.architecture
            viewController.errorFunction = self.errorFunction
            viewController.activationFunctions = self.activationFunctions
            viewController.recreationWeights = self.recreationNetwork?.weights
            viewController.classificationWeights = self.classificationNetwork?.weights
        }
    }
}

extension TrainingViewController: UINavigationControllerDelegate
{
    public func navigationControllerSupportedInterfaceOrientations(_ navigationController: UINavigationController) -> UIInterfaceOrientationMask {
        return .portrait
    }
}

extension TrainingViewController
{
    public func setupProperties()
    {
        self.state = .initial
        
        self.navigationController?.delegate = self
        
        self.buildNetwork(withArchitecture: self.architecture)
    }
    
    public func setupToolbar()
    {
        let beginTrainingButton = UIButton(type: .custom)
        beginTrainingButton.setImage(UIImage(named: "train"), for: .normal)
        beginTrainingButton.addTarget(self, action: #selector(TrainingViewController.beginTraining(_:)), for: .touchUpInside)
        beginTrainingButton.sizeToFit()
        self.beginTrainingBarButton = UIBarButtonItem(customView: beginTrainingButton)
        
        let trainingResultsButton = UIButton(type: .custom)
        trainingResultsButton.setImage(UIImage(named: "results"), for: .normal)
        trainingResultsButton.addTarget(self, action: #selector(TrainingViewController.showResults(_:)), for: .touchUpInside)
        trainingResultsButton.sizeToFit()
        self.trainingResultsBarButton = UIBarButtonItem(customView: trainingResultsButton)
        
        let trainingGraphButton = UIButton(type: .custom)
        trainingGraphButton.setImage(UIImage(named: "line-chart"), for: .normal)
        trainingGraphButton.addTarget(self, action: #selector(TrainingViewController.showTrainingGraph(_:)), for: .touchUpInside)
        trainingGraphButton.sizeToFit()
        self.trainingGraphBarButton = UIBarButtonItem(customView: trainingGraphButton)
        
        let resetTrainingButton = UIButton(type: .custom)
        resetTrainingButton.setImage(UIImage(named: "reset"), for: .normal)
        resetTrainingButton.addTarget(self, action: #selector(TrainingViewController.resetTraining(_:)), for: .touchUpInside)
        resetTrainingButton.sizeToFit()
        self.resetTrainingBarButton = UIBarButtonItem(customView: resetTrainingButton)
        
        let cancelTrainingButton = UIButton(type: .custom)
        cancelTrainingButton.setImage(UIImage(named: "stop"), for: .normal)
        cancelTrainingButton.addTarget(self, action: #selector(TrainingViewController.cancelTraining(_:)), for: .touchUpInside)
        cancelTrainingButton.sizeToFit()
        self.cancelTrainingBarButton = UIBarButtonItem(customView: cancelTrainingButton)
        
        let flexibleSpaceItem = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)
        
        self.setToolbarItems([
            
            self.beginTrainingBarButton,
            flexibleSpaceItem,
            self.trainingResultsBarButton,
            flexibleSpaceItem,
            self.trainingGraphBarButton,
            flexibleSpaceItem,
            self.resetTrainingBarButton,
            flexibleSpaceItem,
            self.cancelTrainingBarButton
            
        ], animated: false)
    }
}

extension TrainingViewController
{
    fileprivate func setUI(forState state: State)
    {
        switch state
        {
        case .initial:
            self.setToolbarEnabled(false)
            self.liveTrainingButton.isEnabled = true
            self.loadTrainingDataButton.isEnabled = true
            
        case .loadingData:
            self.setToolbarEnabled(false)
            
        case .dataLoaded:
            self.setToolbarEnabled(true)
            self.loadTrainingDataButton.isEnabled = true
            self.cancelTrainingBarButton.isEnabled = false
            
        case .training:
            self.setToolbarEnabled(false)
            self.cancelTrainingBarButton.isEnabled = true
            
        case .finishedTraining:
            self.setToolbarEnabled(true)
            self.liveTrainingButton.isEnabled = true
            self.loadTrainingDataButton.isEnabled = true
            self.cancelTrainingBarButton.isEnabled = false
        }
    }
    
    fileprivate func setToolbarEnabled(_ flag: Bool)
    {
        self.biasTextField.isEnabled = flag
        self.epochsTextField.isEnabled = flag
        self.momentumTextField.isEnabled = flag
        self.learningRateTextField.isEnabled = flag
        self.architectureTextField.isEnabled = flag
        
        self.liveTrainingButton.isEnabled = flag
        self.loadTrainingDataButton.isEnabled = flag
        self.beginTrainingBarButton.isEnabled = flag
        self.trainingGraphBarButton.isEnabled = flag
        self.resetTrainingBarButton.isEnabled = flag
        self.cancelTrainingBarButton.isEnabled = flag
        self.trainingResultsBarButton.isEnabled = flag
    }
    
    fileprivate func buildNetwork(withArchitecture architecture: [Int])
    {
        self.classificationNetwork = JPSNeuralNetwork()
        self.classificationNetwork!.delegate = self
        self.classificationNetwork!.bias = 1
        self.classificationNetwork!.weights = JPSNeuralNetwork.weights(forArchitecture: self.architecture)
        
        self.recreationNetwork = JPSNeuralNetwork()
        self.recreationNetwork!.bias = 1
        self.recreationNetwork!.weights = JPSNeuralNetwork.weights(forArchitecture: self.architecture.reversed())
    }
    
    fileprivate func loadDatasets(_ handler: @escaping () -> ())
    {
        self.state = .loadingData
        
        self.testingInputDataset = nil
        self.testingLabelDataset = nil
        self.trainingInputDataset = nil
        self.trainingLabelDataset = nil
        
        let previousLeftBarButtonItem = self.navigationItem.leftBarButtonItem
        
        let activityIndicator = UIActivityIndicatorView(activityIndicatorStyle: .gray)
        self.navigationItem.leftBarButtonItem = UIBarButtonItem(customView: activityIndicator)
        activityIndicator.startAnimating()
        
        DispatchQueue.global(qos: .background).async { [unowned self] in
            
            handler()
            
            DispatchQueue.main.async {
                self.state = .dataLoaded
                self.navigationItem.leftBarButtonItem = previousLeftBarButtonItem
            }
        }
    }
    
    fileprivate func loadXORTrainingDataset()
    {
        self.loadDatasets { [unowned self] in
            
            self.trainingInputDataset = [[0, 0], [1, 0], [0, 1], [1, 1]]
            self.trainingLabelDataset = [[0], [1], [1], [0]]
            
            self.testingInputDataset = self.trainingLabelDataset
            self.testingLabelDataset = self.trainingLabelDataset
            
            DispatchQueue.main.async { self.architectureTextField.text = "2/3/1" }
        }
    }
    
    fileprivate func loadMNISTTrainingDataset()
    {
        let dataLoader = JPSMNISTDataLoader()
        
        self.loadDatasets { [unowned self] in
            
            do
            {
                let trainingData = try dataLoader.loadTrainingData()
                self.trainingInputDataset = trainingData.images
                self.trainingLabelDataset = trainingData.labels
                
                let testingData = try dataLoader.loadTestingData()
                self.testingInputDataset = testingData.images
                self.testingLabelDataset = testingData.labels
                
                DispatchQueue.main.async { self.architectureTextField.text = "784/20/10" }
            }
            catch { print(error) }
        }
    }
    
    fileprivate func testNetwork()
    {
        var outputs = Matrix()
        
        for inputs in self.testingInputDataset!
        {
            let output: Vector = self.classificationNetwork!.feedForward(architecture: self.architecture, activationFunctions: self.activationFunctions, inputs: inputs)
            outputs.append(output)
        }
        
        var correctPredictions = 0
        
        for (output, label) in zip(outputs, self.testingLabelDataset!)
        {
            let maxOutputIndex = output.index(of: output.max()!)
            let maxLabelIndex = label.index(of: label.max()!)
            
            correctPredictions += (maxOutputIndex == maxLabelIndex ? 1 : 0)
        }
        
        self.numberOfLabels = self.testingLabelDataset!.count
        self.numberOfCorrectPredictions = correctPredictions
        self.accuracy = Double(Scalar(correctPredictions) / Scalar(self.testingLabelDataset!.count) * Float(100))
    }
}

extension TrainingViewController
{
    @IBAction func selectTrainingDataset(_ sender: Any)
    {
        let alertController = UIAlertController(title: "Select the Training Dataset", message: nil, preferredStyle: .actionSheet)
        
        let customDatasetAction = UIAlertAction(title: "Custom Dataset", style: .default) { action in
            
            self.performSegue(withIdentifier: self.customDataViewControllerSegueIdentifier, sender: self)
        }
        alertController.addAction(customDatasetAction)
        
        let xorDatasetAction = UIAlertAction(title: "XOR Datasets", style: .default) { action in
            
            self.loadXORTrainingDataset()
        }
        alertController.addAction(xorDatasetAction)
        
        let mnistDatasetAction = UIAlertAction(title: "MNIST Datasets", style: .default) { action in
            
            self.loadMNISTTrainingDataset()
        }
        alertController.addAction(mnistDatasetAction)
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel)
        alertController.addAction(cancelAction)
        
        self.present(alertController, animated: true, completion: nil)
    }
}

extension TrainingViewController
{
    @objc fileprivate func showTrainingGraph(_ sender: UIButton) {
        self.performSegue(withIdentifier: self.graphViewControllerSegueIdentifier, sender: self)
    }
    
    @objc fileprivate func beginTraining(_ sender: UIButton)
    {
        guard let inputs = self.trainingInputDataset else {
            
            let alertController = UIAlertController(title: "Error", message: "Please select the Training Dataset.", preferredStyle: .alert)
            
            let okayAction = UIAlertAction(title: "Okay", style: .cancel)
            alertController.addAction(okayAction)
            
            self.present(alertController, animated: true, completion: nil)
            
            return
        }
        
        guard let labels = self.trainingLabelDataset else {
            
            let alertController = UIAlertController(title: "Error", message: "Please select the Training Dataset.", preferredStyle: .alert)
            
            let okayAction = UIAlertAction(title: "Okay", style: .cancel)
            alertController.addAction(okayAction)
            
            self.present(alertController, animated: true, completion: nil)
            
            return
        }
        
        self.state = .training
        
        if let text = self.architectureTextField.text, text.characters.count != 0
        {
            self.architecture = text.components(separatedBy: "/").map({ value -> Int in
                return Int(value)!
            })
            
            self.classificationNetwork!.weights = JPSNeuralNetwork.weights(forArchitecture: self.architecture)
            self.recreationNetwork!.weights = JPSNeuralNetwork.weights(forArchitecture: self.architecture.reversed())
        }
        
        if let text = self.epochsTextField.text, text.characters.count != 0, let epoch = Int(text) {
            self.epochs = epoch
        }
        
        if let text = self.learningRateTextField.text, text.characters.count != 0, let learningRate = Scalar(text) {
            self.learningRate = learningRate
        }
        
        if let text = self.momentumTextField.text, text.characters.count != 0, let momentum = Scalar(text) {
            self.momentum = momentum
        }
        
        if let text = self.biasTextField.text, text.characters.count != 0, let bias = Scalar(text)
        {
            self.classificationNetwork!.bias = bias
            self.recreationNetwork!.bias = bias
        }
        
        self.startTime = Date.timeIntervalSinceReferenceDate
        
        DispatchQueue.global(qos: .background).async {
            do
            {
                try self.classificationNetwork!.train(epochs: self.epochs, architecture: self.architecture, activationFunctions: self.activationFunctions, errorFunction: self.errorFunction, learningRate: self.learningRate, momentum: self.momentum, trainingInputs: inputs, targetOutputs: labels)
                
                try self.recreationNetwork!.train(epochs: self.epochs, architecture: self.architecture.reversed(), activationFunctions: self.activationFunctions.reversed(), errorFunction: self.errorFunction, learningRate: self.learningRate, momentum: self.momentum, trainingInputs: labels, targetOutputs: inputs)
                
                DispatchQueue.main.async { self.state = .finishedTraining }
            }
            catch {
                DispatchQueue.main.async {
                    self.state = .dataLoaded
                    
                    let alertController = UIAlertController(title: (error as NSError).domain, message: (error as NSError).localizedFailureReason, preferredStyle: .alert)
                    
                    let okayAction = UIAlertAction(title: "Okay", style: .cancel)
                    alertController.addAction(okayAction)
                    
                    self.present(alertController, animated: true, completion: nil)
                }
            }
        }
    }
    
    @objc fileprivate func resetTraining(_ sender: UIButton)
    {
        self.state = .dataLoaded
        
        self.costs = Vector()
        self.currentError = 0
        self.previousError = self.currentError
        self.currentCostLabel.text = "Error: ???"
        self.currentCostLabel.textColor = UIColor.black
        
        self.currentEpoch = 0
        self.currentEpochLabel.text = "Epoch: 0"
        
        self.progressView.progress = 0
        self.progressLabel.text = "0 %"
        
        self.overallProgressView.progress = 0
        self.overallProgressLabel.text = "0 %"
        
        self.elapsedTimeLabel.text = "Elapsed Time: 00:00:00"
    }
    
    @objc fileprivate func cancelTraining(_ sender: UIButton)
    {
        self.state = .finishedTraining
        self.classificationNetwork!.cancelTraining()
        self.recreationNetwork!.cancelTraining()
    }
    
    @objc fileprivate func showResults(_ sender: UIButton)
    {
        let previousBarButtonItem = self.toolbarItems?[2]
        
        let activityIndicator = UIActivityIndicatorView(activityIndicatorStyle: .gray)
        activityIndicator.startAnimating()
        self.toolbarItems?[2] = UIBarButtonItem(customView: activityIndicator)
        
        DispatchQueue.global(qos: .background).async {
            
            self.testNetwork()
            
            DispatchQueue.main.async {
                
                self.toolbarItems?[2] = previousBarButtonItem!
                
                let message = "Accuracy: \(Scalar(self.numberOfCorrectPredictions) / Scalar(self.numberOfLabels) * 100) %"
                    + "\nNumber of Labels: \(self.numberOfLabels)"
                    + "\nNumber of Correct Predictions: \(self.numberOfCorrectPredictions)"
                
                let alertController = UIAlertController(title: "Results", message: message, preferredStyle: .alert)
                
                let okayAction = UIAlertAction(title: "Okay", style: .cancel, handler: nil)
                alertController.addAction(okayAction)
                
                self.present(alertController, animated: true, completion: nil)
            }
        }
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
    public func network(_ network: JPSNeuralNetwork, errorDidChange error: Scalar)
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
    
    public func network(_ network: JPSNeuralNetwork, progressDidChange progress: Float)
    {
        DispatchQueue.main.async
        {
            self.progressView.progress = progress
            self.progressLabel.text = "\(progress * 100) %"
        }
    }
    
    public func network(_ network: JPSNeuralNetwork, overallProgressDidChange progress: Float)
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
