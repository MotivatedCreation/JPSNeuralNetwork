//
//  LiveTrainingViewController.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 5/11/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

import Foundation
import Photos
import Accelerate

public class LiveTrainingViewController: UIViewController
{
    internal enum ColorSpace: Int
    {
        case grayScale
        case infer
        
        internal func pixels(forImage image: UIImage) -> [UInt8]?
        {
            guard let cgImage = image.cgImage else { return nil }
            
            switch self
            {
            case .grayScale:
                return cgImage.grayScalePixels()
                
            case .infer:
                return cgImage.pixels()
            }
        }
        
        internal func image(forPixels pixels: [UInt8], referenceImage: UIImage) -> UIImage?
        {
            guard let cgImage = referenceImage.cgImage else { return nil }
            
            switch self
            {
            case .grayScale:
                guard let imageForPixels = cgImage.grayScaleImage(fromGrayScalePixels: pixels) else { return nil }
                
                return UIImage(cgImage: imageForPixels)
                
            case .infer:
                guard let imageForPixels = cgImage.image(fromPixels: pixels) else { return nil }
                
                return UIImage(cgImage: imageForPixels)
            }
        }
    }
    
    @IBOutlet weak internal var errorLabel: UILabel!
    @IBOutlet weak internal var predictionLabel: UILabel!
    @IBOutlet weak internal var canvasView: JPSCanvasView!
    @IBOutlet weak internal var numberOfOutputsTextField: UITextField!
    @IBOutlet weak internal var correctPredicationTextField: UITextField!
    
    public var weights: Matrix?
    public var weights2: Matrix?
    public var architecture: [Int]?
    public var momentum: Scalar = 0.1
    public var learningRate: Scalar = 0.9
    public var errorFunction: JPSNeuralNetworErrorFunction?
    public var activationFunctions: [JPSNeuralNetworkActivationFunction]?

    internal var imagePixels: Vector?
    internal var resizedImage: UIImage?
    internal var currentError: Float = 0
    internal var previousError: Float = 0
    internal var predictionInverse: Vector?
    internal var animationPixels = Matrix()
    internal var colorSpace = ColorSpace.grayScale
    internal var recreationNetwork: JPSNeuralNetwork?
    internal var classificationNetwork: JPSNeuralNetwork?
    internal let resizedImageSize = CGSize(width: 28, height: 28)
    
    private var prediction: Vector?
    
    override public func viewDidLoad()
    {
        super.viewDidLoad()
        
        self.canvasView.delegate = self
        self.canvasView.layer.cornerRadius = 5
        self.canvasView.layer.masksToBounds = true
        self.canvasView.backgroundColor = UIColor.black
        
        self.correctPredicationTextField.delegate = self
        
        self.buildNetworks(withArchitecture: self.architecture!)
    }
    
    deinit { print("(\((#file as NSString).lastPathComponent) \(#function))") }
    
    private class func normalizedPixels(forImage image: UIImage, colorSpace: ColorSpace) -> Vector?
    {
        guard let resizedImagePixels = colorSpace.pixels(forImage: image) else {
            
            print("Could not load resize image pixels.")
            
            return nil
        }
        
        let maximumComponentValue = Scalar(UInt8.max)
        
        return resizedImagePixels.flatMap({ Scalar($0) / maximumComponentValue })
    }
    
    internal func printNumber(rowSize: Int, pixels: [UInt8])
    {
        let cgImage = self.colorSpace.image(forPixels: pixels, referenceImage: self.resizedImage!)?.cgImage
        self.canvasView.layer.contents = cgImage
    }
    
    internal func feedPixelsForward(inputs: Vector)
    {
        self.prediction = self.classificationNetwork!.feedForward(inputs: inputs)
        self.predictionInverse = self.recreationNetwork!.feedForward(inputs: self.prediction!)
    }
    
    internal func predict(image: UIImage)
    {
        self.imagePixels = LiveTrainingViewController.normalizedPixels(forImage: image, colorSpace: self.colorSpace)
        self.predictionLabel.text = "Prediction: \(self.predictClassification())"
    }
    
    internal func architectureForOutputs() -> [Int]?
    {
        guard let numberOfOutputs = Int(self.numberOfOutputsTextField.text!) else {
            
            print("Incorrect number of outputs.")
            
            return nil
        }
        
        var architecture = self.architecture!
        architecture[architecture.count - 1] = numberOfOutputs
        
        return architecture
    }
    
    internal func canChangeNumberOfOutputs() -> Bool
    {
        guard let architecture = self.architectureForOutputs() else {
            
            print("Can not resize architecture output layer.")
            
            return false
        }
        
        let correctPrediction = self.correctPredicationTextField.text
        
        return (correctPrediction == nil || correctPrediction?.characters.count == 0) || (architecture.last! > Int(correctPrediction ?? String(-1))!)
    }
    
    internal func changeNumberOfOutputs()
    {
        guard let architecture = self.architectureForOutputs() else {
            
            print("Could not resize architecture output layer.")
            
            return
        }
        
        self.buildNetworks(withArchitecture: architecture)
    }
    
    internal func targetOutputsForCorrectPredication() -> Vector?
    {
        guard let correctPredication = Int(self.correctPredicationTextField.text!) else {
            
            print("Could not load correct prediction.")
            
            return nil
        }
        
        var targetOutputs = Vector(repeating: 0, count: self.architecture!.last!)
        targetOutputs[correctPredication] = 1
        
        return targetOutputs
    }
    
    internal func trainNeuralNetwork(epochs: Int, inputs: Matrix, targetOutputs: Matrix)
    {
        DispatchQueue.global().async
        {
            do
            {
                // Train the network using the raw image pixels and the correct prediction.
                try self.recreationNetwork!.train(epochs: epochs, errorFunction: self.errorFunction!, learningRate: self.learningRate, momentum: self.momentum, trainingInputs: targetOutputs, targetOutputs: inputs)
                try self.classificationNetwork!.train(epochs: epochs, errorFunction: self.errorFunction!, learningRate: self.learningRate, momentum: self.momentum, trainingInputs: inputs, targetOutputs: targetOutputs)
            }
            catch { print(error) }
        }
    }
    
    internal func openCamera(fromViewController viewController: UIViewController, usingDelegate delegate: UIImagePickerControllerDelegate & UINavigationControllerDelegate)
    {
        if !UIImagePickerController.isSourceTypeAvailable(.camera) { return }
        
        let cameraUI = UIImagePickerController()
        cameraUI.sourceType = .camera
        cameraUI.allowsEditing = false
        cameraUI.delegate = delegate
        
        guard let availableMediaTypes = UIImagePickerController.availableMediaTypes(for: .camera) else { return }
        cameraUI.mediaTypes = availableMediaTypes
        
        viewController.present(cameraUI, animated: true, completion: nil)
    }
    
    private func buildNetworks(withArchitecture architecture: [Int])
    {
        self.classificationNetwork = JPSNeuralNetwork(architecture: architecture, activationFunctions: self.activationFunctions!)
        self.classificationNetwork!.delegate = self
        self.classificationNetwork!.bias = 1
        self.classificationNetwork!.weights = self.weights
        
        self.recreationNetwork = JPSNeuralNetwork(architecture: architecture.reversed(), activationFunctions: self.activationFunctions!.reversed())
        self.recreationNetwork!.bias = 1
        self.recreationNetwork!.weights = self.weights2
    }
    
    private func predictClassification() -> Scalar
    {
        self.feedPixelsForward(inputs: self.imagePixels!)
        
        let max = self.prediction!.max()!
        let predictedDigit = self.prediction!.index(of: max)!
        return Scalar(predictedDigit)
    }
    
    @IBAction func backpropagate(_ sender: Any)
    {
        guard let targetOutputs = self.targetOutputsForCorrectPredication() else { return }
        
        self.predict(image: self.resizedImage!)
        self.trainNeuralNetwork(epochs: 1, inputs: [self.imagePixels!], targetOutputs: [targetOutputs])
    }
    
    @IBAction func openCamera(_ sender: Any) {
        self.openCamera(fromViewController: self.navigationController!, usingDelegate: self)
    }
    
    @IBAction func changeColorSpace(_ sender: UISegmentedControl)
    {
        self.colorSpace = ColorSpace(rawValue: sender.selectedSegmentIndex)!
        
        let numberOfComponents = CGFloat(self.colorSpace == .grayScale ? 1 : 4)
        
        var architecture = self.architecture!
        architecture[0] = Int(self.resizedImageSize.width * self.resizedImageSize.height * numberOfComponents)
        self.buildNetworks(withArchitecture: architecture)
    }
}

extension LiveTrainingViewController: JPSCanvasViewDelegate
{
    public func canvasViewTouchesBegan(canvasView: JPSCanvasView) { }
    
    public func canvasViewTouchesEnded(canvasView: JPSCanvasView)
    {
        if let image = self.canvasView.getImage()?.resizedImage(self.resizedImageSize, interpolationQuality: .high)
        {
            canvasView.clear()
            
            self.resizedImage = image
            self.predict(image: image)
            
            let pixels = self.predictionInverse!.map({
                return UInt8($0 * Scalar(UInt8.max))
            })
            
            self.printNumber(rowSize: Int(self.resizedImageSize.width), pixels: pixels)
            
            if let targetOutputs = self.targetOutputsForCorrectPredication() {
                self.trainNeuralNetwork(epochs: 1, inputs: [self.imagePixels!], targetOutputs: [targetOutputs])
            }
            else {
                self.currentError = 0
                self.previousError = 0
                self.errorLabel.text = "Error: ???"
                self.errorLabel.textColor = UIColor.black
            }
        }
    }
}

extension LiveTrainingViewController: UITextFieldDelegate
{
    public func textFieldShouldReturn(_ textField: UITextField) -> Bool
    {
        if textField.isEqual(numberOfOutputsTextField)
        {
            if self.canChangeNumberOfOutputs() {
                self.changeNumberOfOutputs()
            }
            else {
                let alertController = UIAlertController(title: "Error", message: "The number of outputs is less than the correct prediction! The number of outputs should be grater than the correct prediction.", preferredStyle: .alert)
                
                let okayAction = UIAlertAction(title: "Okay", style: .default, handler: nil)
                alertController.addAction(okayAction)
                
                self.navigationController?.present(alertController, animated: true, completion: nil)
                
                return false
            }
        }
        
        return textField.resignFirstResponder()
    }
}

extension LiveTrainingViewController: JPSNeuralNetworkDelegate
{
    public func network(_ network: JPSNeuralNetwork, errorDidChange error: Scalar)
    {
        self.previousError = self.currentError
        self.currentError = error
        
        let deltaError = (self.previousError - self.currentError)
        let sign = (deltaError > 0 ? "-" : "+")
        
        DispatchQueue.main.async
        {
            self.errorLabel.text = "Error: \(self.currentError) (\(sign)\(abs(deltaError)))"
            self.errorLabel.textColor = (self.currentError > self.previousError ?  UIColor.red :  UIColor(red: 0, green: (230.0 / 255.0), blue: 0, alpha: 1))
        }
    }
    
    public func network(_ network: JPSNeuralNetwork, progressDidChange progress: Scalar) { }
    
    public func network(_ network: JPSNeuralNetwork, overallProgressDidChange progress: Scalar) { }
}

extension LiveTrainingViewController: UINavigationControllerDelegate { }

extension LiveTrainingViewController: UIImagePickerControllerDelegate
{
    public func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true, completion: nil)
    }
    
    public func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any])
    {
        let selectedImage = (info[UIImagePickerControllerOriginalImage] as? UIImage)?.resizedImage(self.resizedImageSize, interpolationQuality: .high)
        
        guard let image = selectedImage else  {
            
            picker.dismiss(animated: true) { }
            return
        }
        
        self.resizedImage = image
        
        picker.dismiss(animated: true)
        {
            self.canvasView.clear()
            self.predict(image: image)
            
            let pixels = self.predictionInverse!.map({
                return UInt8($0 * Scalar(UInt8.max))
            })
            
            self.printNumber(rowSize: Int(self.resizedImageSize.width), pixels: pixels)
            
            if let targetOutputs = self.targetOutputsForCorrectPredication() {
                self.trainNeuralNetwork(epochs: 1, inputs: [self.imagePixels!], targetOutputs: [targetOutputs])
            }
            else {
                self.currentError = 0
                self.previousError = 0
                self.errorLabel.text = "Error: ???"
                self.errorLabel.textColor = UIColor.black
            }
        }
    }
}

