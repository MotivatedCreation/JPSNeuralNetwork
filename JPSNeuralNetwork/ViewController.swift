//
//  ViewController.swift
//  Convolutional-Neural-Network
//
//  Created by Jonathan Sullivan on 4/4/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

import UIKit
import CorePlot


class ViewController: UIViewController
{
    @IBOutlet weak var progressLabel: UILabel!
    @IBOutlet weak var predictionLabel: UILabel!
    @IBOutlet weak var elapsedTimeLabel: UILabel!
    @IBOutlet weak var currentCostLabel: UILabel!
    @IBOutlet weak var tempImageView: UIImageView!
    @IBOutlet weak var currentEpochLabel: UILabel!
    @IBOutlet weak var progressView: UIProgressView!
    @IBOutlet weak var overallProgressLabel: UILabel!
    @IBOutlet weak var overallProgressView: UIProgressView!
    
    var startTime: TimeInterval?
    
    let epochs = 10000
    let topology = [2, 3, 1]
    let learningRate: Scalar = 0.4
    let activationFunction = JPSNeuralNetworkActivationFunction.sigmoid
    
    let targetOutputs: Matrix = [[0], [1], [1], [0]]
    let inputs: Matrix = [[0, 0], [1, 0], [0, 1], [1, 1]]
    
    var network: JPSNeuralNetwork?
    
    var currentEpoch = 0
    var weights = Matrix()
    var currentCost: Float = 0
    var previousCost: Float = 0
    var trainingData: (labels: [MNISTLabel], images: [MNISTImage])?
    
    var graph: CPTXYGraph?
    
    var lastPoint = CGPoint.zero
    var red: CGFloat = 0.0
    var green: CGFloat = 0.0
    var blue: CGFloat = 0.0
    var brushWidth: CGFloat = 1.0
    var opacity: CGFloat = 1.0
    var swiped = false
    
    override func viewDidLoad()
    {
        super.viewDidLoad()
        
        DispatchQueue.global(qos: .background).async
        {
            let dataLoader = JPSMNISTDataLoader()
            
            do {
                self.trainingData = try dataLoader.loadTrainingData()
            }
            catch { print(error) }
            
            self.startTime = Date.timeIntervalSinceReferenceDate
            
            let _ = JPSNeuralNetwork.train(delegate: self, topology: self.topology, epochs: self.epochs, learningRate: self.learningRate, activationFunction: .sigmoid, costFunction: .meanSquared, trainingInputs: self.inputs, targetOutputs: self.targetOutputs)
        }
    }
    
    internal func renderGraph()
    {
        let hostingView = CPTGraphHostingView(frame: self.view.bounds)
        self.view.addSubview(hostingView)
        
        // Create a graph object which we will use to host just one scatter plot.
        let frame = hostingView.bounds
        self.graph = CPTXYGraph(frame: frame)
        
        let graph = self.graph!
        
        // Add some padding to the graph, with more at the bottom for axis labels.
        graph.plotAreaFrame!.paddingTop = 20.0
        graph.plotAreaFrame!.paddingRight = 20.0
        graph.plotAreaFrame!.paddingBottom = 50.0
        graph.plotAreaFrame!.paddingLeft = 20.0
        
        // Tie the graph we've created with the hosting view.
        hostingView.hostedGraph = graph
        
        // If you want to use one of the default themes - apply that here.
        graph.apply(CPTTheme(named: CPTThemeName.plainWhiteTheme))
        
        // Create a line style that we will apply to the axis and data line.
        let lineStyle = CPTMutableLineStyle()
        lineStyle.lineColor = CPTColor.red()
        lineStyle.lineWidth = 2.0
        
        // Create a text style that we will use for the axis labels.
        let textStyle = CPTMutableTextStyle()
        textStyle.fontName = "Helvetica"
        textStyle.fontSize = 14
        textStyle.color = CPTColor.black()
        
        // Create the plot symbol we're going to use.
        let plotSymbol = CPTPlotSymbol.hexagon()
        plotSymbol.lineStyle = lineStyle
        plotSymbol.size = CGSize(width: 1, height: 1)
        
        
        // Setup some floats that represent the min/max values on our axis.
        let xAxisMin: Float = 0
        let xAxisMax: Float = 1
        let yAxisMin: Float = 0
        let yAxisMax: Float = 1
        
        // We modify the graph's plot space to setup the axis' min / max values.
        let plotSpace = (graph.defaultPlotSpace as! CPTXYPlotSpace)
        plotSpace.xRange = CPTPlotRange(locationDecimal: CPTDecimalFromFloat(xAxisMin), lengthDecimal: CPTDecimalFromFloat(xAxisMax - xAxisMin))
        plotSpace.yRange = CPTPlotRange(locationDecimal: CPTDecimalFromFloat(yAxisMin), lengthDecimal: CPTDecimalFromFloat(yAxisMax - yAxisMin))
        
        // Add a plot to our graph and axis. We give it an identifier so that we
        // could add multiple plots (data lines) to the same graph if necessary.
        let plot = CPTScatterPlot()
        plot.dataSource = self
        plot.dataLineStyle = lineStyle
        plot.plotSymbol = plotSymbol
        plot.interpolation = .curved
        graph.reloadData()
        graph.add(plot)
    }
    
    func drawLineFrom(fromPoint: CGPoint, toPoint: CGPoint)
    {
        UIGraphicsBeginImageContext(self.view.bounds.size)
        let context = UIGraphicsGetCurrentContext()
        self.tempImageView.image?.draw(in: CGRect(x: 0, y: 0, width: self.view.frame.size.width, height: self.view.frame.size.height))
        
        context?.move(to: fromPoint)
        context?.addLine(to: toPoint)
        context?.setLineCap(.round)
        context?.setLineWidth(self.brushWidth)
        context?.setStrokeColor(red: self.red, green: self.green, blue: self.blue, alpha: 1.0)
        context?.setBlendMode(.normal)
        context?.strokePath()
        
        self.tempImageView.image = UIGraphicsGetImageFromCurrentImageContext()
        self.tempImageView.alpha = self.opacity
        UIGraphicsEndImageContext()
        
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?)
    {
        self.swiped = false
        
        if let touch = touches.first {
            self.lastPoint = touch.location(in: self.view)
        }
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?)
    {
        self.swiped = true
        
        if let touch = touches.first
        {
            let currentPoint = touch.location(in: self.view)
            self.drawLineFrom(fromPoint: self.lastPoint, toPoint: currentPoint)
            
            self.lastPoint = currentPoint
        }
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?)
    {
        if !swiped {
            self.drawLineFrom(fromPoint: self.lastPoint, toPoint: self.lastPoint)
        }
        
        self.tempImageView.image = nil
    }
}

extension ViewController: JPSNeuralNetworkDelegate
{
    func network(costDidChange cost: Float)
    {
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
            
            let endTime = Date.timeIntervalSinceReferenceDate
            let elapsedTime = (endTime - self.startTime!).truncatingRemainder(dividingBy: 60)
            
            self.elapsedTimeLabel.text = "Elapsed Time: \(elapsedTime.stringFromTimeInterval())"
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
        }
    }
}

extension ViewController: CPTScatterPlotDataSource
{
    func numberOfRecords(for plot: CPTPlot) -> UInt {
        return 0
    }
    
    func number(for plot: CPTPlot, field fieldEnum: UInt, record idx: UInt) -> Any?
    {
        var number = NSNumber()
        
        let field = CPTScatterPlotField(rawValue: Int(fieldEnum))!
        
        switch field
        {
        case .X:
            number = NSNumber(value: Scalar(idx) / Scalar(self.epochs))
            
        case .Y:
            fallthrough
            
        default:
            number = 0
            break
        }
        
        return number
    }
}

public extension TimeInterval
{
    func stringFromTimeInterval() -> NSString
    {
        let timeInterval = NSInteger(self)
        
        let milliseconds = Int(self.truncatingRemainder(dividingBy: 1) * 1000)
        let seconds = timeInterval % 60
        let minutes = (timeInterval / 60) % 60
        let hours = (timeInterval / 3600)
        
        return NSString(format: "%0.2d:%0.2d:%0.2d.%0.3d", hours, minutes, seconds, milliseconds)
    }
}

public extension UIImage
{
    func mono() -> UIImage
    {
        var image = self
        
        var inputImage = CIImage(image: image)
        let options:[String : AnyObject] = [CIDetectorImageOrientation: (1 as AnyObject)]
        let filters = inputImage!.autoAdjustmentFilters(options: options)
        
        for filter: CIFilter in filters
        {
            filter.setValue(inputImage, forKey: kCIInputImageKey)
            inputImage = filter.outputImage
        }
        
        let context = CIContext(options: nil)
        let cgImage = context.createCGImage(inputImage!, from: inputImage!.extent)
        image = UIImage(cgImage: cgImage!)
        
        let currentFilter = CIFilter(name: "CIPhotoEffectMono")
        currentFilter!.setValue(CIImage(image: UIImage(cgImage: cgImage!)), forKey: kCIInputImageKey)
        
        let output = currentFilter!.outputImage
        let cgimg = context.createCGImage(output!, from: output!.extent)
        let processedImage = UIImage(cgImage: cgimg!)
        image = processedImage
        
        return image
    }
    
    public func normalize() -> UIImage
    {
        let width = 28
        let height = 28
        
        UIGraphicsBeginImageContext(CGSize(width: width, height: height))
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage!
    }
    
    func getPixelColor(pos: CGPoint) -> Float
    {
        let pixelData = self.cgImage!.dataProvider!.data
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        
        let pixelInfo: Int = ((Int(self.size.width) * Int(pos.y)) + Int(pos.x)) * 4
        
        let r = (Float(data[pixelInfo]) / Float(255.0))
        let g = (Float(data[pixelInfo+1]) / Float(255.0))
        let b = (Float(data[pixelInfo+2]) / Float(255.0))
        let a = (Float(data[pixelInfo+3]) / Float(255.0))
        
        return (r + g + b + a)
    }
    
    public func pixelColors() -> [MNISTPixel]
    {
        var result = [MNISTPixel]()
        
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        
        for y in 0..<height
        {
            for x in 0..<width {
                result.append(self.getPixelColor(pos: CGPoint(x: x, y: y)))
            }
        }
        
        print(result)
        
        return result
    }
}
