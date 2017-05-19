//
//  CostGraphViewController.swift
//  JPSNeuralNetwork
//
//  Created by Jonathan Sullivan on 5/11/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

import Foundation
import CorePlot

class CostGraphViewController: UIViewController
{
    @IBOutlet weak var activityIndicatorView: UIActivityIndicatorView!
    
    var epochs: Int?
    var costs: Vector?
    
    override func viewDidLoad()
    {
        super.viewDidLoad()
        
        self.activityIndicatorView.startAnimating()
        
        let viewBounds = self.view.bounds
        let yOffset = (self.navigationController!.navigationBar.frame.height + UIApplication.shared.statusBarFrame.height + 10)
        let graphHeight = (viewBounds.height - yOffset)
        let graphY = (viewBounds.minY + yOffset)
        let frame = CGRect(x: viewBounds.minX, y: graphY, width: viewBounds.width, height: graphHeight)
        self.renderGraph(withFrame: frame, completion: {
            self.activityIndicatorView.stopAnimating()
        })
    }
    
    internal func renderGraph(withFrame frame: CGRect, completion: @escaping () ->())
    {
        DispatchQueue.global(qos: .background).async
        {
            let hostingView = CPTGraphHostingView(frame: frame)
            
            DispatchQueue.main.sync {
                self.view.insertSubview(hostingView, belowSubview: self.activityIndicatorView)
            }
            
            let graph = CPTXYGraph(frame: hostingView.bounds)
            
            // Add some padding to the graph, with more at the bottom for axis labels.
            graph.plotAreaFrame!.paddingTop = 30.0
            graph.plotAreaFrame!.paddingRight = 30.0
            graph.plotAreaFrame!.paddingBottom = 50.0
            graph.plotAreaFrame!.paddingLeft = 30.0
            
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
            
            DispatchQueue.main.sync
            {
                graph.add(plot)
                completion()
            }
        }
    }
}

extension CostGraphViewController: CPTScatterPlotDataSource
{
    func numberOfRecords(for plot: CPTPlot) -> UInt {
        return UInt(self.costs!.count)
    }
    
    func number(for plot: CPTPlot, field fieldEnum: UInt, record idx: UInt) -> Any?
    {
        var number = NSNumber()
        
        let field = CPTScatterPlotField(rawValue: Int(fieldEnum))!
        
        switch field
        {
        case .X:
            number = NSNumber(value: Scalar(idx) / Scalar(self.epochs!))
            
        case .Y:
            fallthrough
            
        default:
            number = NSNumber(value: self.costs![Int(idx)])
            break
        }
        
        return number
    }
}
