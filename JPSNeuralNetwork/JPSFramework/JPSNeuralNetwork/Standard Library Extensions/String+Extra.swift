//
//  String+Extra.swift
//  Neural Network
//
//  Created by Jonathan Sullivan on 8/6/17.
//  Copyright © 2017 Jonathan Sullivan. All rights reserved.
//

import Foundation

extension String
{
    public func attributedString() -> NSAttributedString {
        return NSAttributedString(string: self)
    }
    
    public func with(color: UIColor) -> NSAttributedString {
        return NSAttributedString(string: self, attributes: [NSForegroundColorAttributeName: color])
    }
}
