//
//  Extensions.swift
//  pixy
//
//  Created by Bitan Nath on 23/07/26.
//

import WatchKit

extension UIColor {
    convenience init(hex: Int, alpha: CGFloat = 1) {
        let r = CGFloat((hex >> 16) & 0xFF) / 255
        let g = CGFloat((hex >> 8) & 0xFF) / 255
        let b = CGFloat(hex & 0xFF) / 255
        self.init(red: r, green: g, blue: b, alpha: alpha)
    }
}

extension UIColor {
    static func interpolate(from: UIColor, to: UIColor, fraction: CGFloat) -> UIColor {
        var r1: CGFloat = 0, g1: CGFloat = 0, b1: CGFloat = 0, a1: CGFloat = 0
        var r2: CGFloat = 0, g2: CGFloat = 0, b2: CGFloat = 0, a2: CGFloat = 0
        from.getRed(&r1, green: &g1, blue: &b1, alpha: &a1)
        to.getRed(&r2, green: &g2, blue: &b2, alpha: &a2)
        let f = max(0, min(1, fraction))
        return UIColor(red: r1 + (r2 - r1) * f, green: g1 + (g2 - g1) * f,
                       blue: b1 + (b2 - b1) * f, alpha: a1 + (a2 - a1) * f)
    }

    static func valkyrie(at t: CGFloat) -> UIColor {
        let stops: [(CGFloat, UIColor)] = [
            (0.00, UIColor(hex: 0xC6FFDD)),
            (0.25, UIColor(hex: 0xFBD786)),
            (0.50, UIColor(hex: 0xF7797D)),
            (0.75, UIColor(hex: 0x6DD5ED)),
            (1.00, UIColor(hex: 0xC6FFDD)),
        ]
        let t = max(0, min(1, t))
        for i in 0..<stops.count - 1 {
            if t >= stops[i].0 && t <= stops[i + 1].0 {
                let local = (t - stops[i].0) / (stops[i + 1].0 - stops[i].0)
                return interpolate(from: stops[i].1, to: stops[i + 1].1, fraction: local)
            }
        }
        return stops.last!.1
    }
}
