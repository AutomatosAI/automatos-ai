// Code-graph parser fixture (PRD-262 US-028): a protocol, a class with a method, and a function.
import Foundation

protocol Polite {
    func bow()
}

class Greeter: Polite {
    func greet(name: String) -> String {
        return "Hello, \(name)"
    }

    func bow() {}
}

func makeGreeter() -> Greeter {
    return Greeter()
}
