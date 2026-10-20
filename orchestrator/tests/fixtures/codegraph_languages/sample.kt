// Code-graph parser fixture (PRD-262 US-028): a class with a method, and a function.
package fixtures

class Greeter(private val name: String) {
    fun greet(): String {
        return "Hello, $name"
    }
}

fun makeGreeter(): Greeter = Greeter("world")
