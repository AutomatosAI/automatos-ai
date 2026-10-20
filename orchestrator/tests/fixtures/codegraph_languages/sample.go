// Code-graph parser fixture (PRD-262 US-028): two functions.
package fixtures

import "fmt"

func Greet(name string) string {
	return fmt.Sprintf("Hello, %s", name)
}

func MakeGreeting() string {
	return Greet("world")
}
