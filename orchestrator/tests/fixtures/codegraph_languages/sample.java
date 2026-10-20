// Code-graph parser fixture (PRD-262 US-028): a class with a method, and an interface.
package fixtures;

import java.util.List;

public class Greeter {
    public void greet(String name) {
        System.out.println("Hello, " + name);
    }
}

interface Polite {
    void bow();
}
