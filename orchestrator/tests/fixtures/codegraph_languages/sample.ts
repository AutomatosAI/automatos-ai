// Code-graph parser fixture (PRD-262 US-028): an interface, a class with a method, and a function.
export interface Greeting {
  text: string;
}

export class Greeter {
  greet(name: string): Greeting {
    return { text: `Hello, ${name}` };
  }
}

export function makeGreeter(): Greeter {
  return new Greeter();
}
