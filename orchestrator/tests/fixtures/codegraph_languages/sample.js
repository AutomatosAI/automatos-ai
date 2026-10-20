// Code-graph parser fixture (PRD-262 US-028): a class with a method, and a function.
import { format } from './format.js';

export class Greeter {
  greet(name) {
    return format(`Hello, ${name}`);
  }
}

export function makeGreeter() {
  return new Greeter();
}
