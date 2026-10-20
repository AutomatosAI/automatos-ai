// Code-graph parser fixture (PRD-262 US-028): a component function, and a class with a method.
import { Component } from 'react';

export function Greeting({ name }) {
  return <p className="greeting">Hello, {name}</p>;
}

export class Panel extends Component {
  render() {
    return <Greeting name="world" />;
  }
}
