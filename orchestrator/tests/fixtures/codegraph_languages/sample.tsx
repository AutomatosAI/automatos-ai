// Code-graph parser fixture (PRD-262 US-028): JSX, which only the TSX grammar reads.
interface GreetingProps {
  name: string;
}

export function Greeting({ name }: GreetingProps) {
  return <p className="greeting">Hello, {name}</p>;
}

export class Panel {
  render() {
    return <Greeting name="world" />;
  }
}
