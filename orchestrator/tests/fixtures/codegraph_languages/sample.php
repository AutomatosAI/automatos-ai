<?php
// Code-graph parser fixture (PRD-262 US-028): a class with a method, and a function.

namespace Fixtures;

class Greeter
{
    public function greet(string $name): string
    {
        return "Hello, " . $name;
    }
}

function make_greeter(): Greeter
{
    return new Greeter();
}
