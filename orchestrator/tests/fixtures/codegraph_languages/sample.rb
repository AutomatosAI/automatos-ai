# Code-graph parser fixture (PRD-262 US-028): a class with a method, and a method at the top.
require "json"

class Greeter
  def greet(name)
    "Hello, #{name}"
  end
end

def make_greeter
  Greeter.new
end
