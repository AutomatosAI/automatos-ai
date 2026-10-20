// Code-graph parser fixture (PRD-262 US-028): an interface, and a class with a method.
using System;

namespace Fixtures
{
    public interface IPolite
    {
        void Bow();
    }

    public class Greeter : IPolite
    {
        public string Greet(string name)
        {
            return "Hello, " + name;
        }

        public void Bow() { }
    }
}
