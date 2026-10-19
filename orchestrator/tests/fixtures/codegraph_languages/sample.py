"""Code-graph parser fixture (PRD-262 US-028): a class with a method, and a function."""


class Greeter:
    """Says hello."""

    def greet(self, name):
        """Return a greeting for ``name``."""
        return f"Hello, {name}"


def make_greeter():
    """Build a Greeter."""
    return Greeter()
