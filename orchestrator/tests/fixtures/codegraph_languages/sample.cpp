// Code-graph parser fixture (PRD-262 US-028): a class and a function, in a namespace.
#include <string>

namespace fixtures {

class Greeter {
public:
    std::string greeting;
};

int add(int a, int b)
{
    return a + b;
}

}  // namespace fixtures
