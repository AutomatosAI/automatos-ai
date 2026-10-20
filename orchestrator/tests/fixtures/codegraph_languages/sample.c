/* Code-graph parser fixture (PRD-262 US-028): two functions. */
#include <stdio.h>

static int add(int a, int b)
{
    return a + b;
}

void greet(const char *name)
{
    printf("Hello, %s %d\n", name, add(1, 2));
}
