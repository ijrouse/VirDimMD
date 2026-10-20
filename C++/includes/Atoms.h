#pragma once
#include <iostream>
#include <string>

class Atom{
    double mass;
    std::string label;
    Atom( std::string labelIn, double massIn);
};