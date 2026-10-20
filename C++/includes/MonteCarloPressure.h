#pragma once
#include <Eigen/Dense>
#include <random>
#include <string>
#include "Config.h"
#include "HelperFuncs.h"
#include "Forces.h"

class MonteCarloPressure{
    /*
    Defines a (potentially anisotropic) Monte Carlo barostat acting on all the real dimensions by volume rescaling
    Initialise either providing a vector of the per-axis pressures to enable anisotropic behaviour, or a single pressure and the number of real dimensions
    Input pressures are in bar, converted to internal units during initialisation
    internal units: pressure has units energy / volume (strictly force / area = N/m / m^2 )
    i.e. kJ/mol / nm^3 
    so 1 bar = 1e5 Pa = 1e5 * avo const * 1e-27 *1e3 kJ/mol / nm^3
    1 bar = 0.06 kJ/mol / nm^3
    so if the input is 1 bar, this is set to 0.06 kJ/mol/nm^3 by multiplying by this value
    following initialisation, isotropic behaviour (scaling all axes together) can be switched off
    */
public:
    double barInMDUnits = 0.06;
    int updateFreq = 50;
    std::vector<double> targetPressures;  
 
    bool isotropic = true;
    MonteCarloPressure( int numDim, double initPressure);
    MonteCarloPressure(std::vector<double> &initPressures);
    void MCPressureUpdate(Config &simconfig, Eigen::ArrayXXd &pos, std::vector<Force*> forceSet);
};