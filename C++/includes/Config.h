#pragma once
#include <Eigen/Dense>
#include <random>
#include "Atoms.h"
 

class Config{
    public:
    int numParticles;
    Eigen::ArrayXd mass;
    double dt;
    double lGamma;
    int numRealDim;
    int numVirtDim;
    int ndim;
    double kbVal = 0.0083144; //this sets an energy scale in kJ/mol 
    double temp = 300.0;
    int resetVelocityFreq = -1;
    double realityResetInterval = 0.5 ; //on average reset to reality once every <this> ps 
    bool allowFirebreak = true;
    bool firebreakRatio = 0.5;
    std::vector<double> boxLengths;
    std::vector<Atom> atoms;
    std::mt19937 gener;
    std::normal_distribution<double> unitNormalDist{0.0, 1.0};
    std::exponential_distribution<double> unitExpDist{1.0};
    std::uniform_real_distribution<double> unitUniformDist{0.0, 1.0};
    bool usePBC = true;
    bool clampVirtual = false; //if true then |x| is non-increasing for all virtual dimensions
    Config( int numParticlesIn, int numRealDimIn, int numVirtDimIn, Eigen::ArrayXd &massIn , double dtIn, double lGammaIn ); 
    double getCurrentVolume();
};