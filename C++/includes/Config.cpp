#include <Eigen/Dense>
#include <random>
#include "Atoms.h"
#include "Config.h"

 
    Config::Config( int numParticlesIn, int numRealDimIn, int numVirtDimIn, Eigen::ArrayXd &massIn , double dtIn, double lGammaIn ){

        mass = massIn;
        dt = dtIn;
        lGamma = lGammaIn;
        numParticles = numParticlesIn;
        numRealDim = numRealDimIn;
        numVirtDim = numVirtDimIn;
        ndim = numRealDim + numVirtDim;
        boxLengths = std::vector<double>( ndim, -1.0);
//define a random generator for global use
    std::random_device rd{};
    std::mt19937 gen{rd()};
    gener = gen;
    }
    double Config::getCurrentVolume(){
    double currentVolume = 1.0;
    for(int j =0; j< numRealDim; j++){
        currentVolume = currentVolume * boxLengths[j] ;
    }
    return currentVolume;
}
 