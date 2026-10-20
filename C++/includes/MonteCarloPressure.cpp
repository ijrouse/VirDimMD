 

#include <Eigen/Dense>
#include <random>
#include <string>
#include "Config.h"
#include "HelperFuncs.h"
#include "Forces.h"
#include "MonteCarloPressure.h"

 
    MonteCarloPressure::MonteCarloPressure( int numDim, double initPressure){
        for(int j =0; j<numDim; j++){
            targetPressures.push_back(initPressure*barInMDUnits) ;
        }
    }
    MonteCarloPressure::MonteCarloPressure(std::vector<double> &initPressures){
        for(int j =0; j< initPressures.size(); j++){
            targetPressures.push_back( initPressures[j]*barInMDUnits );
        }
        isotropic = false;

    }
    void MonteCarloPressure::MCPressureUpdate(Config &simconfig, Eigen::ArrayXXd &pos, std::vector<Force*> forceSet){

        
         int scaleAxis = 0; //random from all real axes
        if(isotropic == false){
            std::uniform_int_distribution<int> randomAxisDist(0, simconfig.numRealDim -1); //-1 because the range is inclusive
            scaleAxis = randomAxisDist(simconfig.gener); 
        }
         double uBefore = getPotentialEnergy(forceSet, simconfig, pos,   false, true);
         Eigen::ArrayXXd posBackup = pos;
         bool accepted = false;
         std::vector<double> boxLengthBackup;

         double currentVolume = 1.0;
         for(int j=0; j<simconfig.numRealDim; j++){
            currentVolume = currentVolume * simconfig.boxLengths[j] ;
            boxLengthBackup.push_back(simconfig.boxLengths[j]);
         }
        double scaleWidth = 0.4;
        //to-do: update this so that the width is dynamically based on recent successful moves for that axis
        //e.g. keep a tracker of accepted moves, compute sdev separately for increases and decreases, set width to the lesser of 2*std or 0.4 while keeping mean at 1 for detailed balance
         double randomScaleFactor = 1.0 - scaleWidth/2.0 + ( scaleWidth  ) * simconfig.unitUniformDist(simconfig.gener) ;//generates a volume shift between 0.8 and 1.2 
        double newVolume = randomScaleFactor * currentVolume;
        double dv = newVolume - currentVolume;
         if(isotropic == true){
            double isotropicScaleFactor = pow(randomScaleFactor, 1.0/simconfig.numRealDim);
            for(int j =0; j<simconfig.numRealDim; j++){ 
            simconfig.boxLengths[j] = simconfig.boxLengths[j] * isotropicScaleFactor ;
            pos(Eigen::placeholders::all,j) = pos(Eigen::placeholders::all,j)*isotropicScaleFactor ;
            }
            std::cout <<"Attempting isotropic scale of " << isotropicScaleFactor << "\n";
         }
         else{
            simconfig.boxLengths[scaleAxis ] = simconfig.boxLengths[scaleAxis] * randomScaleFactor;
            pos(Eigen::placeholders::all,scaleAxis) = pos(Eigen::placeholders::all,scaleAxis)*randomScaleFactor ;
            std::cout <<" Attempting anisotropic scale on " << scaleAxis << " of " << randomScaleFactor << "\n";
         }

         double uAfter = getPotentialEnergy(forceSet, simconfig, pos,   true, true);
         std::cout << "Potential energy before: " << uBefore << "  after " <<  uAfter << " delta: " << uAfter - uBefore <<  "\n";
         std::cout <<"Pressure factor: " << targetPressures[scaleAxis]*dv << "\n";
         std::cout << "Log vol factor: " << simconfig.numParticles * simconfig.kbVal * simconfig.temp * std::log(newVolume/currentVolume  ) << "\n";
         double deltaE = (uAfter - uBefore + targetPressures[scaleAxis]*dv - simconfig.numParticles * simconfig.kbVal * simconfig.temp * std::log(newVolume/currentVolume  ));
         double acceptanceProb = std::min(1.0,  std::exp( -1.0* deltaE)/(simconfig.kbVal * simconfig.temp) );
         std::cout <<"Proposed volume scale: " << randomScaleFactor << " delta E: " << deltaE << " acceptance prob " << acceptanceProb << "\n";
         if(simconfig.unitUniformDist(simconfig.gener) < acceptanceProb){
            accepted = true;
         }
         if(accepted == false){
            pos = posBackup;
            uAfter = getPotentialEnergy(forceSet, simconfig, pos,   true, false); //this is just to recompute the forces 
            for(int j=0; j<simconfig.numRealDim; j++){
                simconfig.boxLengths[j] = boxLengthBackup[j] ; //restore the original lengths
            }

         }
         double finalVolume = 1.0;
          std::cout << "Current box lengths: ";
         for(int j=0; j<simconfig.numRealDim; j++){
            std::cout << simconfig.boxLengths[j] << "," ;
            finalVolume = finalVolume * simconfig.boxLengths[j] ;
         }
         std::cout << "\n Box resized from " << currentVolume << " to " << finalVolume << "\n";
        
         

    }
