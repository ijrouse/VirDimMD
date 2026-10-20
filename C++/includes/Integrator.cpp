 
#include <Eigen/Dense>
#include <random>
#include "Config.h"
#include "HelperFuncs.h"
#include "Forces.h"
#include "Integrator.h"
//based on simconfig and particle positions, update the forces Array in-place to give the total force acting on each axis of each particle




void updateForces(Config &simconfig, Eigen::ArrayXXd &pos, Eigen::ArrayXXd &forces ,   std::vector<Force*> forceSet   ){

    forces.fill(0.0);    
    for ( int i = 0 ; i < forceSet.size() ; i++ ){
    forces += forceSet[i]->computeForce(simconfig, pos) ;
    } 
    

    //std::cout << "net force on xp0: " << forces(0,0) << "\n";

}

void LangevinStep(Config &simconfig, Eigen::ArrayXXd &pos, 
    Eigen::ArrayXXd &vel, Eigen::ArrayXXd &forces, Eigen::ArrayXXd &posNew, 
    Eigen::ArrayXXd &velNew, Eigen::ArrayXXd &forcesNew  ,   std::vector<Force*> forceSet ){
//Algorithm reference: https://arxiv.org/pdf/1212.1244 
int nParticles = simconfig.numParticles;
int nDim = simconfig.ndim;
Eigen::ArrayXXd noise(nParticles,nDim);
double noiseScale =   sqrt(2.0 * simconfig.lGamma * simconfig.kbVal * simconfig.temp * simconfig.dt); 


noise.fill(0.0);
forcesNew.fill(0.0);
for(int i=0; i< nParticles;i++){
    for(int j=0; j<nDim; j++){
        noise(i,j) = simconfig.unitNormalDist(simconfig.gener)* noiseScale;
    }
}
Eigen::ArrayXd invMass;
invMass = simconfig.mass.cwiseInverse();

Eigen::ArrayXd bVec =   (1.0 + simconfig.lGamma * simconfig.dt * invMass / (2.0)).cwiseInverse();
//b:  lgamma *dt * inv mass is mass/time * time / mass = dimensionless 
Eigen::ArrayXd dtMass = simconfig.dt * invMass/ (2.0 );
//std::cout <<"updating position\n";


Eigen::ArrayXXd posDelta =   (vel.colwise() *  bVec) * simconfig.dt  +   (forces.colwise() * dtMass ).colwise() * bVec   * simconfig.dt   + (noise.colwise() * bVec   ).colwise() *  dtMass  ;
posNew = pos + posDelta;
updateForces(simconfig, posNew, forcesNew, forceSet);
if( simconfig.clampVirtual==true){
    for(int j=simconfig.numRealDim; j < simconfig.ndim; j++){
        forcesNew( Eigen::placeholders::all, j) = 0.0;
    }
}

//for(int j =0; j < simconfig.ndim; j++){
//std::cout << j << ": " << posNew(0, j) << "," <<  vel(0,j) << " , " << forcesNew(0, j) << ";" ;
//    }
//std::cout << "\n";
//expect a broadcast error here
//std::cout <<"updating vels\n";
//Eigen::ArrayXXd forceSum = forces + forcesNew;
velNew = vel +  (forces + forcesNew).colwise() *dtMass    -     ( posDelta).colwise()*invMass  * simconfig.lGamma +  noise.colwise() * invMass ;
//update all Arrays to the new values
pos = posNew;
vel = velNew;
forces = forcesNew;
}


void EnergyMinStep(Config &simconfig, Eigen::ArrayXXd &pos, 
    Eigen::ArrayXXd &vel, Eigen::ArrayXXd &forces, Eigen::ArrayXXd &posNew, 
    Eigen::ArrayXXd &velNew, Eigen::ArrayXXd &forcesNew  ,   std::vector<Force*> forceSet , bool decayVirtual, double maxDelta  ){
        //Perform a step without using or updating the velocity or adding noise in.
//Algorithm reference: https://arxiv.org/pdf/1212.1244 
int nParticles = simconfig.numParticles;
int nDim = simconfig.ndim;
//double maxDelta = 0.005; //each step can move at most this far to prevent clashes
forcesNew.fill(0.0);
Eigen::ArrayXXd posDelta =     (forces    * simconfig.dt )    ;
double maxDisplacement = posDelta.abs().maxCoeff();
if( maxDisplacement > maxDelta){
    posDelta = posDelta * maxDelta/maxDisplacement;
}



if(decayVirtual == true){
    double maxMove = maxDelta;
    Eigen::ArrayXd secondDelta;
    Eigen::ArrayXd secondDeltaMag(simconfig.numParticles);
    Eigen::ArrayXd realDisplacementMag(simconfig.numParticles);
    secondDeltaMag.fill(0.0001);
    realDisplacementMag.fill(0.0001); //init with a small epsilon to prevent division by zero later
    for(int j=0;j<simconfig.numRealDim;j++){
    realDisplacementMag += posDelta(Eigen::placeholders::all, j).pow(2);
    }
    realDisplacementMag  = realDisplacementMag.sqrt();
    for(int j=simconfig.numRealDim; j<simconfig.ndim; j++){
        posDelta(Eigen::placeholders::all, j) = 0.0 ;
        secondDelta = -1.0 * (pos(Eigen::placeholders::all, j)*0.05).cwiseMin(maxMove).cwiseMax(-1.0 * maxMove) ;
        posDelta(Eigen::placeholders::all, j) = posDelta(Eigen::placeholders::all, j) + secondDelta;
        secondDeltaMag += secondDelta.pow(2);
    }
    secondDeltaMag = secondDeltaMag.sqrt();
        for(int j=0;j<simconfig.numRealDim;j++){
     posDelta(Eigen::placeholders::all, j) +=   secondDeltaMag * posDelta(Eigen::placeholders::all, j)/realDisplacementMag; //if it does an imaginary move of -delta, move +delta outwards based on its current min-energy path
    }
}
posNew = pos + posDelta;

updateForces(simconfig, posNew, forcesNew, forceSet);

//velNew = vel +  (forces + forcesNew).colwise() *dtMass    -     ( posDelta).colwise()*invMass  * simconfig.lGamma  ;
//update all Arrays to the new values
pos = posNew;

//vel = velNew;
forces = forcesNew;
}


void MinimiseEnergy(Config &simconfig, Eigen::ArrayXXd &pos, 
    Eigen::ArrayXXd &vel, Eigen::ArrayXXd &forces, Eigen::ArrayXXd &posNew, 
    Eigen::ArrayXXd &velNew, Eigen::ArrayXXd &forcesNew  ,   std::vector<Force*> forceSet ){

            double deltaE = -500;
    updateForces(simconfig, pos, forces, forceSet);
    double potentialEnergy = getPotentialEnergy(forceSet, simconfig, pos);
    double newPotentialEnergy = potentialEnergy;
    int minN = 0;
    int maxSteps = 21;
    double maxDelta = 0.005;
    Eigen::ArrayXXd posOld = pos;
    std::cout << "starting minimisation \n";
    double lastDeltaE = deltaE;
    while( minN < 20 || deltaE < -1)
    {
        
        posOld  = pos;
        lastDeltaE = deltaE;
        EnergyMinStep(simconfig, pos, vel, forces, posNew, velNew, forcesNew , forceSet, false, maxDelta);
        newPotentialEnergy = getPotentialEnergy(forceSet, simconfig, pos);
        deltaE = newPotentialEnergy - potentialEnergy ;  //should be negative if potentialE is decreasing
        if(deltaE > 0){
            std::cout <<"energy has increased, resetting and decreasing max delta to " << maxDelta << "\n";
            pos = posOld;
            maxDelta = maxDelta * 0.75;
            deltaE = lastDeltaE;
            //break;
        }
        else{
            potentialEnergy = newPotentialEnergy;
            if(minN % 10 ==0){ 
            std::cout << minN << " " << potentialEnergy << " " << deltaE << "\n";
            }
            minN += 1;
        }
        if(minN >= maxSteps || maxDelta < 1e-6){
            break;
        }
    }


    std::cout << "Finished minimisation at " << minN << " " << potentialEnergy << " " << deltaE << "\n";
    }


void RestoreRealityEnergyMin(Config &simconfig, Eigen::ArrayXXd &pos, 
    Eigen::ArrayXXd &vel, Eigen::ArrayXXd &forces, Eigen::ArrayXXd &posNew, 
    Eigen::ArrayXXd &velNew, Eigen::ArrayXXd &forcesNew  ,   std::vector<Force*> forceSet  , int maxAttempts   ){
//int maxAttempts = 50;
bool completed = false;
int numAttempts = 0;
double virtualRMSTarget = 0.0001; 
while(numAttempts < maxAttempts  ){
    EnergyMinStep(simconfig, pos, vel, forces, posNew, velNew, forcesNew , forceSet, true);
    numAttempts +=1 ;
    std::cout << " attempting step " << numAttempts << "\n";
    completed = true; //assume it succeeded until proven otherwise
    for(int j= simconfig.numRealDim; j<simconfig.numRealDim+simconfig.numVirtDim; j++){ 
        double virtualRMS =  sqrt(pos(Eigen::placeholders::all, j).pow(2).mean()  ) ;
        //std::cout << "j: " << virtualRMS << "\n";
        if(virtualRMS > virtualRMSTarget){
            completed = false; 
        }
 
        
    }
    if(completed == true){
        break;
    }
}

if(completed == true){
    for(int j= simconfig.numRealDim; j<simconfig.numRealDim+simconfig.numVirtDim; j++){ 
        pos(Eigen::placeholders::all, j).fill(0.0);
    }
    EnergyMinStep(simconfig, pos, vel, forces, posNew, velNew, forcesNew , forceSet, true);
}


MinimiseEnergy(simconfig, pos, vel, forces, posNew, velNew, forcesNew , forceSet);

}