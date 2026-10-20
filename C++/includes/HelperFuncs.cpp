 
#include <iostream>
#include <Eigen/Dense>
#include "HelperFuncs.h"


//mask an NP x NP array along dimension J 
Eigen::ArrayXXd periodicMask( Eigen::ArrayXXd &distSet, double boxLength, bool isPBC){
    Eigen::ArrayXXd pbcDist ;
    if(boxLength < 0 || isPBC == false){
        return distSet;
    }
    pbcDist =  distSet -  boxLength*  (distSet/boxLength).round();
    return pbcDist; 
}


void initialiseGrid(int numParticles, int numRealDim, int numVirtDim, int numParticlesPerSide, double genGridBoxLength, Eigen::ArrayXXd&  pos){ 
    double beadRadius = genGridBoxLength/(2 * numParticlesPerSide);
    for (int i = 0; i < numParticles; i++)
    {
        for (int j = 0; j < numRealDim; j++)
        {
            int xi = i % numParticlesPerSide;
            if (j > 0)
            {
                xi = int(i / pow(numParticlesPerSide, j)) % numParticlesPerSide;
            }
            //std::cout << i << "," << j << "," << xi << " ";
            pos(i, j) = - genGridBoxLength / 2.0  + beadRadius + xi * 2*beadRadius;
            //std::cout << pos(i,j) << "\n";
        }
        for (int j = numRealDim; j < numRealDim + numVirtDim; j++)
        {
            pos(i, j) = 0.0;
        }
    }

}



void initialiseHCPGrid(int numParticles, int numRealDim, int numVirtDim, int numParticlesPerSide, double genGridBoxLength, Eigen::ArrayXXd&  pos){ 
    /* Generate an ND hexagonal-close packed structure in numRealDim while leaving numVirtDim dimensions at 0.0
    note that this does not have to strictly be the same dimension numbers as the simulation uses, such that a 2D grid can be generated in 3D through (numReal=2,numVirt=1) or 3+1D (numReal=2,numVirt=2) 
    */
    double beadRadius = genGridBoxLength / (2.0*(numParticlesPerSide)  ); //assign an effective radius to each bead to generate the grid of the desied size
    std::cout <<" building HCP \n";
    for (int i = 0; i < numParticles; i++)
    {
        //std::cout << i << ": " ;
        std::vector<int> indexSet; 
        std::vector<int> indexSum;
        for (int j = 0; j < numRealDim; j++)
        {

            int xi = i % numParticlesPerSide;
            
            if (j > 0)
            {
                xi = int(i / pow(numParticlesPerSide, j)) % numParticlesPerSide;
            }

            indexSet.push_back(xi) ;
            indexSum.push_back(0);
            //std::cout <<  " axis " << j << " index " << xi << " ";
            //pos(i, j) = - genGridBoxLength / 2.0 + xi * genGridBoxLength / (numParticlesPerSide-1);
            //std::cout << pos(i,j) << "\n";
        }
        //second pass: for each index[j] sum over all higher indices and modulo two to establish whether this is an odd or even layer
        
        for(int j=0; j< numRealDim; j++){
            for(int j2=j+1; j2<numRealDim; j2++){
                indexSum[j] += indexSet[j2] ;
            }
            indexSum[j] = indexSum[j] % 2 ;
        }
        
        for(int j =0; j<numRealDim; j++){
            double axisShift = sqrt( (j+2.0) / (2.0* (j+1.0)) );
            double oddOffset = axisShift * 1.0/(j+2.0);
            pos(i,j) = -genGridBoxLength/2.0 + beadRadius  + 2*beadRadius*(axisShift*indexSet[j] + oddOffset * indexSum[j] ); 
            //std::cout << pos(i,j) << ", " ;
        }
        //std::cout <<"\n";
        //pos(i, 0) = -genGridBoxLength/2.0  + (2* indexSet[0] + ( indexSet[1] + indexSet[2] ) % 2 )*beadRadius; 
        //pos(i,1) = -genGridBoxLength/2.0 + ( (sqrt(3) * indexSet[1]) +  (( indexSet[2] ) % 2)/sqrt(3)  ) * beadRadius;
        //pos(i,2) = - genGridBoxLength/2.0 + 2 * sqrt(2.0/3.0) * indexSet[2] * beadRadius;
    

        for (int j = numRealDim; j < numRealDim + numVirtDim; j++)
        {
            pos(i, j) = 0.0;
        }
    }

}

void initialiseThermalVelocity(Config &simconfig, Eigen::ArrayXXd& vel){
/* 
Initialise the thermal velocities to the target temperature
As a reminder: 1J = 1kg m^2 /s^2 
S.dev. of the gaussian is sqrt( kbT / mass )  which in the units used here gives approx 1.6 for mass = 1
 for 1 single H atom, the equivalent is 1500 m/s , confirming that the velocity units generated are consistent with nm/ps 
*/


int numParticles = simconfig.numParticles;
int numTotDim  = simconfig.ndim; 
for (int i = 0; i < numParticles; i++)
    {
        for (int j = 0; j < numTotDim; j++)
        {
            vel(i, j) = simconfig.unitNormalDist(simconfig.gener) * sqrt( simconfig.kbVal * simconfig.temp / simconfig.mass(i));
        }
    }
}

void rescaleVelocity(Config &simconfig, Eigen::ArrayXXd &vel){
    //at the target temperature  <v_{ads}^2> = kb T_target / m
    // <v_{actual}^2 > = kb T_actual / m
    //v_ads = v_actual * correctionFactor =>  correctionFactor(i)^2 = kb T_target/ (m * <v_actual^2> )
    int numParticles = simconfig.numParticles;
    int numTotDim  = simconfig.ndim; 
    for(int j =0; j<numTotDim; j++){
        double vSqMean = vel(Eigen::placeholders::all,j).pow(2).mean() ;
        Eigen::ArrayXd correctionFactor = (simconfig.kbVal * simconfig.temp* (simconfig.mass.cwiseInverse() )/(1e-4 + vSqMean)).sqrt();
        vel(Eigen::placeholders::all, j) = correctionFactor * vel(Eigen::placeholders::all, j);
    }

}

double getKineticEnergy(Config &simconfig, Eigen::ArrayXXd& vel){
    double ke =0.5*((vel.pow(2).colwise() * simconfig.mass ).sum() );
    return ke;
}

double getKineticEnergy(Config &simconfig, Eigen::ArrayXXd& vel, int axis ){
    double ke =0.5*((vel(Eigen::placeholders::all, axis).pow(2) * simconfig.mass ).sum() );
    return ke;
}



void printTempComponents(Config &simconfig, Eigen::ArrayXXd& vel){
     for(int j=0; j<simconfig.numRealDim;j++){
        double ke =0.5*((vel(Eigen::placeholders::all, j).pow(2) * simconfig.mass ).sum() );
        std::cout << "T_R_" << j << " " << ke/(0.5*simconfig.kbVal*(simconfig.numParticles-1) ) << " ";
     }   
          for(int j=simconfig.numRealDim; j<simconfig.ndim;j++){
        double ke =0.5*((vel(Eigen::placeholders::all, j).pow(2) * simconfig.mass ).sum() );
        std::cout << "T_V_" << j << " " << ke/(0.5*simconfig.kbVal*(simconfig.numParticles-1) ) << " " ;
     }   
}

void zeroCOM(Config &simconfig, Eigen::ArrayXXd &arr){
  arr = arr.rowwise() - ((arr.colwise() * simconfig.mass).colwise().mean()  / (simconfig.mass.sum() ) );
}


double maxSafeHarmonicConst(Config &simconfig, double safetyFactor ){
    //rule of thumb is that stability requires 20 timesteps per period, one period = 2 pi sqrt(mass/k) i.e.   ( 20/(2 pi) dt )**2 >   mass/k  
    // k < 4 pi^2  mass /  ( eta 20  dt )**2 where eta is a factor of safety
    //max k = 4 pi^2 min(mass) / (eta 20 dt)**2
    return 4 * pow(M_PI,2) * 1.0 / pow( safetyFactor * 20.0 * simconfig.dt,2) ;

}



void rescaleAxis(Eigen::ArrayXXd &pos, int axis, double axisScale){
    pos(Eigen::placeholders::all,axis) = pos(Eigen::placeholders::all, axis)*axisScale;
}

void rescaleToIdealGas(Config &simconfig, Eigen::ArrayXXd &pos, std::vector<double> pressureComponents, double particleVolume){
    double pressureMean = 0.0;
    double pressureProduct = 1.0;
    double pressureScaleFactor = 0.06; //convert from the input bar to MD units
    for(int j =0; j<simconfig.numRealDim; j++){
        pressureProduct = pressureProduct*pressureComponents[j]*pressureScaleFactor  ;
        pressureMean += pressureComponents[j]*pressureScaleFactor/simconfig.numRealDim;
    }
    pressureProduct = std::pow(pressureProduct, 1.0/simconfig.numRealDim); //geometric mean
    double length0 = std::pow( (simconfig.numParticles * simconfig.kbVal * simconfig.temp  ) / pressureMean + simconfig.numParticles*particleVolume , 1.0/simconfig.numRealDim);
    double finalVolume = 1.0;
    for(int j =0; j<simconfig.numRealDim; j++){
        double oldLength = simconfig.boxLengths[j];
        double newLength = length0 *  std::pow(   pressureProduct  / (pressureComponents[j]*pressureScaleFactor) , 1.0) ;
        rescaleAxis(pos,j,newLength/oldLength);
        simconfig.boxLengths[j] = newLength;
        std::cout <<"Axis " << j << " set to " << newLength <<"\n";
        finalVolume = finalVolume * newLength;
    }
    std::cout <<"Final volume: " << finalVolume << " required: "   << (simconfig.numParticles * simconfig.kbVal * simconfig.temp  ) / pressureMean  + simconfig.numParticles*particleVolume   << "\n";
}

void initialiseRandom(Config &simconfig, Eigen::ArrayXXd &pos){
 for(int i = 0; i<simconfig.numParticles; i++){
    for( int j=0; j<simconfig.numRealDim; j++){
        pos(i,j) =  -simconfig.boxLengths[j]/2.0 + simconfig.boxLengths[j]* simconfig.unitUniformDist(simconfig.gener) ;
    }
    for(int j=simconfig.numRealDim; j < simconfig.numRealDim + simconfig.numVirtDim; j++){
        pos(i,j) = -0.1 + 0.2 * simconfig.unitUniformDist(simconfig.gener) ;
    }
 }   
}