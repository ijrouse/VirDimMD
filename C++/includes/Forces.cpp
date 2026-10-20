#include <Eigen/Dense>
#include <random>
#include <string>
#include "Config.h"
#include "HelperFuncs.h"
#include "Forces.h"

 
Force::Force(){
}
Eigen::ArrayXXd Force::computeForce(Config &simconfig, Eigen::ArrayXXd &pos){
    Eigen::ArrayXXd nullForce(simconfig.numParticles, simconfig.ndim);
    nullForce.fill(0.0);
     std::cout << "calling null force (not overriden) \n";
    return nullForce;
}
void Force::updateOnReset(){
    //null
}
void Force::setMaxForceConst(double max){
    //null
}
void Force::setUpdateOnReset(bool newVal){
 //null
}
 
 


 ForceHarmonicBond::ForceHarmonicBond(Eigen::ArrayXd bondConstsIn) : bondConstants(bondConstsIn) {
        numBonds = bondConstsIn.rows() ;
    }
    Eigen::ArrayXXd ForceHarmonicBond::computeForce(Config &simconfig, Eigen::ArrayXXd &pos)  {
        Eigen::ArrayXXd forces(simconfig.numParticles, simconfig.ndim) ;
        Eigen::ArrayXd bondDistMag(numBonds);
        Eigen::ArrayXXd bondDistComponents = pos(atom1Set, Eigen::placeholders::all) - pos(atom2Set, Eigen::placeholders::all);
        bondDistMag.fill(0.0);
        for(int j = 0; j<simconfig.ndim; j++){
            bondDistMag += bondDistComponents(Eigen::placeholders::all, j).pow(2);
        }
        bondDistMag = bondDistMag.sqrt();
        Eigen::ArrayXd forceMag =  bondConstants * (bondDistMag - bondLengths) / (1e-5 + bondDistMag) ;
        for(int j =0; j<simconfig.ndim; j++){
            forces(atom1Set,j) = - forceMag * bondDistComponents(Eigen::placeholders::all, j);
            forces(atom2Set,j) =   forceMag * bondDistComponents(Eigen::placeholders::all, j);
        }
        return forces;
    }


 
    ForceHarmonic::ForceHarmonic(double harmonicConstIn, double x0In, int axisIn){
     harmonicConst = harmonicConstIn;
     x0 = x0In;
     axis = axisIn;
     maxConst = harmonicConst * 8;
    }
    void ForceHarmonic::setMaxForceConst(double newMax){
        maxConst = newMax;
    }
    void ForceHarmonic::updateOnReset()  {
    if(increaseOnReset == true){
        harmonicConst = std::min( maxConst, harmonicConst*1.41); //doubles the force constant every two increments
        std::cout <<" force has been increased to " << harmonicConst << "\n";
    }
    }
    void ForceHarmonic::setUpdateOnReset(bool newVal){
        increaseOnReset = newVal;
    }
    Eigen::ArrayXXd ForceHarmonic::computeForce(Config &simconfig, Eigen::ArrayXXd &pos)  {
    Eigen::ArrayXXd forces(simconfig.numParticles, simconfig.ndim) ;
    forces.fill(0.0);
    forces(Eigen::placeholders::all, axis) = - 1.0 * harmonicConst* pos(Eigen::placeholders::all, axis) ;
    energy = 0.5 * harmonicConst * pos(Eigen::placeholders::all, axis).pow(2).sum() ;
    return forces;   
    }
 
 
        ForceVirtualPenalty::ForceVirtualPenalty(double timeConst0, double scaleFactor0, int axisIn0, Eigen::ArrayXd storedPenaltyIn){
        timeConst = timeConst0;
        scaleFactor = scaleFactor0;
        axis = axisIn0;
        storedPenalty = storedPenaltyIn;
        } 
            Eigen::ArrayXXd ForceVirtualPenalty::computeForce(Config &simconfig, Eigen::ArrayXXd &pos)   {
               // double timeConstSteps = ;
    Eigen::ArrayXXd forces(simconfig.numParticles, simconfig.ndim) ;
    storedPenalty = storedPenalty* std::exp(-1.0 * timeConst * simconfig.dt);
    storedPenalty += simconfig.dt *  pos(Eigen::placeholders::all, axis).abs(); 
    
    forces.fill(0.0);
    forces(Eigen::placeholders::all, axis) = - 1.0 * scaleFactor* storedPenalty *  pos(Eigen::placeholders::all, axis) ;
    energy = 0.5 * scaleFactor* (pos(Eigen::placeholders::all, axis).pow(2)  * storedPenalty  ) .sum() ;
    return forces;   
    }
 

 
    ForceLJ612DH::ForceLJ612DH(Eigen::ArrayXXd sigmaMatrixIn, Eigen::ArrayXXd epsMatrixIn, Eigen::ArrayXXd chargeMatrixIn, int numDim, int numParticles, double kappaIn){
    sigmaMatrix = sigmaMatrixIn;
    chargeMatrix = chargeMatrixIn;
    epsMatrix = epsMatrixIn;
    //pairDists = distComponentsIn;
    kappa = kappaIn;
    isVirialPotential = true;
    Eigen::ArrayXXd virialTermsNew( numDim, numDim);
    virialTermsNew.fill(0.0);
    virialTerms = virialTermsNew;
    //Eigen::ArrayXXd blankArr ;
    for(int j =0; j<numDim; j++){
        Eigen::ArrayXXd *distComponentPointer = new Eigen::ArrayXXd( numParticles, numParticles);
        pairDistComponents.push_back(  distComponentPointer ) ; 
    }
    }

    Eigen::ArrayXXd ForceLJ612DH::computeForce(Config &simconfig, Eigen::ArrayXXd &pos)   {
       // std::cout << "calling LJ612DH \n";
    Eigen::ArrayXXd forces(simconfig.numParticles, simconfig.ndim) ;
    forces.fill(0.0);
    //compute pair-distances between particles
    //std::cout <<"pair dists\n";
    Eigen::ArrayXXd pairDists(simconfig.numParticles, simconfig.numParticles);
    Eigen::ArrayXXd pairDelta(simconfig.numParticles, simconfig.numParticles);
    Eigen::ArrayXXd selfMask = 2000.0 * (Eigen::MatrixXd::Identity( simconfig.numParticles, simconfig.numParticles).array() );
    pairDists.fill(0.0);
    for(int j =0; j<simconfig.ndim; j++){
    Eigen::ArrayXXd distSet = (pos(Eigen::placeholders::all,j).matrix().rowwise().replicate(simconfig.numParticles)   - pos(Eigen::placeholders::all,j).matrix().transpose().colwise().replicate(simconfig.numParticles) )  ;
       pairDists += periodicMask( distSet  , simconfig.boxLengths[j] , simconfig.usePBC  ).array().pow(2); 
       *(pairDistComponents)[j] = distSet ; 
    }
        //std::cout <<"computing forces \n";
        pairDists += selfMask; //before doing division make sure the self-terms are located at a large distance
    Eigen::ArrayXXd invDistSq  = 1.0/pairDists;
        pairDists = pairDists.sqrt();
    //std::cout <<pairDists(0,1) << "\n";
    Eigen::ArrayXXd forceContrib;
    virialTerms.fill(0.0);
    energy = 4.0 *(epsMatrix*(pow(sigmaMatrix,12)/( pairDists.pow(12) )- pow(sigmaMatrix,6)/( pairDists.pow(6) ) )).sum();
    for(int j =0; j<simconfig.ndim; j++){
        //pairDelta = 1.0 *  (pos(Eigen::placeholders::all,j).matrix().rowwise().replicate(simconfig.numParticles) - pos(Eigen::placeholders::all,j).matrix().transpose().colwise().replicate(simconfig.numParticles)    ).array();
        pairDelta = *(pairDistComponents)[j];
        //std::cout << j << " : "  << pos(0,j) << " to " << pos(1,j) << " = "   << pairDelta(0,1)  << " ";
        pairDelta = periodicMask(pairDelta, simconfig.boxLengths[j], simconfig.usePBC);
        forceContrib =  4.0 * epsMatrix *( 12.0 * invDistSq.pow(7) * sigmaMatrix.pow(12) * pairDelta  - 6.0 * pow(invDistSq,4) * sigmaMatrix.pow(6) * pairDelta );
        //forceContrib += chargeMatrix * chargeForceConstNM *  (-kappa * pairDists).exp() * pairDelta*(1 + pairDists*kappa)/(     pairDists.pow(3)   );
        //std::cout << " force on axis j " << forceContrib(0,1 ) << " " ;
        forces( Eigen::placeholders::all, j) = forceContrib.rowwise().sum();

     for(int jprime=j;jprime<simconfig.ndim;jprime++){
      double vjjprime = 0.0;
      Eigen::ArrayXXd virialJJPrime =  *(pairDistComponents)[jprime]  * forceContrib  ;
      //(virialJJPrime.matrix().triangularView<Eigen::StrictlyLower>())  ;
      for(int i=0; i<simconfig.numParticles; i++){
        for(int iprime = 0; iprime<i; iprime++){
            vjjprime += virialJJPrime(i,iprime);
        }
        if(allowVirialSelfTerm==true){
            double invSelfDistSq = 1.0/( pow( simconfig.boxLengths[j] ,2)) ;
            double pairSelfDelta = 1.0*simconfig.boxLengths[j];
            double virialSelfForceTerm = 4.0 * epsMatrix(i,i)*(12.0 * std::pow(invSelfDistSq,7)* std::pow(sigmaMatrix(i,i),12) * pairSelfDelta 
                                                                -6.0 * pow(invSelfDistSq,4) * std::pow(sigmaMatrix(i,i),6)*pairSelfDelta    );
            double virialSelfPlus = simconfig.boxLengths[jprime]*virialSelfForceTerm;
            //double virialSelfMinus = -simconfig.boxLengths[jprime]*virialSelfForceTerm ;
            vjjprime += 2*virialSelfPlus  ;
        }
      }
    virialTerms(j, jprime) = -0.5 * vjjprime;
    virialTerms(jprime, j) = -0.5 * vjjprime;

     }
    }


    return forces;
}
 

double getPotentialEnergy(std::vector<Force*> forceSet, Config &simconfig, Eigen::ArrayXXd &pos, bool recomputeForce , bool virialOnly ){
    double potentialEnergy = 0.0;
    for(int j =0; j < forceSet.size(); j++){ 
        if(recomputeForce == true){ 
            forceSet[j]->computeForce(simconfig, pos);
        }
        //std::cout << " force " << j << " energy " << forceSet[j]->energy << "\n";
        //if(forceSet[j]->isVirialPotential == true || virialOnly == false){
    potentialEnergy += forceSet[j]->energy ;
        //}
    }
    return potentialEnergy;
}

Eigen::ArrayXXd getVirialTensor(std::vector<Force*> forceSet, Config &simconfig, Eigen::ArrayXXd &pos, bool recomputeForce , bool virialOnly ){
Eigen::ArrayXXd virialTensor( simconfig.ndim, simconfig.ndim);
virialTensor.fill(0.0);
std::cout << "Virial tensor terms: " ;
 for(int j =0; j < forceSet.size(); j++){ 
    if( forceSet[j]->isVirialPotential){
        virialTensor += forceSet[j]->virialTerms;
    }
 }
for(int j=0;j<simconfig.ndim;j++){
    std::cout << virialTensor(j,j) << " , " ;
}
std::cout <<"\n";
return virialTensor;
}