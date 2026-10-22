#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <string>
#include "includes/Atoms.h"
#include "includes/Config.h"
#include "includes/Integrator.h"
#include "includes/Writers.h"
#include "includes/HelperFuncs.h"
#include "includes/Forces.h"
#include "includes/MonteCarloPressure.h"

using namespace std;

/*
Unit conventions to use throughout:
Energies are in kJ/mol
Distances are in nanometers
Time is in ps  (1e-12) seconds
Mass is in amu

*/

int particleHandedness(int particleID, int numParticlesPerSide, int numRealDim)
{
    int x = 0;
    for (int j = 0; j < numRealDim; j++)
    {
        x += int(particleID / pow(numParticlesPerSide, j)) % numParticlesPerSide;
    }
    return x % 2;
}

int main(int argc, char **argv)
{
    bool buildHCP = false;
    bool unmixingDemo = false;
    bool spaghettiDemo = true;
    double beadSigma = 0.1;

    int numRealDim = 3;
    int numVirtDim = 1;
    int numTotDim = numRealDim + numVirtDim;
    int numParticlesPerSide = 6;
    int numParticles = pow(numParticlesPerSide, numRealDim);

    double hardBeadSigma = beadSigma * 0.9; // slightly reduced compared to the LJ parameter to produce a slight repulsion between beads initially and strain the configuration
    double autoBoxLength = 1.0 * hardBeadSigma * (numParticlesPerSide);
    // manual override for testing
    double genGridBoxLength = autoBoxLength;
    std::cout << "constructing arrays for " << numParticles << " particles ... ";
    // construct the vectors for holding particle properties
    Eigen::ArrayXXd pos(numParticles, numTotDim);
    pos.fill(0.0);
    if (buildHCP == true)
    {
        initialiseHCPGrid(numParticles, numRealDim, numVirtDim, numParticlesPerSide, genGridBoxLength, pos);
    }
    else
    {
        initialiseGrid(numParticles, numRealDim, numVirtDim, numParticlesPerSide, genGridBoxLength, pos);
    }
    Eigen::ArrayXXd vel(numParticles, numTotDim);
    vel.fill(0.0);

    Eigen::ArrayXXd forces(numParticles, numTotDim);
    forces.fill(0.0);
    Eigen::ArrayXXd posNew(numParticles, numTotDim);
    Eigen::ArrayXXd velNew(numParticles, numTotDim);
    Eigen::ArrayXXd forcesNew(numParticles, numTotDim);
    posNew = pos;
    velNew = vel;
    forcesNew = forces;
    Eigen::ArrayXd mass(numParticles);
    mass.fill(1.0);

    std::cout << "constructed arrays\n";

    double dt = 1e-4;    // 0.02 fs is fairly standard = 1e-2 fs = 1e-5 ps
    double lGamma = 1.0; // units amu/ps
    // std::cout << mass(1) << "\n";
    std::cout << "starting config generation \n";
    std::vector<Force *> forceSet;

    Eigen::ArrayXXd sigmaMatrixIn(numParticles, numParticles);
    Eigen::ArrayXXd epsMatrixIn(numParticles, numParticles);

    sigmaMatrixIn.fill(beadSigma);
    epsMatrixIn.fill(1.0);
    Eigen::ArrayXXd chargeMatrixIn(numParticles, numParticles);
    chargeMatrixIn.fill(0.0);

    if (unmixingDemo == true)
    {
        epsMatrixIn.fill(5.0);
        for (int i = 0; i < numParticles; i++)
        {
            for (int i2 = i; i2 < numParticles; i2++)
            {
                (particleHandedness(i, numParticlesPerSide, numRealDim) == particleHandedness(i2, numParticlesPerSide, numRealDim)) ? epsMatrixIn(i, i2) = 1.0 : epsMatrixIn(i, i2) = 0.1; // odd + odd = strong, odd+even = weak, even + even = strong
                epsMatrixIn(i2, i) = epsMatrixIn(i, i2);                                                                                                                                   // symmetrise
            }
        }
    }

    bool calcAutoBox = true;
    Config simconfig(numParticles, numRealDim, numVirtDim, mass, dt, lGamma);
    for (int j = 0; j < numRealDim; j++)
    {
        simconfig.boxLengths[j] = autoBoxLength;
    }

    // override box lengths for HCP
    if (buildHCP == true)
    {
        for (int j = 0; j < numRealDim; j++)
        {
            double axisShift = sqrt((j + 2.0) / (2.0 * (j + 1.0)));
            double beadRadius = genGridBoxLength / (2.0 * (numParticlesPerSide));
            simconfig.boxLengths[j] = 2 * beadRadius + 2 * beadRadius * (axisShift * (numParticlesPerSide - 1));
            // double minPos = -genGridBoxLength/2.0 + beadRadius;
        }
    }
    bool randomShiftInit = true;
    if (randomShiftInit == true)
    {
        Eigen::ArrayXXd randomShiftArr(numParticles, numTotDim);
        randomShiftArr.setRandom();
        std::cout << "Perturbing initial locations \n";
        pos = pos + 0.01 * randomShiftArr;
    }

    std::cout << "building forces \n";
    forceSet.push_back(new ForceLJ612DH(sigmaMatrixIn, epsMatrixIn, chargeMatrixIn, simconfig.ndim, simconfig.numParticles));
    double virtualRMS = 0.02;
    double virtualForceConst = simconfig.kbVal * simconfig.temp / (virtualRMS * virtualRMS);
    for (int j = simconfig.numRealDim; j < simconfig.ndim; j++)
    {
        Eigen::ArrayXd storedPenalty(numParticles);
        storedPenalty.fill(0.0);
        forceSet.push_back(new ForceHarmonic(virtualForceConst, 0.0, j)); // set up the constraints on virtual axes
        forceSet.back()->setUpdateOnReset(true);
        forceSet.back()->setMaxForceConst(maxSafeHarmonicConst(simconfig));
        // forceSet.push_back( new ForceVirtualPenalty(0.01, virtualForceConst/(0.1*0.1), j, storedPenalty)) ; //a particle held at 0.01 nm for 0.1 ps will have approx 50kj/mol penalty
    }
    // std::cout << "Initial virtual force restraint: " << virtualForceConst << "\n";
    // std::cout << "Estimated max safe value: " << maxSafeHarmonicConst(simconfig) << "\n";
    XYZWriter psfOut("testout.psf", 0, 1, 2);
    
    // bond some atoms together
    // harmonic bond
    if(spaghettiDemo == true){
    int numHarmonicBonds = numParticlesPerSide * numParticlesPerSide * (numParticlesPerSide - 1);
    Eigen::ArrayXi bondAtom1Set(numHarmonicBonds);
    Eigen::ArrayXi bondAtom2Set(numHarmonicBonds);
    Eigen::ArrayXd bondLengths(numHarmonicBonds);
    Eigen::ArrayXd bondConsts(numHarmonicBonds);
     
    for(int b0 = 0;  b0 < numHarmonicBonds  ; b0++ ){
         bondAtom1Set(b0) = b0;
         bondAtom2Set(b0) = b0 + numParticlesPerSide * numParticlesPerSide; 
    }
    //std::cout << "bondAtom1: " << bondAtom1Set << "\n";
    bondLengths.fill(beadSigma*1.0);
    bondConsts.fill(5000.0);
 
    forceSet.push_back(new ForceHarmonicBond(bondAtom1Set, bondAtom2Set, bondConsts, bondLengths));
    psfOut.addBondGroups( bondAtom1Set, bondAtom2Set);
    // angle bond
    int numAngleBonds = numParticlesPerSide * numParticlesPerSide * (numParticlesPerSide - 2); 
    Eigen::ArrayXi abondAtom1Set(numAngleBonds);
    Eigen::ArrayXi abondAtom2Set(numAngleBonds);
    Eigen::ArrayXi abondAtom3Set(numAngleBonds);
    Eigen::ArrayXd abondTheta(numAngleBonds);
    Eigen::ArrayXd abondConsts(numAngleBonds);
 
    for(int b0 = 0;  b0 < numAngleBonds  ; b0++ ){
         abondAtom1Set(b0) = b0;
         abondAtom2Set(b0) = b0 + numParticlesPerSide * numParticlesPerSide; 
         abondAtom3Set(b0) = b0 + 2* numParticlesPerSide * numParticlesPerSide; 
    }


    abondTheta.fill( 150.0 * 3.1415 / 180.0   ) ;  
    abondConsts.fill(500.0);
    forceSet.push_back(new ForceHarmonicAngleBond(abondAtom1Set, abondAtom2Set, abondAtom3Set, abondConsts, abondTheta));
    }
    // compute each force for the first time to get initial energies
    double potentialEnergy = 0.0;
    for (int j = 0; j < forceSet.size(); j++)
    {
        std::cout << "Starting force " << j << " of type " <<  forceSet[j]->forceLabel << "\n";
        forceSet[j]->computeForce(simconfig, pos);
        std::cout << "Force " << j << " energy: " << forceSet[j]->energy << "\n";
        potentialEnergy += forceSet[j]->energy;
    }
    std::cout << "Initial potential energy: " << potentialEnergy << "\n";
    // std::cout << trialCompute(0,0) << "\n";
    std::cout << " setting velocities \n";
    initialiseThermalVelocity(simconfig, vel);
    double kineticEnergy = getKineticEnergy(simconfig, vel);
    double t = 0;
    double endTime = 500000 * simconfig.dt;
    int numSteps = (int)round(endTime / simconfig.dt);
    double estMeanTemp =kineticEnergy / (0.5 * simconfig.kbVal * (numParticles - simconfig.ndim) * simconfig.ndim);

   


    XYZWriter xyzOut("testout_3_1_randoreset1ps.xyz", 0, 1, 2);
    xyzOut.setPBC(simconfig.boxLengths[0], simconfig.boxLengths[1], simconfig.boxLengths[2]);
    xyzOut.x1offset = 0.8;
    std::vector<std::string> atomLabels;
    if (unmixingDemo == true)
    {
        for (int i = 0; i < pos.rows(); i++)
        {
            (particleHandedness(i, numParticlesPerSide, numRealDim) == 0) ? atomLabels.push_back("A") : atomLabels.push_back("B");
        }
    }
    else
    {
        for (int i = 0; i < pos.rows(); i++)
        {
            atomLabels.push_back("A");
        }
    }
    zeroCOM(simconfig, pos);
    zeroCOM(simconfig, vel);
    if(spaghettiDemo == true){ 

    psfOut.writePSF( pos, atomLabels, "TEST", mass);
    }
    bool mcOn = false;
    int mcUpdateFreq = 5000;
    int mcSwitchOn = 1000;
    // MonteCarloPressure MCBarostat( simconfig.numRealDim, 1.0);
    std::vector<double> pressureSet{1.0, 1.0, 1.0};
    MonteCarloPressure MCBarostat(pressureSet);
    Eigen::ArrayXXd virialTensor(simconfig.ndim, simconfig.ndim);
    double particleVol = 4.0 / 3.0 * M_PI * pow(0.1 / 2.0, 3.0);
    // rescaleToIdealGas(simconfig, pos, pressureSet,particleVol);
    // initialiseRandom( simconfig, pos);
    xyzOut.setPBC(simconfig.boxLengths[0], simconfig.boxLengths[1], simconfig.boxLengths[2]);

    std::cout << " Initial minimisation \n";
    double deltaE = -500;
    updateForces(simconfig, pos, forces, forceSet);

    MinimiseEnergy(simconfig, pos, vel, forces, posNew, velNew, forcesNew, forceSet);
    // generate the first reset step, ensuring that at least one full reset interval passes
    int nextResetStep = int(round((1.0 + 0.0 * simconfig.unitExpDist(simconfig.gener)) * simconfig.realityResetInterval / simconfig.dt));
    int disableVirtualStep = int(numSteps * 0.75);

    double currentVolume = 1.0;
    std::cout << "Beginning integration\n";
    for (int n = 0; n < numSteps; n++)
    {
        if (mcOn == true && n > mcSwitchOn && n % mcUpdateFreq == 0)
        {
            MCBarostat.MCPressureUpdate(simconfig, pos, forceSet);
            virialTensor = getVirialTensor(forceSet, simconfig, pos);
            for (int j = 0; j < simconfig.ndim; j++)
            {
                std::cout << "Pressure_" << j << " " << 2.0 / simconfig.getCurrentVolume() * (getKineticEnergy(simconfig, vel, j) - virialTensor(j, j)) / MCBarostat.barInMDUnits << ", ";
            }
            std::cout << "\n";
            xyzOut.setPBC(simconfig.boxLengths[0], simconfig.boxLengths[1], simconfig.boxLengths[2]);
        }

        if (simconfig.numVirtDim > 0 && simconfig.clampVirtual == false && n > disableVirtualStep && n % 50 == 0)
        {
            double virtualRMS = 0;

            for (int j = numRealDim; j < numRealDim + numVirtDim; j++)
            {
                virtualRMS += sqrt(pos(Eigen::placeholders::all, j).pow(2).mean());
            }

 
            if (virtualRMS / (0.00001 + numVirtDim) < 0.001)
            {
                std::cout << "system is effectively real, freezing virtual dimensions \n";
                for (int j = numRealDim; j < numRealDim + numVirtDim; j++)
                {
                    pos(Eigen::placeholders::all, j) = 0.0;
                    vel(Eigen::placeholders::all, j) = 0.0;
                    forces(Eigen::placeholders::all, j) = 0.0;
                }
                simconfig.clampVirtual = true;
            }
        }

        if( simconfig.allowFirebreak == true && std::fabs(estMeanTemp - simconfig.temp)/(simconfig.temp+0.001)    > simconfig.firebreakRatio){
           std::cout <<"System seems unstable, performing firebreak \n";
           MinimiseEnergy(simconfig, pos, vel, forces, posNew, velNew, forcesNew , forceSet);
           initialiseThermalVelocity(simconfig, vel);
           updateForces(simconfig, pos, forces, forceSet);
        }
         
        if (n > nextResetStep)
        {
            nextResetStep = n + int(round((1.0 + 0.0 * simconfig.unitExpDist(simconfig.gener)) * simconfig.realityResetInterval / simconfig.dt));
            //only reset if the temperature is close to the target value
            if( std::fabs(estMeanTemp - simconfig.temp)/(simconfig.temp+0.001)    < 0.2   ){ 
            // MinimiseEnergy(simconfig, pos, vel, forces, posNew, velNew, forcesNew , forceSet);
            for (int i = 0; i < forceSet.size(); i++)
            {
                forceSet[i]->updateOnReset();
            }

            // MinimiseEnergy(simconfig, pos, vel, forces, posNew, velNew, forcesNew , forceSet);
            std::cout << " resetting reality, next at step " << nextResetStep << "\n";
            //rescaleVelocity(simconfig, vel);
        }
        else{
            std::cout << "Simulation is too thermally unstable, skipping reset \n";
        }
        }

        if (simconfig.resetVelocityFreq > 0 && n % simconfig.resetVelocityFreq == 0)
        {
            rescaleVelocity(simconfig, vel);
        }
        if (n % 100 == 0)
        {
            xyzOut.writeXYZFrame(pos, atomLabels, "REM");
        }
        LangevinStep(simconfig, pos, vel, forces, posNew, velNew, forcesNew, forceSet);
        // remove the COM motion
        zeroCOM(simconfig, pos);
        zeroCOM(simconfig, vel);
        t += simconfig.dt;
        if (n % 50 == 0)
        {
            potentialEnergy = getPotentialEnergy(forceSet, simconfig, pos);
            kineticEnergy = getKineticEnergy(simconfig, vel);
            estMeanTemp = kineticEnergy / (0.5 * simconfig.kbVal * (numParticles - simconfig.ndim) * simconfig.ndim);
            std::cout << n + 1 << " " << t << ": " << potentialEnergy << " " << kineticEnergy << " " << potentialEnergy + kineticEnergy << "p1 vel: " << vel(0, 0) << " temp est: " << estMeanTemp << "\n";
            printTempComponents(simconfig, vel);
            std::cout << "\n";
            for (int j = numRealDim; j < numRealDim + numVirtDim; j++)
            {
                std::cout << "v" << j << " sqrt<x2> : " << sqrt(pos(Eigen::placeholders::all, j).pow(2).mean()) << "\n";
            }
        }
    }
    // clean up force set at the end
    for (int i = 0; i < forceSet.size(); i++)
    {
        delete forceSet[i];
    }
    return 0;
}