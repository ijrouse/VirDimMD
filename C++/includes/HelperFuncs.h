#pragma once
#include <iostream>
#include <Eigen/Dense>
#include "Config.h"


//mask an NP x NP array along dimension J 
Eigen::ArrayXXd periodicMask( Eigen::ArrayXXd &distSet, double boxLength, bool isPBC);


void initialiseGrid(int numParticles, int numRealDim, int numVirtDim, int numParticlesPerSide, double genGridBoxLength, Eigen::ArrayXXd&  pos);



void initialiseHCPGrid(int numParticles, int numRealDim, int numVirtDim, int numParticlesPerSide, double genGridBoxLength, Eigen::ArrayXXd&  pos);

void initialiseThermalVelocity(Config &simconfig, Eigen::ArrayXXd& vel);

void rescaleVelocity(Config &simconfig, Eigen::ArrayXXd &vel);

double getKineticEnergy(Config &simconfig, Eigen::ArrayXXd& vel);

double getKineticEnergy(Config &simconfig, Eigen::ArrayXXd& vel, int axis );


void printTempComponents(Config &simconfig, Eigen::ArrayXXd& vel);

void zeroCOM(Config &simconfig, Eigen::ArrayXXd &arr);


double maxSafeHarmonicConst(Config &simconfig, double safetyFactor = 2.0);

void rescaleAxis(Eigen::ArrayXXd &pos, int axis, double axisScale);

void rescaleToIdealGas(Config &simconfig, Eigen::ArrayXXd &pos, std::vector<double> pressureComponents, double particleVolume=0.0);

void initialiseRandom(Config &simconfig, Eigen::ArrayXXd &pos);