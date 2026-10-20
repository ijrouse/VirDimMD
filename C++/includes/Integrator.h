#pragma once
#include <Eigen/Dense>
#include <random>
#include "Config.h"
#include "HelperFuncs.h"
#include "Forces.h"
//based on simconfig and particle positions, update the forces Array in-place to give the total force acting on each axis of each particle




void updateForces(Config &simconfig, Eigen::ArrayXXd &pos, Eigen::ArrayXXd &forces ,   std::vector<Force*> forceSet   );

void LangevinStep(Config &simconfig, Eigen::ArrayXXd &pos, 
    Eigen::ArrayXXd &vel, Eigen::ArrayXXd &forces, Eigen::ArrayXXd &posNew, 
    Eigen::ArrayXXd &velNew, Eigen::ArrayXXd &forcesNew  ,   std::vector<Force*> forceSet );

void EnergyMinStep(Config &simconfig, Eigen::ArrayXXd &pos, 
    Eigen::ArrayXXd &vel, Eigen::ArrayXXd &forces, Eigen::ArrayXXd &posNew, 
    Eigen::ArrayXXd &velNew, Eigen::ArrayXXd &forcesNew  ,   std::vector<Force*> forceSet , bool decayVirtual = false, double maxDelta = 0.005);


void MinimiseEnergy(Config &simconfig, Eigen::ArrayXXd &pos, 
    Eigen::ArrayXXd &vel, Eigen::ArrayXXd &forces, Eigen::ArrayXXd &posNew, 
    Eigen::ArrayXXd &velNew, Eigen::ArrayXXd &forcesNew  ,   std::vector<Force*> forceSet );


void RestoreRealityEnergyMin(Config &simconfig, Eigen::ArrayXXd &pos, 
    Eigen::ArrayXXd &vel, Eigen::ArrayXXd &forces, Eigen::ArrayXXd &posNew, 
    Eigen::ArrayXXd &velNew, Eigen::ArrayXXd &forcesNew  ,   std::vector<Force*> forceSet  , int maxAttempts = 50 );