#pragma once
#include <Eigen/Dense>
#include <random>
#include <string>
#include "Config.h"
#include "HelperFuncs.h"


class Force{
public:
std::string forceLabel;
double energy = 0.0;
bool isVirialPotential = false; //if true this force contributes to the MC pressure calculation and is expected to provide a defined virial terms
Eigen::ArrayXXd virialTerms; //placeholder 
Force();
virtual Eigen::ArrayXXd computeForce(Config &simconfig, Eigen::ArrayXXd &pos);
virtual void updateOnReset();
virtual void setMaxForceConst(double max);
virtual void setUpdateOnReset(bool newVal);
 
};


class ForceHarmonicBond : public Force{
    public:
    Eigen::ArrayXd bondConstants;
    Eigen::ArrayXd bondLengths;
    Eigen::ArrayXd atom1Set;
    Eigen::ArrayXd atom2Set;
    int numBonds;
    ForceHarmonicBond(Eigen::ArrayXd bondConstsIn) ;
    Eigen::ArrayXXd computeForce(Config &simconfig, Eigen::ArrayXXd &pos) override ;
};

class ForceHarmonic : public Force{
    public:
    double harmonicConst;
    double maxConst;
    double x0;
    int axis;
    bool increaseOnReset = false;
    ForceHarmonic(double harmonicConstIn, double x0In, int axisIn);
    void setMaxForceConst(double newMax);
    void updateOnReset() override;
    void setUpdateOnReset(bool newVal);
    Eigen::ArrayXXd computeForce(Config &simconfig, Eigen::ArrayXXd &pos) override;
};

class ForceVirtualPenalty : public Force{
    public:
        double timeConst; //decay rate in ps^-1
        double scaleFactor; //force penalty in kj/Mol /nm^2 /ps - set this such that e.g. a particle held at  at 0.1 nm for 0.1 ps has an extra 50 kJ/mol penalty 
        int axis;
        Eigen::ArrayXd storedPenalty;
        ForceVirtualPenalty(double timeConst0, double scaleFactor0, int axisIn0, Eigen::ArrayXd storedPenaltyIn);
            Eigen::ArrayXXd computeForce(Config &simconfig, Eigen::ArrayXXd &pos) override;
};

class ForceLJ612DH : public Force { 
    public:
    Eigen::ArrayXXd sigmaMatrix;
    Eigen::ArrayXXd epsMatrix;
    Eigen::ArrayXXd chargeMatrix;
    std::vector<Eigen::ArrayXXd*> pairDistComponents;
    
    double kappa;
    double chargeForceConstNM = (9e9) * pow( (1.6e-19) ,2)   * 1e9  /(1.66e-21); //base constant * charge^2 * nm/m  then convert to kj/mol
    bool allowVirialSelfTerm = true; //each particle applies a pure self-repulsion when computing the virial to prevent axis collapse in NPT 

    ForceLJ612DH(Eigen::ArrayXXd sigmaMatrixIn, Eigen::ArrayXXd epsMatrixIn, Eigen::ArrayXXd chargeMatrixIn, int numDim, int numParticles, double kappaIn = 1.0/0.7);

    Eigen::ArrayXXd computeForce(Config &simconfig, Eigen::ArrayXXd &pos) override ;
};

double getPotentialEnergy(std::vector<Force*> forceSet, Config &simconfig, Eigen::ArrayXXd &pos, bool recomputeForce = false, bool virialOnly = false);

Eigen::ArrayXXd getVirialTensor(std::vector<Force*> forceSet, Config &simconfig, Eigen::ArrayXXd &pos, bool recomputeForce = false, bool virialOnly = false);