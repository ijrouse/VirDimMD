#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <iomanip>
 

class XYZWriter{
    public:
    std::string outputFilename;
    std::ofstream fileHandle;
    int axis1, axis2, axis3;
    double scaleFactor = 10.0; //default to shifting to angstrom
    bool wrapPBC = false;
    bool wrapMoleculeCOM = true; //if wrapping to PBC is on, shift molecules by their COM to keep bonds together
    double boxlength1 = 1.0;
    double boxlength2 = 1.0;
    double boxlength3 = 1.0;
    double x1offset = 0.0;
    bool hasBonds = false;
    Eigen::ArrayXi bondAtomSet1;
    Eigen::ArrayXi bondAtomSet2;
    XYZWriter( std::string fileIn, int a1, int a2, int a3);
    void setPBC(double b1, double b2, double b3);
    double pbcWrap(bool doWrap, double x, double boxLength);
    void writeXYZFrame( Eigen::ArrayXXd &pos, std::vector<std::string> atomLabels , std::string comment);
    void writePDBFrame(Eigen::ArrayXXd &pos, std::vector<std::string> atomLabels , std::string comment  );
    void writePSF(Eigen::ArrayXXd &pos, std::vector<std::string> atomLabels , std::string comment, Eigen::ArrayXd mass  );
    void addBondGroups(Eigen::ArrayXi &bondAtom1, Eigen::ArrayXi &bondAtom2);
};


 