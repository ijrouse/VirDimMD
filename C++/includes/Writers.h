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
    double boxlength1 = 1.0;
    double boxlength2 = 1.0;
    double boxlength3 = 1.0;
    double x1offset = 0.0;
    XYZWriter( std::string fileIn, int a1, int a2, int a3);
    void setPBC(double b1, double b2, double b3);
    double pbcWrap(bool doWrap, double x, double boxLength);
    void writeFrame( Eigen::ArrayXXd &pos, std::vector<std::string> atomLabels , std::string comment);
};