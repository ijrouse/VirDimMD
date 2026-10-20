
#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <iomanip>
#include "Writers.h"

 
    XYZWriter::XYZWriter( std::string fileIn, int a1, int a2, int a3){
        outputFilename = fileIn;
        axis1 = a1;
        axis2 = a2;
        axis3 = a3;
        fileHandle.open(outputFilename);
    }
    void XYZWriter::setPBC(double b1, double b2, double b3){
        boxlength1 = b1;
        boxlength2 = b2;
        boxlength3 = b3;
        wrapPBC = true;
    }
    double XYZWriter::pbcWrap(bool doWrap, double x, double boxLength){
        if(doWrap == false){
            return x;
        }
        return x - boxLength * round(x/boxLength); 
    }
    void XYZWriter::writeFrame( Eigen::ArrayXXd &pos, std::vector<std::string> atomLabels , std::string comment){
        int numAtoms = pos.rows();
        fileHandle << numAtoms << "\n";
        fileHandle << comment << "\n";
        for(int i = 0; i < numAtoms; i++){
            double x = (pbcWrap( wrapPBC, pos(i,axis1), boxlength1) + x1offset)*scaleFactor;
            double y = pbcWrap( wrapPBC, pos(i,axis2), boxlength2)*scaleFactor;
            double z = pbcWrap( wrapPBC, pos(i,axis3), boxlength3)*scaleFactor;
            fileHandle << atomLabels[i] << " " << std::setw(8) << x << " " << std::setw(8) << y << " " << std::setw(8) << z << "\n";
        }
        fileHandle.flush() ;
    }
 