'''
This code is intended only as a demonstration of the algorithm

Internal units assume energy is expressed in terms of kbT with T = 300 K. The temperature specified is scaled to 300K and used for initial velocities and noise.

I. Rouse 25/03/2026
'''
import numpy as np
import VirDimMD as VDMD
import os

def calcAngle( atom1, atom2, atom3):
    b1 = atom2 - atom1
    b2 = atom2 - atom3
    print(b1, b2)
    b2db1 = np.sum( b1*b2 )
    b1L = np.sqrt(np.sum ( b1*b1) )
    b2L =np.sqrt( np.sum(b2*b2) )
    print(b1L)
    print(b2L)
    return 180.0/np.pi * np.arccos(  b2db1 / (b1L * b2L) )  

def runPolymer():
    trialForce = VDMD.HarmonicAngleForce("HarmonicAngleForce", [100.0, 179.0*np.pi/180.0], np.array([[0], [1], [2]] ) )
    forceComponents = trialForce.evaluate( np.array([ [0, -1, -0.01]  , [0.3, 0, 0], [0,1,0.01] ] ))
    print(forceComponents)
    #quit()

    
    outputFolder = "polymer_angle_100_v2"
    os.makedirs(outputFolder,exist_ok=True)
    addVirtual = True
    numRealDim = 3
    numVirtDim = 0
    if numVirtDim == 0:
        addVirtual = False

        
    numParticlesPerSide  = 11
    numParticles = numParticlesPerSide**3
    beadSigma = 0.1
    dt = 1e-4
    temp = 1.0 #
    #realBoxLength = 2.1

    realBoxLength = 1.0 * beadSigma *(numParticlesPerSide - 1.5 )
    virtualLengthScale = 2*beadSigma #RMS fluctuation size in the virtual dimension
    kbVal = 1.0
    config = VDMD.SimConfig( numRealDim,numVirtDim, temp ,dt, kbVal)
    #harmonic oscillator: equipartition implies 1/2 k <x^2> =  1/2 kbT
    virtualHarmonicConst = config.kbVal * temp / ( virtualLengthScale**2 )  
    print("VHC from length scale: ", virtualHarmonicConst)
    #alternate method from LJ-esque balancing
    #1/2 * k * alpha^2 == epsilon
    virtualHarmonicConst = 2*1.0 / (beadSigma ** 2)
    print("setting harmonic constant to", virtualHarmonicConst)
    
    
    xWall = VDMD.WallForce("XWall", [0, realBoxLength/2.0] )
    yWall = VDMD.WallForce("YWall", [1, realBoxLength/2.0] )
    zWall = VDMD.WallForce("ZWall", [2, realBoxLength/2.0] )
    config.addSPF(xWall)
    config.addSPF(yWall)
    config.addSPF(zWall)
    
    virtualConstraints = []
    for i in range(numVirtDim):
        virtualConstraints.append( VDMD.HarmonicForce("VirSqueeze"+str(i) , [i+numRealDim, virtualHarmonicConst] ) )
        config.addSPF( virtualConstraints[-1] ) 

    addHarmonic = True
    if addHarmonic == True:
        #atom1Set = np.arange( numParticles - numParticlesPerSide**2 )
        #atom2Set = atom1Set + numParticlesPerSide**2
        atom1Set = []
        atom2Set = []
        centralSet = []
        mols =  np.array_split( np.arange(numParticles), 11*11 )
        for mol in mols:
            print(mol)
            molAt1 = mol[:-1]
            molAt2 = molAt1 + 1
            atom1Set.append(molAt1)
            atom2Set.append(molAt2)
            centralSet.append( mol[1:-1] )
        atom1Set = np.array(atom1Set).flatten()
        atom2Set = np.array(atom2Set).flatten()
        centralSet = np.array(centralSet).flatten()
        print(atom1Set[:25])
        print(atom2Set[:25])
        print(centralSet)
        #quit()
        bondForce = VDMD.HarmonicBondForce( "HarmonicBondForce" ,  [2000, 0.1 ] , [atom1Set,atom2Set] )
        bondAngleForce = VDMD.HarmonicAngleForce("HarmonicAngleForce", [100.0, 179.0*np.pi/180.0], [centralSet  - 1, centralSet, centralSet + 1] )
        config.addPPF(bondForce)
        config.addPPF(bondAngleForce)
    ljSigma = [beadSigma] * numParticles
    ljEps = [1.0] * numParticles
    ljEpsMatrix = np.sqrt( np.outer( ljEps, ljEps ) )
    atomLabels =["A"]*numParticles


    ljSigmaMatrix = 0.5*np.add.outer( ljSigma, ljSigma)
    ljForce = VDMD.LJ612( "LJ612", [ ljEpsMatrix, ljSigmaMatrix ],"", maskSize = numParticles) 
    config.addPPF(ljForce)

    
    positions = np.zeros( (numParticles, numRealDim+numVirtDim) )
    velocities = np.zeros_like(positions)
    for i in range( numRealDim + numVirtDim):
        velocities[:,i] = np.random.normal(0, np.sqrt(config.kbVal*temp/config.mass) , numParticles )
    for i in range(numRealDim):
        positions[:,i] = np.random.uniform( -realBoxLength/2.0, realBoxLength/2.0, numParticles )

    initVirtual = True
    if initVirtual == True:
        for i in range(  numVirtDim):
            velocities[:,i+numRealDim] = np.random.normal(0, np.sqrt(config.kbVal*temp/config.mass) , numParticles )        
        #for i in range(numVirtDim):
        #    positions[:, i+numRealDim] = np.random.uniform( -virtualLengthScale, virtualLengthScale, numParticles )

    cubicInitialise = True
    if cubicInitialise == True:
        basisPoints = np.linspace( -realBoxLength/2.0, realBoxLength/2.0, num=numParticlesPerSide , endpoint=True)
        for p in range( numParticles ):
            xi = p % numParticlesPerSide
            yi = int( p / numParticlesPerSide) % numParticlesPerSide
            zi = int( p / numParticlesPerSide**2 )  % numParticlesPerSide
            positions[p,0] = basisPoints[xi]
            positions[p,1] = basisPoints[yi]
            positions[p,2] = basisPoints[zi]
            #print(p, xi,yi,zi, positions[p])



    forces = VDMD.getForces(positions,config)
    offsetVec = [numVirtDim*beadSigma *(numParticlesPerSide), 0, 0]
    dimStr = "d"+str(numRealDim)+"_"+str(numVirtDim)
    if addVirtual == False:
        xyzOut = open(outputFolder+"/polymerangle100_coords_xyz_"+dimStr+".xyz","w")
        offsetVec =  [0,0,0]
    else:
        xyzOut = open(outputFolder+"/polymerangle100_coords_xyz_"+dimStr+".xyz","w")
        xyaOut = open(outputFolder+"/polymerangle100_coords_xya_"+dimStr+".xyz","w")
    for i in range(50000):
        #positions,velocities,forces = VerletStep(positions, velocities,forces,config)
        if i == 40000:
            print("restoring reality")
            config.allowVirtual = False
        positions,velocities,forces = VDMD.GJFStep(positions, velocities,forces,config)
        print(i, VDMD.calcRealTemp(velocities, config) , VDMD.calcUnreality(positions, config) , positions[0], positions[1], np.sqrt( np.sum( (positions[0] - positions[1])**2  )) , calcAngle( positions[0], positions[1], positions[2]) )
        if i % 100 == 0:
            VDMD.writeXYZFrame(xyzOut, positions, atomLabels, [0,1,2], offset=offsetVec)
            if addVirtual == True:
                VDMD.writeXYZFrame(xyaOut, positions, atomLabels, [0,1,3],offset=offsetVec)
        #distSet = getDistMatrix(positions)
        #minDist = np.amin ( distSet[ distSet > 1e-10 ] ) 
        #print( "current min dist: ", minDist )
    xyzOut.close()
    xyaOut.close()
if __name__ == '__main__':
    runPolymer()
