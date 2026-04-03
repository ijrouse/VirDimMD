'''
This code is intended only as a demonstration of the algorithm

Internal units assume energy is expressed in terms of kbT with T = 300 K. The temperature specified is scaled to 300K and used for initial velocities and noise.

I. Rouse 25/03/2026
'''
import numpy as np

class SimConfig:
    def __init__(self, realDims=3, virtDims=0, temp=1.0, dt=1e-5, kbVal = 1.0):
        self.realDims = realDims
        self.virtDims = virtDims
        self.temp = temp
        self.singleParticleForces = []
        self.pairParticleForces = []
        self.dt = dt
        self.mass = 1.0
        self.langevinGamma = 0.01/dt
        self.kbVal = kbVal
        self.forceCap = 1000000
        self.allowVirtual = True
        self.virtualPenaltyForceConst = 1000.0 #energetic penalty applied in terms of an additional force to virtual dimensions when allowVirtual = false
    def getSPF(self):
        return self.singleParticleForces
    def getPPF(self):
        return self.pairParticleForces
    def addSPF(self, spf):
        updatedSPF = self.singleParticleForces
        updatedSPF.append( spf)
        self.singleParticleForces = updatedSPF
    def addPPF(self, ppf):
        updatedPPF = self.pairParticleForces
        updatedPPF.append( ppf)
        self.pairParticleForces = updatedPPF

class SPF:
    def __init__(self,label,params):
        self.label=label
        self.params = params
    def updateParams(newParams):
        self.params = newParams
        
class WallForce(SPF):
    def evaluate(self,positions):
        force = np.zeros_like(positions)
        #print(positions)
        startLoc = self.params[1]
        axis = self.params[0]
        mask = np.abs(positions[:, axis]) > startLoc
        sign = np.sign(positions[:,axis])
        sigma = 0.01
        deltaE = 10.0
        wallForceConst = 4.0 * deltaE/(sigma**4)
        strength = -wallForceConst *( np.abs( positions[:,axis] ) - startLoc)**3
        #print(mask)
        #print(sign)
        #print(positions[0,axis])
        force[:,axis] = np.where( mask, strength*sign  , 0.0 )
        #print(force[0,axis])
        return force
class HarmonicForce(SPF):
    def evaluate(self,positions):
        force = np.zeros_like(positions)
        forceConst = self.params[1]
        axis = self.params[0]
        force[:,axis] = -1.0 * forceConst * positions[:, axis]
        return force


class PPF:
    def __init__(self,label,params,pairs,maskSize=-1):
        self.label=label
        self.params = params
        self.pairs = pairs
        if maskSize > 0:
            self.selfIntMask = np.eye( maskSize  ,dtype=bool)
class LJ612(PPF):
    '''Params[0] contains the epsilon matrix, params[1] contains the sigma matrix'''
    def evaluate(self,positions):
        params = self.params
        force = np.zeros_like(positions)
        numParticles = len(positions)
        numDims = len( positions[0] )
        distSet = -1.0*(np.reshape(positions, (1, numParticles, numDims) )  - np.reshape(positions, (numParticles,1,numDims) ) )

        distSqMatrix = np.sum(distSet**2  , axis=-1)
        distSqMatrix[self.selfIntMask] = 100000.0 #to avoid nans during processing
        invDistSq =  np.reshape( 1.0/distSqMatrix, (numParticles,numParticles,1) )
        ljMatrix = np.reshape( params[0], (numParticles,numParticles,1) )
        sigmaMatrix = np.reshape( params[1], (numParticles,numParticles,1) )
        forceTerms = 4.0 * ljMatrix *( 12.0 * invDistSq**7 * sigmaMatrix**12 * distSet  - 6.0 * invDistSq**4 * sigmaMatrix**6 * distSet )
        force = np.sum(forceTerms, axis=-2)
        return force

class LJ612DH(PPF):
    '''Params[0] contains the epsilon matrix, params[1] contains the sigma matrix, params[2] contains the charge matrix, params[3] contains the DH screening constant '''
    def evaluate(self,positions):
        params = self.params
        force = np.zeros_like(positions)
        numParticles = len(positions)
        numDims = len( positions[0] )
        distSet = -1*(np.reshape(positions, (1, numParticles, numDims) )  - np.reshape(positions, (numParticles,1,numDims) ) )
        dhKappa = params[3]
        epsRel = 80.0
        '''
        inv4PiEps0 = 9e9 #N m^2 C^-2
        elemCharge = 1.6e-19
        chargeForceConst = elemCharge**2 *inv4PiEps0 #N m^2 = J m
        '''
        chargeForceConstKBTNM = 55.628 
        distSqMatrix = np.sum(distSet**2  , axis=-1)
        distSqMatrix[self.selfIntMask] = 100000.0 #to avoid nans during processing
        distMatrix = np.sqrt(distSqMatrix)
        invDistSq =  np.reshape( 1.0/distSqMatrix, (numParticles,numParticles,1) )
        ljMatrix = np.reshape( params[0], (numParticles,numParticles,1) )
        sigmaMatrix = np.reshape( params[1], (numParticles,numParticles,1) )
        forceTerms = 4.0 * ljMatrix *( 12.0 * invDistSq**7 * sigmaMatrix**12 * distSet  - 6.0 * invDistSq**4 * sigmaMatrix**6 * distSet )
        forceTerms += params[2] * chargeForceConstKBTNM* np.exp(-dhKappa * distMatrix) * distSet*(1 + distMatrix*dhKappa)/(     distSqMatrix**(1.5)   )
        force = np.sum(forceTerms, axis=-2)
        return force

class HarmonicBondForce(PPF):
    ''' params[0] contains the bond constants k , params[1] contains the ideal distance, pairs[0] is the set of atoms 1, pairs[1] is the set of atoms 2'''
    def evaluate(self, positions):
        params = self.params
        force = np.zeros_like(positions)
        atom1 = self.pairs[0]
        atom2 = self.pairs[1]
        numBonds = len(atom1)
        numDims = len( positions[0] )
        distSet = positions[atom1] - positions[atom2]
        scalarDist = np.sqrt(  np.sum(distSet**2 ,axis=-1 ))
        #debugAtom1 = np.argmax(scalarDist)
        bondConst = params[0]
        #print(scalarDist.shape )
        #print(bondConst.shape)
        #print(distSet.shape)
        #print(params[1].shape)
        forceMag = np.reshape( bondConst * (scalarDist - params[1] ) / ( scalarDist + 1e-10 ), (-1,1) )
        forceTerms1 = - forceMag *distSet 
        forceTerms2 = forceMag * distSet 
        force[atom1] += forceTerms1
        force[atom2] += forceTerms2
        return force


class HarmonicAngleForce(PPF):
    ''' params[0] contains the bond constants k , params[1] contains the ideal angle in radians'''
    def evaluate(self, positions):
        force = np.zeros_like(positions)
        atom1 = self.pairs[0]
        atom2 = self.pairs[1]
        atom3 = self.pairs[2]
        bondConsts = self.params[0]
        bondAngle0 = self.params[1]
        bond1 = positions[ atom2] - positions[atom1]
        bond2 = positions[atom2] - positions[atom3]
        #print(bond1 * bond2)
        bondEps = 1e-5 #minimum bond length to ensure numerical stability
        bond1L = np.maximum( np.sqrt(   np.sum(bond1**2, axis=-1) ) , bondEps)
        bond2L = np.maximum(np.sqrt(np.sum(bond2**2,axis=-1)) ,bondEps)
        b1dotb2 = np.sum( bond1*bond2, axis=-1)
        #print(b1dotb2)
        #print(bond1L)
        #print(bond2L)
        
        acosArg =  np.clip( b1dotb2/( bond1L*bond2L  ), -0.9999,0.9999 ) #numerically safe value 
        #print(acosArg)
        bondAngle = np.arccos( acosArg) #back-project the safe value to an equivalent 
        #print(bondAngle)
        prefactor = (bondAngle -bondAngle0 ) * bondConsts / np.sqrt( 1 - acosArg**2 )
        prefactor = np.reshape( prefactor, (-1,1))
        acosArg = np.reshape( acosArg, (-1,1) )
        bond1L = np.reshape( bond1L, (-1,1))
        bond2L = np.reshape(bond2L, (-1,1) )
        forceTerms1 = prefactor*( acosArg * bond2L * bond1 - bond1L * bond2)/( bond1L**2 * bond2L)
        forceTerms2 = prefactor* (- acosArg * bond1/(bond1L**2) + ( bond2 + bond1)/( bond1L * bond2L) - bond2*acosArg/(bond2L**2) )
        forceTerms3 = prefactor * (   -1.0*bond2L * bond1 + acosArg * bond1L * bond2)/ (   bond1L * bond2L**2 )
        force[atom1] += forceTerms1
        force[atom2] += forceTerms2
        force[atom3] += forceTerms3
        return force

    


def getDistMatrix(positions):
        numParticles = len(positions)
        numDims = len( positions[0] )
        distSet = -1.0*(np.reshape(positions, (1, numParticles, numDims) )  - np.reshape(positions, (numParticles,1,numDims) ) )

        distSqMatrix = np.sum(distSet**2  , axis=-1)
        return np.sqrt(distSqMatrix)
    
def getForces(positions,config):
    numAtoms = len(positions)
    forces = np.zeros_like(positions)
    #1-particle forces
    for spf in config.getSPF():
        #print(spf.label)
        forces += spf.evaluate(positions)
    for ppf in config.getPPF():
        #print(ppf.label)
        forces += ppf.evaluate(positions)
    return np.clip( forces, -config.forceCap, config.forceCap )

def VerletStep( positions, velocities,forces,config):
    halfVel = velocities + forces/(2.0*config.mass ) * config.dt
    positions = positions + halfVel*config.dt
    forces = getForces( positions, config)
    velocities = halfVel + forces/(2.0 * config.mass) * config.dt
    return positions, velocities ,forces


def GJFStep(positions, velocities, forces, config):
    '''Langevin integration step from https://arxiv.org/pdf/1212.1244 '''
    noise = np.random.normal( 0.0, np.sqrt( 2.0 * config.langevinGamma * config.kbVal *  config.temp * config.dt ), positions.shape )

    if config.allowVirtual == False:
        #noise[:, config.realDims:] = np.where(   np.logical_and( positions[:, config.realDims: ]  > 0  ,noise[:, config.realDims:] > 0 )  ,  0, noise[:, config.realDims:] )
        #noise[:, config.realDims:] = np.where(   np.logical_and( positions[:, config.realDims: ]  < 0  ,noise[:, config.realDims:] < 0 )  ,  0, noise[:, config.realDims:] )
        noise[:, config.realDims:] = 0.0
    #noise[:,3] = 0
    b = 1.0/(1.0 + config.langevinGamma*config.dt/(2*config.mass) )
    dtMass = config.dt/(2.0 * config.mass)
    positionsNew = positions + b * config.dt * velocities + b * config.dt * dtMass * forces + b * dtMass * noise
    #print( (positionsNew - positions)[:2] )
    forcesNew = getForces(positionsNew,config)

    if config.allowVirtual == False:
        forcesNew[:, config.realDims:] = -virtualPenaltyForceConst * positionsNew 
    #    forcesNew[:, config.realDims:] = np.where(   np.logical_and( positions[:, config.realDims: ]  > 0  ,forcesNew[:, config.realDims:] > 0 )  ,  0, forcesNew[:, config.realDims:] )
    #    forcesNew[:, config.realDims:] = np.where(   np.logical_and( positions[:, config.realDims: ]  < 0  ,forcesNew[:, config.realDims:] < 0 )  ,  0, forcesNew[:, config.realDims:] )
    
    velocitiesNew = velocities + dtMass*(forces + forcesNew) - config.langevinGamma/config.mass*( positionsNew - positions) + 1.0/config.mass*noise

    #virtualMask = np.zeros_like( positions, dtype=bool)
    if config.allowVirtual == False:
        velocitiesNew[:, config.realDims:] = np.where(   np.logical_and( positions[:, config.realDims: ]  > 0  ,velocitiesNew[:, config.realDims:] > 0 )  ,  0, velocitiesNew[:, config.realDims:] )
        velocitiesNew[:, config.realDims:] = np.where(   np.logical_and( positions[:, config.realDims: ]  < 0  ,velocitiesNew[:, config.realDims:] < 0 )  ,  0, velocitiesNew[:, config.realDims:] )
    return positionsNew, velocitiesNew, forcesNew

def calcRealTemp( velocities, config):
    #return np.sum( config.mass *  velocities[:, 0:config.realDims ]**2   ) *2.0 /(config.realDims * config.kbVal * len(velocities) * 3)
    #print("mean velocities", np.mean(  config.mass * velocities[:, 0:config.realDims]**2, axis= 0) )
    return np.mean(  config.mass * velocities[:, 0:config.realDims]**2, axis= 0)/(config.kbVal)  # == 1/2 kbT

def calcUnreality(positions, config):
    return np.sqrt( np.mean(   positions[:,  config.realDims:(config.realDims+config.virtDims)]**2, axis=0 ) )

def writeXYZFrame(fileHandle, positions, labels, axes, offset=[0,0,0], scale=10):
    fileHandle.write(str(len(positions))+"\n")
    fileHandle.write("comment \n")
    for i,atom in enumerate(positions):
        label = labels[i]
        fileHandle.write(label+" " +  str( scale* (offset[0]+atom[ axes[0] ] ))+" " + str(scale*  (offset[1]+atom[axes[1] ] )) + " " + str(scale* (offset[2] + atom[axes[2]])) + "\n" )
    fileHandle.flush()

def runDemo():
    addVirtual = True
    numRealDim = 3
    numVirtDim = 1
    if numVirtDim == 0:
        addVirtual = False

        
    numParticlesPerSide  = 11
    numParticles = numParticlesPerSide**3
    beadSigma = 0.1
    dt = 1e-4
    temp = 1.0 #
    #realBoxLength = 2.1

    realBoxLength = 1.0 * beadSigma *(numParticlesPerSide - 1.0)
    virtualLengthScale = 2*beadSigma #RMS fluctuation size in the virtual dimension
    kbVal = 1.0
    config = SimConfig( numRealDim,numVirtDim, temp ,dt, kbVal)
    #harmonic oscillator: equipartition implies 1/2 k <x^2> =  1/2 kbT
    virtualHarmonicConst = config.kbVal * temp / ( virtualLengthScale**2 )  
    print("VHC from length scale: ", virtualHarmonicConst)
    #alternate method from LJ-esque balancing
    #1/2 * k * alpha^2 == epsilon
    virtualHarmonicConst = 2*1.0 / (beadSigma ** 2)
    print("setting harmonic constant to", virtualHarmonicConst)
    
    
    xWall = WallForce("XWall", [0, realBoxLength/2.0] )
    yWall = WallForce("YWall", [1, realBoxLength/2.0] )
    zWall = WallForce("ZWall", [2, realBoxLength/2.0] )
    config.addSPF(xWall)
    config.addSPF(yWall)
    config.addSPF(zWall)
    
    virtualConstraints = []
    for i in range(numVirtDim):
        virtualConstraints.append( HarmonicForce("VirSqueeze"+str(i) , [i+numRealDim, virtualHarmonicConst] ) )
        config.addSPF( virtualConstraints[-1] ) 

    addHarmonic = True
    if addHarmonic == True:
        atom1Set = np.arange( numParticles - numParticlesPerSide**2 )
        atom2Set = atom1Set + numParticlesPerSide**2
        bondForce = HarmonicBondForce( "HarmonicBondForce" ,  [200, 0.1 ] , [atom1Set,atom2Set] )
        config.addPPF(bondForce)
    ljSigma = [beadSigma] * numParticles
    ljEps = [1.0] * numParticles
    ljEpsMatrix = np.sqrt( np.outer( ljEps, ljEps ) )
    atomLabels =["A"]*numParticles
    #override the ljEpsMatrix to produce one with alternating [ [1,0,1,0... ] [0, 1, 0, 1 ... ] ... ]
    #such that each particle has favourable interactions with half of the other particles
    generateImmiscible = True
    if generateImmiscible == True:
        immEps = 0.1
        #ljEpsMatrix = np.indices(ljEpsMatrix).sum(axis=0) % 2
        ljEpsMatrix[1::2, ::2] = immEps
        ljEpsMatrix[::2, 1::2] = immEps
        print(ljEpsMatrix)
        #ljEpsMatrix = ljEpsMatrix/2.0 + 0.5 #map to 1, 0.5, 1, 0.5 ...
        for i in range(numParticles):
            if i%2 == 1:
                atomLabels[i] = "B"
        print(atomLabels)
    ljSigmaMatrix = 0.5*np.add.outer( ljSigma, ljSigma)
    ljForce = LJ612( "LJ612", [ ljEpsMatrix, ljSigmaMatrix ],"", maskSize = numParticles) 
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



    forces = getForces(positions,config)
    offsetVec = [numVirtDim*beadSigma *(numParticlesPerSide), 0, 0]
    if addVirtual == False:
        xyzOut = open("coords_novirt_bondimmisc0p1_d0.xyz","w")
        offsetVec =  [0,0,0]
    else:
        xyzOut = open("coords_withvirt_bondimmisc0p1_d"+str(numVirtDim)+".xyz","w")
        xyaOut = open("coords_alpha_bondimmisc0p1_d"+str(numVirtDim)+".xyz","w")
    for i in range(50000):
        #positions,velocities,forces = VerletStep(positions, velocities,forces,config)
        if i == 40000:
            print("restoring reality")
            config.allowVirtual = False
        positions,velocities,forces = GJFStep(positions, velocities,forces,config)
        print(i, calcRealTemp(velocities, config) , calcUnreality(positions, config))
        if i % 100 == 0:
            writeXYZFrame(xyzOut, positions, atomLabels, [0,1,2], offset=offsetVec)
            if addVirtual == True:
                writeXYZFrame(xyaOut, positions, atomLabels, [0,1,3],offset=offsetVec)
        #distSet = getDistMatrix(positions)
        #minDist = np.amin ( distSet[ distSet > 1e-10 ] ) 
        #print( "current min dist: ", minDist )
    xyzOut.close()
    xyaOut.close()
if __name__ == '__main__':
    runDemo()
