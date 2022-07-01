# Contains routines for handling the brightness map of a star.
#Notes:
# This could be merged with magneticGeom.py or with geometryStellar.py pretty easily.  

import numpy as np

#Saves the brightness map as a set of pixels, 
# using the grid of colatitude and longitude supplied at initialization.  
class brightMap:
    def __init__(self, clat, lon):
        self.clat = clat
        self.lon = lon
        #assume we initialize to a uniform (normalized) brightness of one
        self.npt = clat.shape[0]
        self.bright = np.ones(self.npt)

    def makeRoundSpot(self, clat, lon, radius, brigthness):
        dist = np.arccos(np.sin(self.clat)*np.sin(clat)*np.cos(self.lon-lon) + np.cos(self.clat)*np.cos(clat))
        
        for i in range(self.npt):
            if( dist[i] < radius ):
                self.bright[i] = brigthness


#Save a brightness map (as in the class above), to a specified file name
def saveMap(brightMap, fileName):
    outFile = open(fileName, 'w')
    for i in range(brightMap.npt):
        outFile.write('%8.6f %8.6f %11.8f\n' % (brightMap.clat[i], brightMap.lon[i], brightMap.bright[i]) )


#Read a brightness map in from a file (same format as saveMap())
#Check the input map's stellar grid coordinates against the grid coordinates the user supplies.
#If the coordinate systems are not consistent return an error.
def readMap(fileName, userClat, userLon, verbose=1):

    clat, lon, bright = np.loadtxt(fileName, unpack=True)

    if ((clat.shape[0] != userClat.shape[0]) | (lon.shape[0] != userLon.shape[0])):
        print('ERROR reading map {:} requested coordinate grid shape does not match input coordinate grid shape'.format(fileName))
        return

    #small should be set to account for the precision of the coordinates used in the saved map.
    small = 1e-5
    if((np.abs(clat - userClat) < small).all() & (np.abs(lon - userLon) < small).all()):
        bMap = brightMap(clat, lon)
        bMap.bright = bright
        if(verbose == 1):
            print('Initialized brightness map from {:}'.format(fileName))
        return bMap
    else:
        print('ERROR reading map {:} requested coordinate grid does not match input coordinate grid'.format(fileName))
        return


def SetupBrightMap(sGrid, initBrightFromFile, initBrightFile, defaultBright, verbose=1):
    #initialize the brightness map from a file, or set to the 'default brightness'
    if (initBrightFromFile == 1):
        briMap = readMap(initBrightFile, sGrid.clat, sGrid.long, verbose)
    else:
        briMap = brightMap(sGrid.clat, sGrid.long)
        briMap.bright[:] = defaultBright
    
    return briMap

