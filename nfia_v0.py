#!/usr/bin/env python

import sys
import os
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import Image
from pylab import *
from scipy import optimize

class SifFile:
    """SifFile is the Python representation of an Andor SIF image
    file. Image data is stored in a numpy array indexed as [row,
    column] instead of [x, y]."""

    def __init__(self, path=""):
        self.data = 0
        if path != "":
            self.open(path)

    def __add__(self, other):
        new_sif = self.__class__()
        new_sif.__dict__ = self.__dict__
        if isinstance(other, (SifFile, np.ndarray)):
            new_sif.data = new_sif.data + other.data
        elif isinstance(other, (int, float)):
            new_sif.data = new_sif.data + other
        else:
            raise TypeError("Addition of SIF data requires another SifFile instance, a numpy array, or a scalar.")
        return new_sif

    def __sub__(self, other):
        new_sif = self.__class__()
        new_sif.__dict__ = self.__dict__
        if isinstance(other, (SifFile, np.ndarray)):
            new_sif.data = new_sif.data - other.data
        elif isinstance(other, (int, float)):
            new_sif.data = new_sif.data - other
        else:
            raise TypeError("Subtraction of SIF data requires another SifFile instance, a numpy array, or a scalar.")
        return new_sif

    def __mul__(self, other):
        new_sif = self.__class__()
        new_sif.__dict__ = self.__dict__
        if isinstance(other, (SifFile, np.ndarray)):
            new_sif.data = new_sif.data * other.data
        elif isinstance(other, (int, float)):
            new_sif.data = new_sif.data * other
        else:
            raise TypeError("Multiplcation of SIF data requires another SifFile instance, a numpy array, or a scalar.")
        return new_sif

    def __rmul__(self, other):
        return self.__mul__(other)

    def open(self, path):
        """Opens the SIF file at path and stores the data in
        self.data."""
        sif = open(path, "rb")

        # Verify we have a SIF file
        if sif.readline().strip() != "Andor Technology Multi-Channel File":
            sif.close()
            raise Exception("File %s is not an Andor SIF file." % path)

        # Ignore lines until we get to camera model
        for i in range(2):
            sif.readline()

        # Get camera model
        self.cammodel = sif.readline().strip()

        # Get CCD dimension in pixels
        shape = sif.readline().split()
        self.ccd_size = (int(shape[0]), int(shape[1]))

        # Ignore next 24 (17 with old version of sif) lines prior to superpixeling information
        for i in range(24):
            sif.readline()

        # Read superpixeling data
        line = sif.readline().split()
        #self.shape = (self.ccd_size[1]/int(line[5]), self.ccd_size[0]/int(line[6]))
        print line
        # Skip next 2 lines (1 with old version of sif)
        for i in range(2):
            sif.readline()

        # Read data
        self.data = np.fromstring(sif.read(), dtype=np.float32)
        self.data = self.data[:len(self.data)-2]
        if line[3] < line[2]:
            self.shape = (len(self.data)/int(line[3]), int(line[3]))
        else:
            # I'm not sure if this is correct...
            # Needs more testing.
            self.shape = (int(line[2]), len(self.data)/int(line[2]))
        self.data = np.reshape(self.data, self.shape)
        sif.close()
        
    
        
class fileName():

	def __init__(self,fname):
		self.fname = fname
	
	def _getpos(self,char):
		return self.fname.find(char)
	
	def _getsize(self,char1,char2):
		"""string size of the parameter value"""
		pos1 = self._getpos(char1)
		pos2 = self._getpos(char2)-len(char2)+1
		return pos2-pos1
		
	def getvalue(self,param,unit):
		"""return parameter value between parameter and unit"""
		size = self._getsize(param,unit)
		pos = self._getpos(param)+len(param)
		return float(self.fname[pos:pos+size])
	
	def getdate(self):
		""" return data acquisition date"""
		return self.fname[:8]
	
	def getshotnumber(self,extension):
		"""return shot number"""
		pos4 = self.fname.find('.'+extension)
		pos3 = self.fname.rindex('_')+1
		return int(self.fname[pos3:pos4])
			
def dir_scan(path,extension):
	"""return list of the filename with .extension"""
	extension = '*.'+extension
	filenames = glob.glob( os.path.join(path, extension))
	return filenames	

# function background sub choix bg ou interp

class backgroundCorrection():
	"""several tools for background correction and information about background"""
	def __init__(self,image,bgCorrection,path = None):
		self.image = image
		self.bgCorrection = bgCorrection
		if bgCorrection == None: 
			bgCorrection == False	
		if path == None:
			self.path = ""
			
	def backgroundInterp(self):
		"""4 points linear interpolation of background based on corner image count"""
		size_x,size_y = self.image.shape
		background = np.ones((size_x,size_y))
		marge = 50
		corner = np.array([self.image[marge,marge],self.image[size_y-marge,marge],self.image[size_x-marge,marge],self.image[size_x-marge,size_y-marge]])
		#for i in range(size_x):
		#	for j in range(size_y):
				#background[i,j] = (corner[1]-corner[0])*i/size_x+(corner[3]-corner[2])*j/size_y
		background = np.mean(corner)*background
		return background
		
	def getBgcImage(self):
		"""return background image"""
		bgFilename = glob.glob( os.path.join(self.path,'bg'))
		if bgFilename == []:
			backgroundImage = self.backgroundInterp()
		else:
			backgroundImage = SifFile(bgFilename[0])
		return backgroundImage
		
	def applyBgCorrection(self):
		"""apply correction to image"""
		if self.bgCorrection == True:
			self.image = self.image-self.getBgcImage()
		else:
			self.image = self.image
		return self.image
			
			
# voir pour liminer les pixels brillants

# function crop avec ginput

def cropImage(imagedata):
	size_x,size_y = imagedata.shape
	axis([0, size_x, 0, size_y])
	imshow(imagedata, interpolation='nearest')
	print "Please click to draw you square selection clockwise"
	pts = ginput(5) # it will wait for three clicks
	print "The point selected are"
	print pts # ginput returns points as tuples
	x=map(lambda x: x[0],pts) # map applies the function passed as 
	y=map(lambda x: x[1],pts) # first parameter to each element of pts
	plot(x,y,'-o')
	axis([0, size_x, 0, size_y])
	show()
	pta = np.array(pts,dtype=int16)
	length_x = int((pta[0,1]-pta[0,0])/2)
	#print length_x
	length_y = int((pta[2,1]-pta[3,1])/2)
	#print length_y
	ptc = pta[4,:]
	#print ptc
	return ptc, length_x, length_y
	
def roiImage(imagedata,ptc,length_x,length_y):
	return imagedata[ptc[1]-length_y:ptc[1]+length_y,ptc[0]-length_x:ptc[0]+length_x]
	
# function fit gaussian 2d

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p
 		
# function sum_count, methode threshold


# function max pos

#def maxImage(image):
			
#****************************************************************************************
if __name__=="__main__":
	testname = '20130123_2p0J_t10ps_dt05ns__19.sif'
	# path of the current folder
	path = ""
	extension = 'sif'
	fileNames = dir_scan(path,extension)
	# file number
	nbf = len(fileNames)
	print 'Number of files' : nbf
	
	#parameter1
	param1 = '_t'
	#unit 1
	unit1 = 'ps_'
	#parameter1
	param2 = '_dt'
	#unit 2
	unit2 = 'ns_'
	
	# extract date
	date = fileNames[0].getdate()
	
	# datafile name where you record measurement
	datafileMeas = 'data'+date+'.txt'
	
	# define the region of interest on the first image
	imageTest = SifFile(fileNames[0]).data[300:700,300:700]
	roic,roi_x,roi_y = cropImage(imageTest)
	roiData = roiImage(imageTest,roic,roi_x,roi_y)
	print 'Selected ROI:,' roiData.shape
	figure = plt.figure(2)
	imshow(roiData,interpolation='nearest')
	plt.show()
	
	

	for in range(nbf):
	
	#open SIF image with pre-crop
	imageData = SifFile(testname).data[
	
	#data background correction  
	imageDataC = backgroundCorrection(imageData,True).applyBgCorrection()
	#imageBackground = backgroundCorrection(imageData,True).getBgcImage()
	
	print fname.getvalue('_t','ps_')
	print fname.getshotnumber('sif')
	
	
	# test
	params = fitgaussian(roiData)
	print 'max :',np.max(np.max(roiData))
	print 'params:', params[0]
	fit = gaussian(*params)
	levels = np.array((0.0214,0.1359,0.5))*np.max(np.max(roiData))
	print 'levels:',levels
	
	# figure =plt.figure(3)
# 	contour(fit(*indices(roiData.shape)),levels,cmap=cm.copper)
# 	contourf(roiData,256,cmap=cm.jet)
# 	colorbar()
# 	ax = gca()
# 	(height, x, y, width_x, width_y) = params
# 	text(0.95, 0.05, """
# 	x : %.1f
# 	y : %.1f
# 	width_x : %.1f
# 	width_y : %.1f""" %(x, y, width_x, width_y),
# 	fontsize=16, horizontalalignment='right',
# 	verticalalignment='bottom', transform=ax.transAxes)
# 	plt.show()
	
	# Integrated Count Calculation by mask defined by gaussian fit
	mask1 = np.zeros(roiData.shape)
	for i in range(roiData.shape[0]):
		for j in range(roiData.shape[1]):
			test = (i-x)**2/(2*width_x)**2+(j-y)**2/(2*width_y)**2
			if test < 1:
				mask1[i,j] = 1
			else:
				mask1[i,j] = 0
	count1 = np.sum(roiData*mask1)
	print 'Count by Gauss Int :', count1	
	
	# Integrated Count Calculation by mask defined by threshold
	mask2 = roiData > levels[1]
	sint = mask2*roiData
	count2 = np.sum(sint)
	print 'Count by thresholding:',count2
	
