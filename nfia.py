#!/usr/bin/env python

import sys
import os
import time
import numpy as np
import glob
import math
import matplotlib
matplotlib.use('TkAgg')
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
        #print line
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
		pos2 = self._getpos(char2)
		return pos2-pos1-len(char1)
		
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
		#print self.image.shape
		background = np.ones((size_x,size_y))
		marge = 10
		corner = np.array([self.image[marge,marge],self.image[size_x-marge,marge],self.image[marge,size_y-marge],self.image[size_x-marge,size_y-marge]])
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
	figcrop = plt.figure(0)
	imshow(imagedata, interpolation='nearest',origin = 'lower')
	print "Please click to center of spot"
	ptc = ginput(1) # it will wait for three clicks
	print "The point selected are"
	ptc = np.array((ptc[0]),dtype=int16)
	print ptc # ginput returns points as tuples
	plt.show()
	plt.close(0)
	length_x = 75
	length_y = 75
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

def averageColumn(col1,col2,matrix):
	elt = np.zeros((matrix.shape[0],4))        
	elt[0,0] =  matrix[0,col1]
	elt[0,1] = matrix[np.where(matrix[:,col1]==matrix[0,col1])].shape[0]
	elt[0,2] = matrix[np.where(matrix[:,col1]==matrix[0,col1]),col2].sum()
	elt[0,3] =  matrix[np.where(matrix[:,col1]==matrix[0,col1]),col2].sum()/matrix[np.where(matrix[:,0]==matrix[0,0])].shape[0]
	j = 0
	for i in range(1,matrix.shape[0]):
		if (matrix[i,col1] != matrix[i-1,col1]) and ((matrix[i,col1] != elt[:,0]).all()):
			j = j+1
			elt[j,0]=matrix[i,col1]
			elt[j,1]=matrix[np.where(matrix[:,col1]==matrix[i,col1])].shape[0]
			elt[j,2]=matrix[np.where(matrix[:,col1]==matrix[i,col1]),col2].sum()
			elt[j,3]=matrix[np.where(matrix[:,col1]==matrix[i,col1]),col2].sum()/matrix[np.where(matrix[:,col1]==matrix[i,col1])].shape[0]
	elements = elt[np.where(elt>0)]
	results = elements.reshape(-1,4)
	return results
 		
		
#****************************************************************************************
if __name__=="__main__":
	print "Matlplotlib backEnd", matplotlib.pyplot.get_backend()
	testname = '20130123_2p0J_t10ps_dt05ns__19.sif'
	# path of the current folder
	path = ""
	extension = 'sif'
	fileNames = dir_scan(path,extension)
	# file number
	nbf = len(fileNames)
	print 'Number of files :', nbf
	# data initialisation array
	dataline = np.empty((nbf,11))
	dataline_average = np.zeros((nbf,11))	
	#parameter1
	param1 = '_t'
	#unit 1
	unit1 = 'ps_'
	#parameter1
	param2 = '_dt'
	#unit 2
	unit2 = 'ns_'
	
	# extract date
	date = fileName(fileNames[0]).getdate()
	print date
	# datafile name where you record measurement
	datafileMeas = 'data'+str(date)+'.txt'
	datafileMeas2 = 'data'+str(date)+'_average'+'.txt'
	# define headers
	headers = 'date\t'+'shotnumber'+'\t'+param1+' '+unit1+'\t'+param2+' '+unit2+'\tS_int_th[cc]\tS_int_gauss\tS_max[cc]\tx_max[pix]\tymax[pix]\tsig_x[pix]\tsig_y[pix]'
	# write headers
	f = open(datafileMeas,'ab')
	f.write(headers)
	f.close()
	f = open(datafileMeas2,'ab')
	f.write(headers)
	f.close()
	
	# define the region of interest on the first image
	imageTest = SifFile(fileNames[10]).data[700:,200:600]
	roic,roi_x,roi_y = cropImage(imageTest)
	roiData = roiImage(imageTest,roic,roi_x,roi_y)
	print 'Selected ROI:', roiData.shape
	figure = plt.figure(1)
	plt.title('ROI Image')
	plt.imshow(roiData,interpolation='nearest', origin = 'lower')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.colorbar()
	plt.show()
	plt.close(1)
	
	
	fig2 = plt.figure(2)
	ion()
	# main loop on files 
	for  l in range(nbf):
		print 'fileNumber :', l
		#open SIF image with pre-crop
		imageData = SifFile(fileNames[l]).data[700:,200:600]
		#data background correction 
		imageDataC = backgroundCorrection(imageData,True).applyBgCorrection()
		#background count
		imageBackground = backgroundCorrection(imageData,True).getBgcImage()
		print 'Background mean total count :', np.mean(imageBackground) 
		
		#definition of the ROI image 
		roiData = roiImage(imageDataC,roic,roi_x,roi_y)
		
		value1 = fileName(fileNames[l]).getvalue(param1,unit1)
		value2 = fileName(fileNames[l]).getvalue(param2,unit2)
		shotNumber = fileName(fileNames[l]).getshotnumber(extension)
		
		# 2D gaussian fit on the roiData
		params = fitgaussian(roiData)
		s_max = np.max(np.max(roiData))
		print 'max :', s_max
		#fit = gaussian(*params)
		
		# Integrated Count Calculation by mask defined by gaussian fit
		mask1 = np.zeros(roiData.shape)
		# times sigma of the gaussian for integrations
		fact = 1.5
		for i in range(roiData.shape[0]):
			for j in range(roiData.shape[1]):
				test = (i-params[1])**2/(fact*params[3])**2+(j-params[2])**2/(fact*params[4])**2
				if test < 1:
					mask1[i,j] = 1
				else:
					mask1[i,j] = 0
		count1 = np.sum(roiData*mask1)
		print 'Count by Gauss Int :', int(count1)
		
		# Integrated Count Calculation by mask defined by threshold
		levels = np.array((0.0214,0.1359,0.15))*s_max
		# change index or value in levels to change threshold
		mask2 = roiData > levels[2]
		sint = mask2*roiData
		count2 = np.sum(sint)
		print 'Count by thresholding:',int(count2)
		
		# Plot each slice as an independent subplot
		data = [roiData*mask2,roiData*mask1]
		
		plt.subplot(121)
   		plt.imshow(data[0],interpolation = 'nearest',origin = 'lower')
   		plt.subplot(122)
   		plt.imshow(data[0],interpolation = 'nearest',origin = 'lower')
		#plt.imshow(roiData*mask2,cmap=plt.cm.hsv)
		fig2.canvas.draw()
		#plt.colorbar()
		time.sleep(1e-6)
		
		# store measurement in an array
		measurement = [int(date),shotNumber,value1,value2,count2,count1,s_max,params[1],params[2],params[3],params[4]]
		#print measurement
		dataline[l,:] = np.array(measurement)
		# average on same values
		

	results_average = averageColumn(3,4,dataline)		
	# save the measurement array	
	fid = open(datafileMeas,'ab')
	np.savetxt(fid,dataline,fmt='%s',delimiter='\t')
	print 'measurement of file is saved'
	fid.close()
	fid = open(datafileMeas2,'ab')
	np.savetxt(fid,results_average,fmt='%s',delimiter='\t')
	print 'average measurement of file is saved'
	fid.close()
	
	#figure = plt.figure(3)
	#plt.scatter(dataline_average[:,3],dataline_average[:,5])
	#plt.xlabel(param2)
	#plt.show()
	#plt.close('all')
	