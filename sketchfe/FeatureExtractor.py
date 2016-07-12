# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:31:44 2016

@author: kurmanbek kaiyrbekov
@author: ozan can altiok

"""

import Point
import Stroke
import Sketch
import math
from abc import ABCMeta, abstractmethod
import numpy as np
import shapecreator
import matplotlib.pyplot as plt
from scipy import ndimage
from random import randint

# an abstract class representing a feature extractor for sketches
class FeatureExtractor:
    __metaclass__ = ABCMeta
    
    # an abstract method supposed to give an output having features
    @abstractmethod
    def extract(self,sketch): pass

# an IDM feature extractor class
class IDMFeatureExtractor(FeatureExtractor):    
    def extract(self, sketch):
        #Necessary variables
        resample_interval = 50.0
        sigma = 10.0
        hsize = 4.0
        grid_width = 12.0 
        grid_height = 12.0
        plot = False
        #Resample each stroke at const spatial frequency 
        resampled_sketch = sketch.resample(resample_interval)     
        #normalize sketch
        n_sketch = resampled_sketch.normalize()
        #get angles for each stroke point
        theta = self.coords2angles( n_sketch )
        endpt = True
        
        featim1 = self.extractFeatureImage( theta, n_sketch, grid_width, grid_height, 0, not endpt, sigma, hsize  )
        featim2 = self.extractFeatureImage( theta, n_sketch, grid_width, grid_height, 45, not endpt, sigma, hsize  )
        featim3 = self.extractFeatureImage( theta, n_sketch, grid_width, grid_height, 90, not endpt, sigma, hsize  )
        featim4 = self.extractFeatureImage( theta, n_sketch, grid_width, grid_height, 135, not endpt, sigma, hsize  )
        featim5 = self.extractFeatureImage( theta, n_sketch, grid_width, grid_height, 0,  endpt, sigma, hsize  )
        
      #  print featim1, featim2, featim3, featim4, featim5
        
        feature = np.concatenate([featim1.flatten(1), featim2.flatten(1), featim3.flatten(1), featim4.flatten(1), featim5.flatten(1)])
        
        if plot:
            self.plotIDMFeature(featim1, featim2, featim3, featim4, featim5)
       
        return feature
    
    def plotIDMFeature(self, featim1, featim2, featim3, featim4, featim5 ):
        '''Plots all feature images
        '''
        plt.figure(randint(0, 20))
        plt.subplot(2, 3, 1)
        plt.imshow(featim1, origin = [0,0])
        plt.subplot(2, 3, 2)
        plt.imshow(featim2, origin = [0,0])
        plt.subplot(2, 3, 3)
        plt.imshow(featim3, origin = [0,0])
        plt.subplot(2, 3, 4)
        plt.imshow(featim4, origin = [0,0])
        plt.subplot(2, 3, 5)
        plt.imshow(featim5, origin = [0,0])
        plt.subplot(2, 3, 6)
        plt.imshow((featim1 + featim2 + featim3 + featim4 + featim5) / 5, origin = [0,0] )
      
    
    def extractFeatureImage( self,theta, sketch, grid_width, grid_height, cur_angle, endpt, sigma, hsize  ):
        '''Extracts a 24x24 feature image for each reference angle.
        '''
        center = np.array( [0, 0] )
        imsize = np.array([ 2 * grid_width, 2 * grid_height ])
        scale  = ( imsize )/ ( 2.5 * 2 )
        cur_angle2 = ( cur_angle + 180 ) % 360
        image = np.zeros( imsize )
        pixels = []
        
        if( endpt == False ):
            angle_dist1 = self.getAngleDistance( theta, cur_angle  )
            angle_dist2 = self.getAngleDistance( theta, cur_angle2 )

            for ind in range(0, len(angle_dist1) ):
                #Compute elementwise minimum of the angles
                min_dist = np.minimum( angle_dist1[ind], angle_dist2[ind] )
                #Assign actual pixel values of each stroke point 
                pix_values = self.pixelValues( min_dist )
                pixels.append( np.delete( pix_values, -1 ) )
                
            image = self.pointsToImage(sketch, imsize, pixels, center, scale) 
        else:
            for stroke in sketch.strokes:
                coords = stroke.listCoordinates()
                #get endpoints of stroke
                ends = coords[[0,-1], :]
                #render end points
                transformed = self.transform(ends, center, scale, imsize)
                #Assign 1.0 to endpoints
                for ind in range(0, len(transformed)):
                    image[ transformed[ind, 1], transformed[ind, 0] ] = 1.0 #WHY X is second coordinate?
                
        #smooth feature image
        image = self.smoothim(image, sigma, hsize)
        #downsample feature image
        image = self.downsample(image)

        return image
        
    def extractimage_test(self):
        sk1,sk2,sk3 = shapecreator.test()
        self.extract(sk3)
        return

    def pointsToImage(self, sketch, imsize, pixels, center, scale ):
        '''Assigns pixel values to a 24x24 grid.
        Returns 24x24 grid filled with the pixel values.
        '''
        image = np.zeros( imsize )
        for ind in range(0, len(sketch.strokes)):
            #Render coordinates of each stroke to range between 0-23
            transformed = self.transform( sketch.strokes[ind].listCoordinates(), center, scale, imsize )
            #put pixel values to all coordintes
            image = self.drawBresenham(image, transformed, pixels[ind])
            
        return image
        
    def drawBresenham( self, image, coords, pixels ):
        ''' Assigns (maximum) pixel values to a corresponding coordinates,
        using bresenham drawing algorithm. 
        '''
        for ind in range( 0, len(pixels) ):
            c = np.array([coords[ ind ], coords[ ind + 1 ]])

            if( pixels[ind] > 0 ):
                x, y = self.bresenham( c[0,0], c[0, 1], c[1,0], c[1,1])
                for j in range(0, len(x)):
                    if(image[y[j], x[j]] < pixels[ind]):
                        image[y[j], x[j]] = pixels[ind]
                        
        return image
            
            
            
    def bresenham( self, x1, y1, x2, y2 ):
        x1 = np.round( x1 )
        x2 = np.round( x2 )
        y1 = np.round( y1 )
        y2 = np.round( y2 )
        dx = np.abs( x2 - x1 )
        dy = np.abs( y2 - y1 )
        steep = np.abs( dy ) > np.abs( dx )
        
       # print steep, dx, dy, x1, x2, y1, y2
        if( steep ):
            temp = dx
            dx   = dy
            dy   = temp
            
        if( dy == 0 ):
            q = np.zeros((dx+1, 1))
        else:
            temp = np.array( [0] )
            arr = np.arange( np.floor( dx/2 ), -dy*dx + np.floor( dx/2 ) - 1, -dy )
            q = np.append( temp, np.diff( np.mod( arr, dx ) ) >= 0 )
         
    #    print q
        if( steep ):
            
            if y1 <= y2:
                y = np.arange( y1, y2 + 1 )
            else:
                y = np.arange( y1, y2 - 1, -1 )
                
            if x1 <= x2:
                x = x1 + np.cumsum(q)
            else:
                x = x1 - np.cumsum(q)
                
        else:
            
            if x1 <= x2:
                x = np.arange( x1, x2 + 1 )
            else:
                x = np.arange( x1, x2 - 1, -1 )
                
            if y1 <= y2:
                y = y1 + np.cumsum(q)
            else:
                y = y1 - np.cumsum(q)
        
      #  print(x, y)
                
        return ( x , y )
        
    def pixelValues(self, distance ): 
        '''Assign pixel values are calculated as a difference between stroke angle and the reference angle
        and vary linearly between 1.0(if the two are equal) and 0.0(if they differ by more than 45 degrees)
        Returns pixel values for each stroke.
        '''
        angle_threshold = 45
        valid_indices   = distance <= angle_threshold
        invalid_indices = -valid_indices
        distance[ valid_indices ]   = 1 - ( distance[ valid_indices ] / angle_threshold )
        distance[ invalid_indices ] = 0
        
        return distance
        
    def getAngleDistance(self, thetas, cur_angle ):        
        '''Calculates an angle difference betweeen reference angle and current angle
        and truncates it betweeen 0 and 180.
        Returns list of angle differences for each stroke.
        '''
        angle_dist = []
        
        for theta in thetas:
            #Get the difference between stroke angle and a reference angle 
            diff = theta - cur_angle
            #Make changes for truncation
            diff[diff >=  180] = diff[diff >=  180] - 360
            diff[diff <= -180] = diff[diff <= -180] + 360
            #Truncate angles between 0 and 180            
            angle_dist.append( np.absolute(diff) )
            
        return angle_dist
        
    # a tester method for coords2angles method of this class
    def coords2angles_test(self):
        sk1,sk2,sk3 = shapecreator.test()
        sr = sk1.resample(50)
        srn = sr.normalize()
        
        return self.coords2angles(srn)
        
    # transforming the sketch coordinates by given scale, around the center
    def transform(self,scoords,center,scale,imsize):
        '''Render each coordinate of the stroke to a 24x24 grid.
        Returns the coordinates of the stroke bounded between 0 and 23.
        '''
        n,scc = scoords.shape
        ones = np.array([1, 1])
        transformed = np.tile(imsize/2 - ones,(n,1)) + (np.tile(scale,(n,1))*(scoords - np.tile(center,(n,1))))
        transformed = np.floor(transformed)
        
        #Fit to bounding box if not fitted already
        mins = np.amin(transformed, axis = 0)
        if(mins[0] < 0):
            transformed[: , 0] = transformed[: , 0] - mins[0]
        if(mins[1] < 0):
            transformed[: , 1] = transformed[: , 1] - mins[1]
        
        maxs = np.amax(transformed, axis = 0) 
        if(maxs[0] > imsize[0] - 1):
            transformed[: , 0] = transformed[: , 0] * (imsize[0] - 1) / maxs[0]
        if(maxs[1] > imsize[1] - 1):
            transformed[: , 1] = transformed[: , 1] * (imsize[1] - 1) / maxs[1]
        transformed = np.floor(transformed)
        
        '''
        (trcount,dmm) = transformed.shape
        for i in range(0, dmm):
            for j in range(0, trcount):
                if transformed[j,i] > imsize[0] - 1:
                    transformed[j,i] = imsize[0] - 1
                
                if transformed[j,i] < 1:
                    transformed[j,i] = 1
     #   print( 'tranformed',  transformed )         '''
        return transformed
        
    # creates 2D gaussian filter with given sigma and kernel size
    def fgaussian(self,size, sigma):
        m,n = size
        h, k = m//2, n//2
        x, y = np.mgrid[-h:h, -k:k]
        # apply the bivariate gaussian distribution with zero mean
        return np.exp(-(x**2 + y**2)/(2*sigma**2))
        
    # downsampler method for a given image as numpy matrix
    def downsample(self,im):
        # obtain the image size
        dims = im.shape
        # create the zeros matrix with the same size
        result = np.zeros((dims[0]/2,dims[1]/2))
        
        # for each pixel in the image
        for i in range(dims[0]/2):
            for j in range(dims[1]/2):
                # get the maximum of 2x2 window around the pixel
                result[i,j] = np.max( im[range(i*2,i*2+2),range(j*2,j*2+2)] )
                
        return result
        
    # image smoothing using gaussian filter with given sigma and kernel size
    def smoothim(self,im,sigma,hsize):
        # constructu the filter
        gfilter = self.fgaussian((hsize,hsize),sigma)
        # create a new image with the same size as the given one
        result = np.zeros(im.shape)
        xd,yd = im.shape
        szx,szy = gfilter.shape
        
        # this is where the filter and the image are convolved
        for i in range(xd):
            for j in range(yd):
                summ = 0
                
                for fi in range(szy):
                    for fj in range(szx):
                        ay = i + (fi - hsize//2)
                        ax = j + (fj - hsize//2)
                        
                        if ax >= 0 and ay >= 0 and ax < xd and ay < yd:
                            summ = summ + im[ay,ax] * gfilter[fi,fj]
                            
                result[i,j] = summ
        
        # normalize the resulting image by the maximum pixel of the image itself
        if(np.max(np.max(result)) == 0):
            return result
        else:
            return result/(np.max(np.max(result))*1.0)    
        
    # a basic test method of smoothim function
    def smoothim_test(self):
        b = [[1,2,3,4],[5,6,7,8],[9,0,1,2],[3,4,5,6]]
        a = np.asarray(b)
        return self.smoothim(a,3,3)
        
        
    def coords2angles(self,sketch):
        '''Extract the angles between stroke points.
        For each stroke corresponding numpy array of angles is returned.
        '''
        # init angles array
        theta = []
        
        # for each stroke in the sketch given as input
        for stroke in sketch.strokes:
            # get the coordinates of the stroke
            scoords = stroke.listCoordinates()
            # get the number of points in the stroke
            numcoords,ncnt = scoords.shape
            tht = []
            # for each point, get the difference to the next point
            # and then find the angle of this point
            for ind in range(1,numcoords):
                diff = scoords[ind,:] - scoords[ind-1,:]
                tht.append( ( math.atan2(diff[1],diff[0]) % (math.pi*2) ) * (180/math.pi) )
            tht.append(float('nan'))
            
            # add the angles list to the resulting list
            theta.append(np.asarray(tht))
            
        return theta
