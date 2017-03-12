#!/usr/bin/python
#copyright LI YU

import urllib  #module to fetching URLs
import urllib2
import libxml2
#tutorial: https://mohsinpage.wordpress.com/2011/02/03/xpath-libxml2-in-python/
import inspect
import sys
import zlib
import base64
import struct
import numpy 
import cv2
import csv
import os
from math import*
from os import path


BaseUri = 'http://cbk0.google.com/cbk';

# panoid is the value from panorama metadata
# c.f. : http://cbk0.google.com/cbk?output=xml&dm=1&pm=1&ll=40.720032,%20-73.988354
# OR: supply lat/lon/radius to find the nearest pano to lat/lon within radius

def GetPanoramaMetadata(panoid = None, lat = None, lon = None, radius = 2000):
        url =  '%s?'
        url += 'output=xml'                     # metadata output
        #url += '&v=4'                           # version
        url += '&dm=1'                          # include depth map
        url += '&pm=1'                          # include pano map
        if panoid == None:
                url += '&ll=%s,%s'              # lat/lon to search at
                url += '&radius=%s'             # search radius
                url = url % (BaseUri, lat, lon, radius)
        else:
                url += '&panoid=%s'             # panoid to retrieve
                url = url % (BaseUri, panoid)
        print url
        
        findpanoxml = GetUrlContents(url)		# meta data of url as '.xml'
        if not findpanoxml.find('data_properties'):
                return None
        return PanoramaMetadata(libxml2.parseDoc(findpanoxml)) # a class

# panoid is the value from the panorama metadata
# zoom range is 0->NumZoomLevels inclusively
# x/y range is 0->?

def GetPanoramaTile(panoid, zoom, x, y):
        url =  '%s?'
        url += 'output=tile'                    # tile output
        url += '&panoid=%s'                     # panoid to retrieve
        url += '&zoom=%s'                       # zoom level of tile
        url += '&x=%i'                          # x position of tile
        url += '&y=%i'                          # y position of tile
        #url += '&fover=2'                       # ???
        #url += '&onerr=3'                       # ???
        url += '&renderer=spherical'            # standard speherical projection
        url += '&v=4'                           # version
        url = url % (BaseUri, panoid, zoom, x, y)
        
        #print url
        return GetUrlContents(url)

def GetUrlContents(url):
        f = urllib2.urlopen(url)
        data = f.read()
        f.close()
        return data

class PanoramaMetadata:
        
        def __init__(self, panodoc):
                self.PanoDoc = panodoc
                panoDocCtx = self.PanoDoc.xpathNewContext()

                self.PanoId = panoDocCtx.xpathEval("/panorama/data_properties/@pano_id")[0].content
                self.ImageWidth = panoDocCtx.xpathEval("/panorama/data_properties/@image_width")[0].content
                self.ImageHeight = panoDocCtx.xpathEval("/panorama/data_properties/@image_height")[0].content
                self.TileWidth = panoDocCtx.xpathEval("/panorama/data_properties/@tile_width")[0].content
                self.TileHeight = panoDocCtx.xpathEval("/panorama/data_properties/@tile_height")[0].content
                self.NumZoomLevels = panoDocCtx.xpathEval("/panorama/data_properties/@num_zoom_levels")[0].content
                self.Lat = panoDocCtx.xpathEval("/panorama/data_properties/@lat")[0].content
                self.Lon = panoDocCtx.xpathEval("/panorama/data_properties/@lng")[0].content
                self.Alt = panoDocCtx.xpathEval("/panorama/data_properties/@elevation_wgs84_m")[0].content
                self.OriginalLat = panoDocCtx.xpathEval("/panorama/data_properties/@original_lat")[0].content
                self.OriginalLon = panoDocCtx.xpathEval("/panorama/data_properties/@original_lng")[0].content
                self.Copyright = panoDocCtx.xpathEval("/panorama/data_properties/copyright/text()")[0].content
                
                #self.Text = panoDocCtx.xpathEval("/panorama/data_properties/text/text()")[0].content
                #self.Region = panoDocCtx.xpathEval("/panorama/data_properties/region/text()")[0].content
                #self.Country = panoDocCtx.xpathEval("/panorama/data_properties/country/text()")[0].content

                self.ProjectionType = panoDocCtx.xpathEval("/panorama/projection_properties/@projection_type")[0].content
                self.ProjectionPanoYawDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@pano_yaw_deg")[0].content
                self.ProjectionTiltYawDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@tilt_yaw_deg")[0].content
                self.ProjectionTiltPitchDeg = panoDocCtx.xpathEval("/panorama/projection_properties/@tilt_pitch_deg")[0].content
                
                self.AnnotationLinks = []
                for cur in panoDocCtx.xpathEval("/panorama/annotation_properties/link"):
                        self.AnnotationLinks.append({ 'YawDeg': cur.xpathEval("@yaw_deg")[0].content,
                                            'PanoId': cur.xpathEval("@pano_id")[0].content,
                                            'RoadARGB': cur.xpathEval("@road_argb")[0].content,
                                           #'Text': cur.xpathEval("link_text/text()")[0].content,
                       })
                
                #Decode panomap

                tmp = panoDocCtx.xpathEval("/panorama/model/pano_map/text()")
                if len(tmp) > 0:
                        tmp = tmp[0].content
                        tmp = zlib.decompress(base64.urlsafe_b64decode(tmp + self.MakePadding(tmp)))                    
                        self.DecodePanoMap(tmp)
                
                #Decode depthmap
                tmp = panoDocCtx.xpathEval("/panorama/model/depth_map/text()")
                if len(tmp) > 0:
                        tmp = tmp[0].content
                        tmp = zlib.decompress(base64.urlsafe_b64decode(tmp + self.MakePadding(tmp)))
                        self.DecodeDepthMap(tmp)


        def MakePadding(self, s):
                return (4 - (len(s) % 4)) * '='


        def DecodePanoMap(self, raw):
                pos = 0
                
                (headerSize, numPanos, panoWidth, panoHeight, panoIndicesOffset) = struct.unpack('<BHHHB', raw[0:8])
                if headerSize != 8 or panoIndicesOffset != 8:
                        print "Invalid panomap data"
                        return
                pos += headerSize
                
                self.PanoMapIndices = [ord(x) for x in raw[panoIndicesOffset:panoIndicesOffset + (panoWidth * panoHeight)]]
                pos += len(self.PanoMapIndices)
                
                self.PanoMapPanos = []
                for i in xrange(0, numPanos - 1):
                        self.PanoMapPanos.append({ 'panoid': raw[pos: pos+ 22]})
                        pos += 22
                        
                for i in xrange(0, numPanos - 1):
                        (x, y) = struct.unpack('<ff', raw[pos:pos+8])
                        self.PanoMapPanos[i]['x'] = x
                        self.PanoMapPanos[i]['y'] = y
                        pos+=8

        #decode the depth map (// cf. paul wagener)

        def DecodeDepthMap(self, raw):
               
               #decode base 64 
                pos = 0

                (headerSize, numPlanes, panoWidth, panoHeight, planeIndicesOffset) = struct.unpack('<BHHHB', raw[0:8])
                if headerSize != 8 or planeIndicesOffset != 8:
                        print "Invalid depthmap data"
                        return
                pos += headerSize
                self.smallHeight = panoHeight
                self.smallWidth = panoWidth

                self.DepthMapIndices = [ord(x) for x in raw[planeIndicesOffset:planeIndicesOffset + (panoWidth * panoHeight)]]
                
                pos += len(self.DepthMapIndices)
                
                self.DepthMapPlanes = []
                self.depth = []
                self.nx =[]
                self.ny =[]
                self.nz =[]

                for i in xrange(0, numPlanes - 1):
                        (nx, ny, nz, d) = struct.unpack('<ffff', raw[pos:pos+16])

                        self.DepthMapPlanes.append( [d, nx, ny, nz]) 

                        self.depth.append(d)
                        self.nx.append(nx)
                        self.ny.append(ny)
                        self.nz.append(nz)

                        #self.DepthMapPlanes.append({ 'd': d, 'nx': nx, 'ny': ny, 'nz': nz })
                        # nx/ny/nz = unit normal, d = distance from origin
                        pos += 16


        def __str__(self):
                tmp = ''
                for x in inspect.getmembers(self):
                        if x[0].startswith("__") or inspect.ismethod(x[1]):
                                continue
                        
                        tmp += "%s: %s\n" % x
                return tmp

        def computeDepthMap(self):
            from numpy import sin as sin 
            from numpy import cos as cos
            from math import pi  as pi
            from math import floor as floor
    

            width = int(self.ImageWidth) #360deg
            height = int(self.ImageHeight)
            self.depthMap = numpy.zeros(shape=(height,width))
            phi = (numpy.linspace(0,width-1,width)+0.5)*2*pi/width-pi
            theta = pi/2-(numpy.linspace(0,height-1,height)+0.5)*pi/height # 90 degres en haut (ie i = 0)
        
            cosPhi = cos(phi)
            sinPhi = sin(phi)
            cosTheta = cos(theta)
            sinTheta = sin(theta)

            smallWidth = self.smallWidth
            smallHeight = self.smallHeight
            cosPhi = numpy.mat(cosPhi).transpose()
            sinPhi = numpy.mat(sinPhi).transpose()
            cosTheta = numpy.mat(cosTheta).transpose()
            #print cosPhi.shape
            #print cosTheta.shape
            #print cosTheta.transpose().shape
            x1 = cosTheta.dot(cosPhi.transpose())   
            y1 = cosTheta.dot(sinPhi.transpose())   
            i2 = numpy.linspace(0,height-1,height)
            iSmallVec = (i2/height*smallHeight)
            j2 = numpy.linspace(0,width -1,width )
            jSmallVec = (j2/width *smallWidth )
    
            #print x1.shape  
            for i in range(0,height):
                iSmall = int(iSmallVec[i])
                for j in range(0,width):
                    jSmall = int(jSmallVec[j])           
                    index = self.DepthMapIndices[jSmall+iSmall*smallWidth]
                    if (index >0 and index < len(self.nx)):
                        nx = self.nx[index]
                        ny = self.ny[index]
                        nz = self.nz[index]
                        d  = self.depth[index]
                        x = x1[i,j] # cosPhi[j]*cosTheta[i]
                        y = y1[i,j] # sinPhi[j]*cosTheta[i]
                        z = sinTheta[i]
            
                        self.depthMap[i,j] = numpy.fabs(d/(x*ny+y*nx+z*nz))    #d/ cos(angle)
                        #print self.depthMap[i,j]
                    else:
                        self.depthMap[i,j] = 255
#####################################################################################

pano = GetPanoramaMetadata(lat= 48.858175, lon= 2.274230,radius=16)

#manhattan: 40.746642, -73.980394  
#la muette: 48.858175, 2.274230
#monparnass2: 48.843423, 2.324268

#numpy.savetxt("depthIndices.txt", pano.DepthMapIndices, fmt="%s")
#numpy.savetxt("depth.txt", pano.depth, fmt="%s")
#numpy.savetxt("nx.txt", pano.nx, fmt="%s")
#numpy.savetxt("ny.txt", pano.ny, fmt="%s")
#numpy.savetxt("nz.txt", pano.nz, fmt="%s")



#print pano.PanoId
#print "The real Lat/Lon of this panorama is: "
#print pano.Lat
#print pano.Lon


# Download tiles to generate panorama by OpenCV
# Generate panorama from tiles
# ATTENTION: USING PYTHON TO CREATE THE PANORAMA IS TOO SLOW
# An efficient C++ program is proposed to generated from tiles.
print pano.PanoId
           
print "Construct Panorama RGB now ...."
     

width = int(pano.ImageWidth)
tileWidth = int(pano.TileWidth)
numTileWidth = width / tileWidth
height = int(pano.ImageHeight)
tileHeight = int(pano.TileHeight)
numTileHeight = height /tileHeight


from PIL import Image as im

panoRGB = numpy.zeros(shape=(height,width,3),dtype=numpy.uint8)

for i in range(0,numTileWidth):
    for j in range(0,numTileHeight):
        currentTile = GetPanoramaTile(pano.PanoId, 5, i, j)
        with open('tile.jpg','w') as f:
            f.write(currentTile)
        img = im.open('tile.jpg')
        imgArr = numpy.array(img)
        panoRGB[j*tileHeight:(j+1)*tileHeight,i*tileWidth:(i+1)*tileWidth,:] = imgArr

RGBOut = im.fromarray(panoRGB)
        
path ='./'
savePanoName = 'pano.png'
savePanoName= os.path.join(path,savePanoName)
RGBOut.save(savePanoName)    
        
print "Save the depth map saved ..."

print "----------------"
print "Construct the depth map now ..."

pano.computeDepthMap()

print "Save the depth map ..."
#numpy.savetxt("depthMap.txt", pano.depthMap, fmt="%s")

depthFinal = numpy.zeros(shape=pano.depthMap.shape, dtype=numpy.uint8) #unit 8 #set to float to high precision
heightD = depthFinal.shape[0]
widthD = depthFinal.shape[1]

     

path2 = './'
saveDepthName ='depth.png'

saveDepthName = os.path.join( path2, saveDepthName)
cv2.imwrite(saveDepthName,pano.depthMap)
        





