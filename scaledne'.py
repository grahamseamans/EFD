# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 18:17:14 2021

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy import distance
import math
import csv 

#Geopy uses geodetic distance between two points in lat/lon to find meter distance

def dist_from_ll(p1,p2):
    return distance.distance(p1, p2).meters
#To make everything in 0.1 meter grid could multiply all input by 10, or remove int and round
def xy_from_ll(p1,p2):
    p2xd = (p2[0],p1[1])
    p2yd = (p1[0],p2[1])
    x = distance.distance(p1, p2xd).meters
    y = distance.distance(p1, p2yd).meters
    return (round(x),round(y))


gpsloc = pd.read_csv(r'C:\Users\user\Downloads\GPS_Loc.txt')
gpsloc.index = gpsloc['StationName']
gpsloc = gpsloc[['Lat', 'Long', 'Northing', 'Easting', 'Zone']]

#LEMS 17 is the bottom right corner to use as (0,0) for grid
lemsdf = gpsloc[gpsloc.index.str.match('LEMS')]

x_pixel_ord=np.arange(0,1024,1)
y_pixel_ord=np.arange(0,768,1)

pixelx = pd.read_csv(r'C:\Users\user\Desktop\pixelx.csv',names=x_pixel_ord)
pixely = pd.read_csv(r'C:\Users\user\Desktop\pixely.csv',names=x_pixel_ord)

scale = 10 # pixels per meter
dtsrest = 8 # dts measurements per meter (8 is max of device)
orientation = 295

#http://www.pythonforengineers.in/2019/08/dda-line-drawing-algorithms-line.html
def drawDDA(point1,point2,num_measurements):
    """Maybe make length a third argument based on the number of DTS measurements in the line segment"""
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    points = []
    x,y = x1,y1
    #length = np.abs(x2-x1) if np.abs(x2-x1) > np.abs(y2-y1) else np.abs(y2-y1)
    #old dynamically generated length
    length=num_measurements
    dx = (x2-x1)/float(length)
    dy = (y2-y1)/float(length)
    points.append((round(x),round(y)))
    for i in range(length):
        x += dx
        y += dy
        points.append((round(x),round(y)))
    return points
"""        
4851.78 m -> From Total station position of all buckets and including 60 m for up and down tower. 
Distance from Temperature baths to first bucket not included
Distance of coils in temperature baths not included
"""
#distance between points 3 and 4 is very small because of the tower
dts_cum_tfirst = [dtsrest* i  for i in [0,3.85,46.89,47.58,89.29,174.31,581.59,1096.51,1719.86,2343.20,2518.87,2989.44,3133.85,3230.36,3841.09,3947.10,4545.73,4588.90,4751.29,4791.78]]

dts_lengths = []
for i in range(len(dts_cum_tfirst)-1):
    dts_lengths.append(round(dts_cum_tfirst[i+1]-dts_cum_tfirst[i]))
#60m of cable going up and down tower
dts_lengths[2] += 60*dtsrest

latlon = gpsloc[['Lat','Long']]
northeast = scale*gpsloc[['Northing','Easting']]

#northeast is a df with the north and east distances from LEMS 17 of each sensor
northeast = round(northeast-northeast.loc['LEMS No 17'])
stationxy = northeast.iloc[:-20,:]

dtsll = northeast[-20:].values
    
#dtspoints is a list of lists of tuples of (x,y) distances from LEMS 17 for each dts sensor location using DDA at 1m resolution
dtspoints = []
for i in range(len(dtsll)-1):
    d1a = (dtsll[i][0],dtsll[i][1])
    d1b = (dtsll[i+1][0],dtsll[i+1][1])
    dtspoints.append(drawDDA(d1a,d1b,dts_lengths[i])[1:])
    print(d1a,d1b)

#dtsxy is a list of tuples of (x,y) distances from LEMS 17 for each dts sensor at 1m resolution
dtsxy = []
count = 0
for i in dtspoints:
    if count < 2: del i[0:dtsrest] #removes 17 meters of positions due to mismatch between sensors and cumulative distances
    for j in i:
        dtsxy.append(j)
    count += 1
        
#pxy is a list of tuples which are (x,y) distance from bottom center of image, where x and y are the local directions not N/S
pxy = []
for i in pixelx.index:
    for j in pixelx.columns:
        pxy.append((scale*pixelx.iloc[i,j],scale*pixely.iloc[i,j]))

"""
redundant following change to Travis' code to not subtract starting row

imdispl = scale*7.4*np.tan(math.radians(vangl-12.3))
imdispx = imdispl*np.sin(math.radians(-1*a))
imdispy = imdispl*np.cos(math.radians(-1*a))

"""
angle = 360-orientation
#pixelpos is a list of tuples which are (x,y) in meters from LEMS 17, the zeroth row is the bottom of the image, and the zeroth column is furthest to the left of image
#a,b,c,d are the ratio between the hypotenuese of x (a,b) and y (c,d) to the true x,y 
a = np.cos(math.radians(angle))
b = np.sin(math.radians(-1*angle))
c = np.sin(math.radians(angle))
d = np.cos(math.radians(-1*angle))



pixelpos = []
for i in pxy:
    pixelpos.append((int(round(i[0]*a+i[1]*b+northeast.loc['Thermal Camera'][1])),
                     int(round(i[0]*c+i[1]*d+northeast.loc['Thermal Camera'][0]))))



# opening the csv file in 'w+' mode 
file1 = open('dtsxy10.csv', 'w+', newline ='') 
file2 = open('tirxy10.csv', 'w+', newline ='') 


# writing the data into the file 
with file1:	 
	write = csv.writer(file1) 
	write.writerows(dtsxy) 

with file2:	 
	write = csv.writer(file2) 
	write.writerows(pixelpos)
    
stationxy.to_csv('stationxy10.csv')
