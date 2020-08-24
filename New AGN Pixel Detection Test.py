#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import leastsq
from scipy import ndimage
from func import *


# In[2]:


def agn_brightest_pixel(QSO_cube,wo_cube,wo_wave,z,weirdness=False):
    QSO_slice = QSO_cube[0,:,:]
    if weirdness:
        k = 1 + z
        [guess_y,guess_x] = ndimage.measurements.maximum_position(QSO_slice)
        
        #select = (wo_wave >4750*k) & (wo_wave<5090*k) 
        #wo_cube = wo_cube[select]
        print (np.shape(wo_cube))
        test_cube = wo_cube[:,guess_y-5:guess_y+5,guess_x-5:guess_x+5]
        test_slice = test_cube[1,:,:]
        [z0,y0,x0] = ndimage.measurements.maximum_position(test_cube)
        [yn,xn] = (y0+guess_y-5,x0+guess_x-5)
        #(xn,yn) = brightest_pixel(QSO_cube,wo_cube,wo_wave,z)
    else:
        (xn,yn) = alternative_brightest_pixel(QSO_cube)
    return xn,yn

def central_pix_tab(obj,x0,y0,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    data = [x0,y0]
    column_names={'x0':0,'y0':1}
    columns=[]
    for key in column_names.keys():
        columns.append(fits.Column(name=key,format='E',array=[data[column_names[key]]]))
    coldefs = fits.ColDefs(columns)
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto('%s/%s/%s_AGNpix.fits'%(destination_path_cube,obj,obj),overwrite=True)

    


# In[3]:


def algorithm_script(obj,z,weirdness,prefix_path_cube="/home/mainak/ftp.hidrive.strato.com/users/login-carsftp/IFU_data",destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    print ('%s'%(obj))
    try:
        (orig_cube,orig_err,orig_wave,orig_header) = loadCube('%s/MUSE/%s/%s.binned.fits'%(prefix_path_cube,obj,obj))
    except IOError:
        (orig_cube,orig_err,orig_wave,orig_header) = loadCube('%s/MUSE/%s/%s.unbinned.fits'%(prefix_path_cube,obj,obj))   
    (cont_cube,cont_err,cont_wave,cont_header) = loadCube('%s/MUSE/%s/fitting/full/%s.cont_model.fits'%(prefix_path_cube,obj,obj))
    (QSO_cube,QSO_err,QSO_wave,QSO_header) = loadCube('%s/MUSE/%s/%s.QSO_full.fits'%(prefix_path_cube,obj,obj))
    (wo_cube,wo_err,wo_wave,wo_header) = loadCube('%s/%s/%s.wo_absorption.fits'%(destination_path_cube,obj,obj)) 
    
    (x0,y0) = agn_brightest_pixel(QSO_cube,wo_cube,wo_wave,z,weirdness)
    (x,y) = alternative_brightest_pixel(QSO_cube)
    print (x0,y0)
    print (x,y)
    central_pix_tab(obj,x0,y0)


# In[ ]:


z = {"HE0021-1810":0.05352,"HE0021-1819":0.053197,"HE0040-1105":0.041692,"HE0108-4743":0.02392,"HE0114-0015":0.04560
    ,"HE0119-0118":0.054341,"HE0212-0059":0.026385,"HE0224-2834":0.059800,"HE0227-0913":0.016451,"HE0232-0900":0.043143
    ,"HE0253-1641":0.031588,"HE0345+0056":0.031,"HE0351+0240":0.036,"HE0412-0803":0.038160,"HE0429-0247":0.042009
    ,"HE0433-1028":0.035550,"HE0853+0102":0.052,"HE0934+0119":0.050338,"HE1011-0403":0.058314,"HE1017-0305":0.049986
    ,"HE1029-1831":0.040261,"HE1107-0813":0.058,"HE1108-2813":0.024013,"HE1126-0407":0.061960,"HE1237-0504":0.009
    ,"HE1248-1356":0.01465,"HE1330-1013":0.022145,"HE1353-1917":0.035021,"HE1417-0909":0.044,"HE2128-0221":0.05248
    ,"HE2211-3903":0.039714,"HE2222-0026":0.059114,"HE2233+0124":0.056482,"HE2302-0857":0.046860}

objs = z.keys()


weirdness = {"HE0021-1810":True,"HE0021-1819":False,"HE0040-1105":True,"HE0108-4743":False,"HE0114-0015":False
    ,"HE0119-0118":False,"HE0212-0059":True,"HE0224-2834":False,"HE0227-0913":False,"HE0232-0900":False
    ,"HE0253-1641":False,"HE0345+0056":False,"HE0351+0240":False,"HE0412-0803":False,"HE0429-0247":False
    ,"HE0433-1028":False,"HE0853+0102":False,"HE0934+0119":False,"HE1011-0403":False,"HE1017-0305":False
    ,"HE1029-1831":False,"HE1107-0813":False,"HE1108-2813":False,"HE1126-0407":False,"HE1237-0504":False
    ,"HE1248-1356":False,"HE1330-1013":False,"HE1353-1917":False,"HE1417-0909":False,"HE2128-0221":True
    ,"HE2211-3903":False,"HE2222-0026":False,"HE2233+0124":False,"HE2302-0857":False}


for obj in objs:
     algorithm_script(obj,weirdness[obj],z[obj]) 


# In[1]:


z = {"HE2302-0857":0.04686}

difficulty = {"HE2302-0857":False}

objs = z.keys()

for obj in objs:
     algorithm_script(obj,difficulty[obj],z[obj])


# In[ ]:


z = {"HE0021-1810":0.05352,"HE0021-1819":0.053197,"HE0040-1105":0.041692,"HE0108-4743":0.02392,"HE0114-0015":0.04560
    ,"HE0119-0118":0.054341,"HE0212-0059":0.026385,"HE0224-2834":0.059800,"HE0227-0913":0.016451,"HE0232-0900":0.043143
    ,"HE0253-1641":0.031588,"HE0345+0056":0.031,"HE0351+0240":0.036,"HE0412-0803":0.038160,"HE0429-0247":0.042009
    ,"HE0433-1028":0.035550,"HE0853+0102":0.052,"HE0934+0119":0.050338,"HE1011-0403":0.058314,"HE1017-0305":0.049986
    ,"HE1029-1831":0.040261,"HE1107-0813":0.058,"HE1108-2813":0.024013,"HE1126-0407":0.061960,"HE1237-0504":0.009
    ,"HE1248-1356":0.01465,"HE1330-1013":0.022145,"HE1353-1917":0.035021,"HE1417-0909":0.044,"HE2128-0221":0.05248
    ,"HE2211-3903":0.039714,"HE2222-0026":0.059114,"HE2233+0124":0.056482,"HE2302-0857":0.046860}

objs = z.keys()


weirdness = {"HE0021-1810":True,"HE0021-1819":False,"HE0040-1105":True,"HE0108-4743":False,"HE0114-0015":False
    ,"HE0119-0118":False,"HE0212-0059":True,"HE0224-2834":False,"HE0227-0913":False,"HE0232-0900":False
    ,"HE0253-1641":False,"HE0345+0056":False,"HE0351+0240":False,"HE0412-0803":False,"HE0429-0247":False
    ,"HE0433-1028":False,"HE0853+0102":False,"HE0934+0119":False,"HE1011-0403":False,"HE1017-0305":False
    ,"HE1029-1831":False,"HE1107-0813":False,"HE1108-2813":False,"HE1126-0407":False,"HE1237-0504":False
    ,"HE1248-1356":False,"HE1330-1013":False,"HE1353-1917":False,"HE1417-0909":False,"HE2128-0221":True
    ,"HE2211-3903":False,"HE2222-0026":False,"HE2233+0124":False,"HE2302-0857":False}


for obj in objs:
     algorithm_script(obj,weirdness[obj],z[obj]) 

