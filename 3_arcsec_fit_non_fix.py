
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import leastsq
from numpy import exp
from scipy import ndimage
from func import *


# In[2]:


def create_3_arcsec_minicube(wo_cube,wo_err,wo_header,cont_cube,brightest_pixel_x,brightest_pixel_y,sampling_rate,box_size=3):
    if sampling_rate == 0.4:
        mini_cube_data = wo_cube[:,brightest_pixel_y-box_size:brightest_pixel_y+box_size+1,brightest_pixel_x-box_size:brightest_pixel_x+box_size+1]
        mini_cube_err = wo_err[:,brightest_pixel_y-box_size:brightest_pixel_y+box_size+1,brightest_pixel_x-box_size:brightest_pixel_x+box_size+1]
    else:
        mini_cube_data = wo_cube[:,brightest_pixel_y-2*box_size:brightest_pixel_y+2*box_size+1,brightest_pixel_x-2*box_size:brightest_pixel_x+2*box_size+1]
        mini_cube_err = wo_err[:,brightest_pixel_y-2*box_size:brightest_pixel_y+2*box_size+1,brightest_pixel_x-2*box_size:brightest_pixel_x+2*box_size+1]
    wo_header['CRPIX1'] = wo_header['CRPIX1'] - (brightest_pixel_x-box_size)
    wo_header['CRPIX2'] = wo_header['CRPIX2'] - (brightest_pixel_y-box_size)
    return mini_cube_data, mini_cube_err,wo_header

def fit(int_spectrum,int_error,mini_wave,p_init,broad2=False,MC_loops=10,min_wave=4750,max_wave=5090):
    if broad2:
        full_gauss = full_gauss2
    else:
        full_gauss = full_gauss1
    (spectrum,error) = (int_spectrum,int_error)   
    popt_full_fit,pcov_full_fit = leastsq(full_gauss,x0=p_init,args=(mini_wave,spectrum,error),maxfev = 10000000)       
    fitted=(full_gauss(popt_full_fit,mini_wave,spectrum,error))*(error)+spectrum 
    residual = spectrum - fitted
    
    spec_parameters_MC = np.zeros((len(popt_full_fit),MC_loops))
    for l in range(MC_loops):
        iteration_data = np.random.normal(spectrum,error)   
        popt_spec_MC,pcov_spec_MC = leastsq(full_gauss,x0=popt_full_fit,args=(mini_wave,iteration_data,error),maxfev = 10000000)
        spec_parameters_MC[:,l]=popt_spec_MC
    spec_parameters_err = np.std(spec_parameters_MC,1)
    return popt_full_fit,spec_parameters_err,fitted,residual

def central_table(obj,output_par,output_par_err):
    column_names={'amp_Hb':0,'amp_OIII5007':1,'vel_OIII':2,'vel_sigma_OIII':3,'amp_Hb_br':4,'amp_OIII5007_br':5,'vel_OIII_br':6,
              'vel_sigma_OIII_br':7,'amp_Hb1':8,'amp_Fe5018_1':9,'vel_Hb1':10,'vel_sigma_Hb1':11,'amp_Hb2':12,
              'amp_Fe5018_2':13,'vel_Hb2':14,'vel_sigma_Hb2':15,'m':16,'c':17}
    columns=[]
    for key in column_names.keys():
        columns.append(fits.Column(name=key,format='E',array=[output_par[column_names[key]]]))
        columns.append(fits.Column(name=key+'_err',format='E',array=[output_par_err[column_names[key]]]))
    coldefs = fits.ColDefs(columns)
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto('%s_non_spectro_central_fit.fits'%(obj),overwrite=True)

    
    
    
def plot(mini_wave,int_spectrum,output_par,fitted,residual):
    (amp_Hb_fit,amp_OIII5007_fit,vel_OIII_fit,vel_sigma_OIII_fit,amp_Hb_br_fit,amp_OIII5007_br_fit,vel_OIII_br_fit,vel_sigma_OIII_br_fit,amp_Hb1_fit,amp_Fe5018_1_fit,vel_Hb1_fit,vel_sigma_Hb1_fit,amp_Hb2_fit,amp_Fe5018_2_fit,vel_Hb2_fit,vel_sigma_Hb2_fit,m_fit,c_fit) = output_par
    print output_par
    plt.plot(mini_wave,int_spectrum,'k-',label='data')
    plt.plot(mini_wave,fitted,'r-',label='fit')
    plt.plot(mini_wave,residual,label='residual')
    plt.plot(mini_wave,Hb_O3_gauss(mini_wave,amp_Hb_fit,amp_OIII5007_fit,vel_OIII_fit,vel_sigma_OIII_fit) + continuum(mini_wave,m_fit,c_fit),label='core',color = 'green')
    plt.plot(mini_wave,Hb_O3_gauss(mini_wave,amp_Hb_br_fit,amp_OIII5007_br_fit,vel_OIII_br_fit,vel_sigma_OIII_br_fit) + continuum(mini_wave,m_fit,c_fit),label='wing',color ='magenta')
    plt.plot(mini_wave,Hb_Fe_doublet_gauss(mini_wave,amp_Hb1_fit,amp_Fe5018_1_fit,vel_Hb1_fit,vel_sigma_Hb1_fit) + continuum(mini_wave,m_fit,c_fit),label='BLR1',color ='blue')
    plt.plot(mini_wave,Hb_Fe_doublet_gauss(mini_wave,amp_Hb2_fit,amp_Fe5018_2_fit,vel_Hb2_fit,vel_sigma_Hb2_fit) + continuum(mini_wave,m_fit,c_fit),label='BLR2',color = 'orange')
    plt.title('%s_integrated spectrum (Free parameter fit)'%(obj))
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(r"Flux Density ($\times 10^{-16}$ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$) ")
    plt.legend()
    plt.show()


# In[3]:


def spectroastrometric_script(obj,z,broad2,asymmetry,p_init,box_size=7,min_wave=4750,max_wave=5090,MC_loops=5,prefix_path_cube="/home/mainak/xdata/ftp.hidrive.strato.com/users/login-carsftp"):
    #try:
        #(orig_cube,orig_err,orig_wave,orig_header) = loadCube('%s/MUSE/%s/%s.binned.fits'%(prefix_path_cube,obj,obj))
    #except IOError:
        #(orig_cube,orig_err,orig_wave,orig_header) = loadCube('%s/MUSE/%s/%s.unbinned.fits'%(prefix_path_cube,obj,obj))  
    #(cont_cube,cont_err,cont_wave,cont_header) = loadCube('%s/MUSE/%s/%s.cont_model.fits'%(prefix_path_cube,obj,obj))
    #(QSO_cube,QSO_err,QSO_wave,QSO_header) = loadCube('%s/MUSE/%s/%s.QSO_full.fits'%(prefix_path_cube,obj,obj))
    #(wo_cube,wo_err,wo_wave,wo_header) = loadCube('%s.wo_absorption.fits'%(obj))
    
    #[brightest_pixel_x,brightest_pixel_y] = brightest_pixel(QSO_cube,wo_cube,wo_wave,z)
    #sampling_rate = sampling_size(cont_cube)
    #(mini_cube,mini_err,wo_header)=create_3_arcsec_minicube(wo_cube,wo_err,wo_header,cont_cube,brightest_pixel_x,brightest_pixel_y,sampling_rate,box_size=4)
    #k = 1+z
    #select = (wo_wave > min_wave*k) & (wo_wave < max_wave*k)  
    #mini_header = wo_header
    #mini_wave = wo_wave[select] 
    #mini_cube = mini_cube[select] 
    #mini_err = mini_err[select]
    #store_cube('%s.3_arcsec_minicube.fits'%(obj),mini_cube,mini_wave,mini_err,mini_header)
    print '%s'%(obj)
    (mini_cube,mini_err,mini_wave,mini_header) = loadCube('%s.3_arcsec_minicube.fits'%(obj))
    (int_spectrum,int_error) = int_spec(mini_cube,mini_err)
    (output_par,output_par_err,fitted,residual) = fit(int_spectrum,int_error,mini_wave,p_init,broad2,MC_loops=10,min_wave=4750,max_wave=5090)
    central_table(obj,output_par,output_par_err)
    plot(mini_wave,int_spectrum,output_par,fitted,residual)
    


# In[4]:


z = {"HE0021-1819":0.053197,"HE0040-1105":0.041692 #,"HE0108-4743":0.02392,"HE0114-0015":0.04560
    ,"HE0119-0118":0.054341,"HE0224-2834":0.059800,"HE0227-0913":0.016451,"HE0232-0900":0.043143,"HE0253-1641":0.031588
    ,"HE0345+0056":0.031,"HE0351+0240":0.036,"HE0412-0803":0.038160,"HE0429-0247":0.042009,"HE0433-1028":0.035550
    ,"HE0853+0102":0.052,"HE0934+0119":0.050338,"HE1011-0403":0.058314,"HE1017-0305":0.049986,"HE1029-1831":0.040261
    ,"HE1107-0813":0.058,"HE1108-2813":0.024013,"HE1126-0407":0.061960,"HE1330-1013":0.022145,"HE1353-1917":0.035021
    ,"HE1417-0909":0.044,"HE2211-3903":0.039714,"HE2222-0026":0.059114,"HE2233+0124":0.056482,"HE2302-0857":0.046860}

#z_remaining = {"HE2128-0221":0.05248,"HE1248-1356":0.01465}
#p_init_of_them = 'HE0108-4743':[1.139,1.5,7176.0,50.0,1.0,3.0,6976.0,200.0,1.0,1.0,7176,1000.0,1.0,1.0,7176,1000.0,-0.001,2.0]
       # ,'HE0114-0015':[0.1,1.5,13680,50.0,1.0,3.0,13480,100.0,1.0,1.0,13680,1000.0,0,0,13680,1000.0,-0.001,0.3]
       
objs = z.keys()

broad2= {'HE0021-1819':False,'HE0040-1105':False #,'HE0108-4743':True,'HE0114-0015':False
        ,'HE0119-0118':True,'HE0224-2834':False,'HE0227-0913':True,'HE0232-0900':False,'HE0253-1641':True
        ,'HE0345+0056':True,'HE0351+0240':True,'HE0412-0803':False,'HE0429-0247':True,'HE0433-1028':True
        ,'HE0853+0102':True,'HE0934+0119':True,'HE1011-0403':True,'HE1017-0305':False,'HE1029-1831':True
        ,'HE1107-0813':True,'HE1108-2813':False,'HE1126-0407':True,'HE1330-1013':True,'HE1353-1917':True
        ,'HE1417-0909':False,'HE2211-3903':False,'HE2222-0026':True,'HE2233+0124':True,'HE2302-0857':True}

asymmetry = {"HE0021-1819":True,"HE0040-1105":True #,"HE0108-4743":dunno,"HE0114-0015":dunno
    ,"HE0119-0118":False,"HE0224-2834":False,"HE0227-0913":False,"HE0232-0900":False,"HE0253-1641":False
    ,"HE0345+0056":False,"HE0351+0240":False,"HE0412-0803":False,"HE0429-0247":False,"HE0433-1028":False
    ,"HE0853+0102":False,"HE0934+0119":True,"HE1011-0403":False,"HE1017-0305":False,"HE1029-1831":False
    ,"HE1107-0813":False,"HE1108-2813":True,"HE1126-0407":False,"HE1330-1013":True,"HE1353-1917":True
    ,"HE1417-0909":False,"HE2211-3903":False,"HE2222-0026":False,"HE2233+0124":False,"HE2302-0857":False}


p_init= {'HE0021-1819':[1,12.5,15959,50.0,3,3,15759,200.0,2,2,15959,1000.0,0,0,15959,1000.0,-0.001,0.1]
         ,'HE0040-1105':[1.139,1.5,12507.0,50.0,1.0,3.0,12307.0,100.0,1.0,1.0,12507,1000.0,0,0,12507,1000.0,-0.001,2.0]
         ,'HE0119-0118':[22,125,16302,50.0,3,3,16002,100.0,10,1,16302,1000.0,10,1,16302,1500.0,-0.001,0.1]
         ,'HE0224-2834':[7.06899403e+00,7.13458601e+01,1.79847558e+04,1.18000676e+02,1.45515224e+00,1.90068487e+01,1.79693125e+04,2.72340813e+02,8.44672986e+00,9.16885096e-01,1.80245745e+04,2.16641488e+03,0,0,0,0,-2.79613744e+00,2.29434665e+01]  
         ,'HE0227-0913':[15,70,4935,50.0,10,20,4835,100.0,100,20,4935,1000.0,30,10,4935,500.0,-0.001,0.1]
         ,'HE0232-0900':[2,40,12942,50.0,5,5,12742,100.0,5,1,12942,1000.0,0,0,12942,1000.0,-0.001,0.1] 
         ,'HE0253-1641':[21,200,9476,90.0,11,70,9176,200.0,26,7,9476,1000.0,15,3.5,9476,1000.0,-0.001,0.5]
         ,'HE0345+0056':[14,180,9300,155,45,123,9100,400,90,18,9300,1500,270,26,9300,400,-7.0,7.16315181e+00]
         ,'HE0351+0240':[2.19264502e+00,3.70402407e+01,1.06360169e+04,7.72338448e+01,1.10298241e+00,1.50249383e+00,1.06183228e+04,2.77549457e+02,1.52283432e+00,7.54204519e-02,1.15806639e+04,3.17978311e+02,3.47577206e+00,3.51692202e-01,1.06529589e+04,1.34441236e+03,-2.90461186e-01,3.33119886e+00]
         ,'HE0412-0803':[1.139,1.5,11448,50.0,0.1,0.3,11248,100.0,0.1,0.1,11448,1000.0,0,0,11448,1000.0,-0.001,0.5]
         ,'HE0429-0247':[0.3,1,12602,40.0,7,0.2,12502,500.0,0.1,0.1,12602,460.0,0.1,0.1,12602,1000.0,-0.001,0.1]
         ,'HE0433-1028':[10,100,10665.0,20.0,100,1.0,10465.0,200.0,25,1.0,10665,1000.0,55,0.1,10665,2500.0,-0.1,1.0]
         ,'HE0853+0102':[0.1,0.9,15600,50.0,0.1,0.1,15400,100.0,0.1,0.1,15600,1000.0,0.1,0.1,15600,1000.0,-0.001,0.1]
         ,'HE0934+0119':[11,47,15101,50.0,7,24,14901,100.0,28,3.0,15101,1000.0,3,1,15101,1000.0,0.001,0.7]
         ,'HE1011-0403':[6,40,17494,60.0,6,6,17250,200.0,20,4,17494,1000.0,5,3,17494,500.0,0.001,0.7]
         ,'HE1017-0305':[4,35,14995,50,2,3,14895,100,16,3,14995,2000,0,0,14995,1000,-0.3,2.0]
         ,'HE1029-1831':[30,9,12078,30.0,5,45,11978,100.0,5,0.1,12078,1000.0,12,2,12078,1000.0,-0.001,0.2]
         ,'HE1107-0813':[5,10,17400,50,20,3,17200,400,8.66709804e-01,3.59444240e-01,1.75395296e+04,9.89119070e+02,8.77384324e-01,7.64366776e-02,1.68588382e+04,2.00171135e+03,-8.35532281e-01,7.28333927e+00]
         ,'HE1108-2813':[28,35,7200.0,50.0,11,28,7000.0,100.0,16,4.2,7200,1000.0,0,0,7200,1000.0,-8.3,60]
         ,'HE1126-0407':[1.00409949e+00,1.34246331e+01,1.80273208e+04,1.24499810e+02,2.46452048e+00,2.86320853e+00,1.77600119e+04,3.51254916e+02,9.96139933e+00,1.30416561e+00,1.80789175e+04,7.18418498e+02,7.31717375e+00,1.77825637e+00,1.79704287e+04,1.77768741e+03,-9.59655619e-01,1.47131358e+01]                                           
         ,'HE1330-1013':[3,12,6643,88,2,2,6435,215,12,2,6643,1500,3,1,6643,616,-1.74500992e-02,2.87685339e-01]
         ,'HE1353-1917':[6,50,10506.0,100.0,3,10,9906.0,350.0,5,1,10506.0,1000.0,5,1,10506.0,1000.0,-0.001,0.002]
         ,'HE1417-0909':[7,100,13200,50.0,1.6,20,13000,100.0,10,1.2,13200,1000.0,0,0,13200,1000.0,-0.001,0.1]
         ,'HE2211-3903':[10,42,11914,50.0,2,5,11714,100.0,5,1,11914,1000.0,0,0,11914,100.0,-0.001,0.2]
         ,'HE2222-0026':[2,10,17734.0,50.0,2,0.1,17634.0,300.0,3,1.5,17634,2000.0,3,1,17834,500.0,-0.001,0.002]
         ,'HE2233+0124':[3,10,16944.0,170.0,2,2,16744.0,500.0,2,2,16944,1000.0,4,2,16944,4000.0,-0.001,0.01]
         ,'HE2302-0857':[25,220,14058,50,25,50,14258,500,31,15,14058,1500,20,10,14058,1700,-0.9,6.0]}
              
for obj in objs:
    spectroastrometric_script(obj,z[obj],broad2[obj],asymmetry[obj],p_init[obj])              

