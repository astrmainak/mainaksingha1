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


######################## Scale Factor Error Flux Maps #####################
def err_ratio(data,err):
    data[2:-2,2:-2]=0
    err[2:-2,2:-2]=0
    edge_fluxmap = data[data!=0]
    edge_errmap = err[err!=0]
    err_fluxmap = np.std(edge_fluxmap)
    err_errmap = np.mean(edge_errmap)
    fact = err_fluxmap/err_errmap
    return fact

def flux_dat(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    hdu = fits.open('%s/Flux Maps/%s/9_arcsec_subcube_par_%s.fits'%(destination_path_cube,obj,obj))
    Hb1 = hdu[5].data
    Hb2 = hdu[6].data
    OIII_br = hdu[3].data
    OIII_nr = hdu[2].data
    hdu.close()
    return Hb1,Hb2,OIII_br,OIII_nr

def flux_err(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    hdu = fits.open('%s/Flux Maps/%s/9_arcsec_subcube_par_err_%s.fits'%(destination_path_cube,obj,obj))
    Hb1 = hdu[5].data
    Hb2 = hdu[6].data
    OIII_br = hdu[3].data
    OIII_nr = hdu[2].data
    hdu.close()
    return Hb1,Hb2,OIII_br,OIII_nr


# In[3]:


def flux_data_err(obj,emp_Hb1,emp_Hb2,emp_wing,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    hdu = fits.open('%s/Flux Maps/%s/9_arcsec_subcube_par_%s.fits'%(destination_path_cube,obj,obj))
    (OIII_nr,OIII_br,Hb1_blr_br,Hb2_blr_br) = (hdu[2].data,hdu[3].data,hdu[5].data,hdu[6].data)
    hdu.close()
    
    hdu = fits.open('%s/Flux Maps/%s/9_arcsec_subcube_par_err_%s.fits'%(destination_path_cube,obj,obj))
    (OIII_nr_err,OIII_br_err,Hb1_blr_br_err,Hb2_blr_br_err) = (hdu[2].data,hdu[3].data,hdu[5].data,hdu[6].data)
    hdu.close()
       
    (amp_OIII_nr,amp_OIII_br,amp_Hb1_blr_br,amp_Hb2_blr_br) = (np.max(OIII_nr),np.max(OIII_br),np.max(Hb1_blr_br),np.max(Hb2_blr_br))
    if amp_Hb1_blr_br > amp_Hb2_blr_br:
        (Hb_blr_br,amp_Hb_blr_br,Hb_blr_err,emp_Hb_blr) = (Hb1_blr_br,amp_Hb1_blr_br,Hb1_blr_br_err,emp_Hb1)
    else:
        (Hb_blr_br,amp_Hb_blr_br,Hb_blr_err,emp_Hb_blr) = (Hb2_blr_br,amp_Hb2_blr_br,Hb2_blr_br_err,emp_Hb2)
    (blr_err_final,wing_err_final,core_err_final) = (emp_Hb_blr*Hb_blr_err,emp_wing*OIII_br_err,OIII_nr_err)
    return Hb_blr_br,OIII_br,OIII_nr,amp_Hb_blr_br,amp_OIII_br,amp_OIII_nr,blr_err_final,wing_err_final,core_err_final
    
def moffat_fit(data,error,box_size,amp,x0,y0,MC_loops,gamma_fixed=None,alpha_fixed=None):
    y, x = np.mgrid[:box_size, :box_size] 
    if gamma_fixed is None and alpha_fixed is None:
        p_init = models.Moffat2D(amplitude=amp,x_0=x0,y_0=y0,gamma=1,alpha=1)
    else:
        p_init = models.Moffat2D(amp,x0,y0,gamma_fixed,alpha_fixed,fixed={'gamma':True,'alpha':True})
    f = fitting.LevMarLSQFitter()
    p = f(p_init, x, y, data)
    model = p(x,y)
    residual = data - model
    res = (residual/error)
    [amp_out,x0_out,y0_out,gamma_out,alpha_out]= p.parameters
    fwhm_out = 2*gamma_out*np.sqrt(2**(1/alpha_out)-1)
    p_parameters = np.append(p.parameters,fwhm_out)
    parameters_MC = np.zeros((len(p_parameters),MC_loops))
    for l in range(MC_loops):
        iteration_data = np.random.normal(data,error) 
        if gamma_fixed is None and alpha_fixed is None:
            p_MC_init = models.Moffat2D(amplitude=amp,x_0=x0,y_0=y0,gamma=1,alpha=1)
        else:
            p_MC_init = models.Moffat2D(amp,x0,y0,gamma_fixed,alpha_fixed,fixed={'gamma':True,'alpha':True})
        f = fitting.LevMarLSQFitter()
        p_MC = f(p_MC_init, x, y, iteration_data)
        [amp_MC,x0_MC,y0_MC,gamma_MC,alpha_MC]= p_MC.parameters
        fwhm_MC = 2*gamma_MC*np.sqrt(2**(1/alpha_MC)-1)
        p_MC_parameters = np.append(p_MC.parameters, fwhm_MC)
        parameters_MC[:,l] = p_MC_parameters    
    parameters_err = np.std(parameters_MC,1) 
    [amp_err,x0_err,y0_err,gamma_err,alpha_err,fwhm_err] = parameters_err    
    if gamma_fixed is None and alpha_fixed is None:
        (par,err) = ([amp_out,x0_out,y0_out,gamma_out,alpha_out,fwhm_out],[amp_err,x0_err,y0_err,gamma_err,alpha_err,fwhm_err])   
    else:
        (par,err) = ([amp_out,x0_out,y0_out],[amp_err,x0_err,y0_err])   
    return par,err,model,res


# In[4]:


############################### Extract all the 3" parameters###########################
def red_data_err(data,model,res,x0,y0,x_m,y_m,muse_sampling_size,box_size=4):
    if muse_sampling_size == 0.4:
        (y_cen,x_cen)=(y_m,x_m)
        data_red = data[y_cen-box_size:y_cen+box_size+1,x_cen-box_size:x_cen+box_size+1]
        mod_red = model[y_cen-box_size:y_cen+box_size+1,x_cen-box_size:x_cen+box_size+1]
        residual_red = res[y_cen-box_size:y_cen+box_size+1,x_cen-box_size:x_cen+box_size+1]
        (x0,y0)=(x0-7,y0-7)
    else:
        (y_cen,x_cen)=(y_m,x_m)
        data_red = data[y_cen-2*box_size:y_cen+2*box_size+1,x_cen-2*box_size:x_cen+2*box_size+1]
        mod_red =model[y_cen-2*box_size:y_cen+2*box_size+1,x_cen-2*box_size:x_cen+2*box_size+1]
        residual_red =res[y_cen-2*box_size:y_cen+2*box_size+1,x_cen-2*box_size:x_cen+2*box_size+1]
        (x0,y0)=(x0-14,y0-14)
    #plt.imshow(data_red,origin='lower')
    #splt.show()
    err_red = (data_red - mod_red)/residual_red
    return data_red,mod_red,residual_red,err_red,x0,y0 


# In[5]:


####################### Extract the chi squared #########################################
def res_filter(res):
    res[res>10**10]=0
    return res

def chi_squared(data,model,err):
    res = data - model
    a = np.sum((res/err)**2)
    return a

def red_chi_squared(data,model,err,n_free):
    dof = len(data.flatten()) - n_free
    res = data - model
    k = res/err
    k[k>10**10]=0
    l = k[k!=0]
    a = np.sum(l**2)
    red = a/dof
    return red


# In[6]:


def central_table(obj,parm_err,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    column_names={'red_chisq_blr':0,'red_chisq_wing':1,'red_chisq_core':2,'norm_chisq_wing':3,'norm_chisq_core':4,
                  'psf_amp_blr':5,'psf_amp_wing':6,'psf_amp_core':7,'flux_ratio':8,'gamma':9,'alpha':10,'Hb_x':11,
                  'Hb_y':12,'OIII_br_x':13,'OIII_br_y':14,'OIII_nr_x':15,'OIII_nr_y':16,'off_wing_x':17,'off_wing_y':18,
                  'off_core_x':19,'off_core_y':20}
    columns=[]
    for key in column_names.keys():
        columns.append(fits.Column(name=key+'_err',format='E',array=[parm_err[column_names[key]]]))
    coldefs = fits.ColDefs(columns)
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto('%s/%s/%s_parameters_err_alternative.fits'%(destination_path_cube,obj,obj),overwrite=True)


# In[7]:


def algorithm_script(obj,z,broad2,p_init,box_size=11,min_wave=4750,max_wave=5090,prefix_path_cube="/home/mainak/ftp.hidrive.strato.com/users/login-carsftp/IFU_data",destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    print ('%s'%(obj))
    (wo_data,wo_err,wo_wave,wo_header) = loadCube('%s/%s/%s.wo_absorption.fits'%(destination_path_cube,obj,obj)) 
    (cont_cube,cont_err,cont_wave,cont_header) = loadCube('%s/MUSE/%s/fitting/full/%s.cont_model.fits'%(prefix_path_cube,obj,obj))
    MC_loops = 100
    Monte_Carlo_loops = 30
    popt_MC = np.zeros((21,Monte_Carlo_loops))
    muse_sampling_size = sampling_size(cont_cube)
    [x0,y0] = agn_location(obj)
    [brightest_pixel_x,brightest_pixel_y] = [int(x0),int(y0)]
        
    for l in range(Monte_Carlo_loops):
        k=1+z
        wo_cube = np.random.normal(wo_data,wo_err)
        
        (Hb1,Hb2,wing,core) = flux_dat(obj)
        (Hb1_err,Hb2_err,wing_err,core_err) = flux_err(obj)
        (Hb1,Hb2,wing,core) = (np.random.normal(Hb1,Hb1_err),np.random.normal(Hb2,Hb2_err),np.random.normal(wing,wing_err),np.random.normal(core,core_err))

        (emp_Hb1,emp_Hb2,emp_wing) = (err_ratio(Hb1,Hb1_err),err_ratio(Hb2,Hb2_err),err_ratio(wing,wing_err))
             
        (Hb_blr_br_data,OIII_br_data,OIII_nr_data,amp_Hb_blr_br,amp_OIII_br,amp_OIII_nr,Hb_blr_br_err,OIII_br_err,OIII_nr_err) = flux_data_err(obj,emp_Hb1,emp_Hb2,emp_wing)
        box_length = np.shape(Hb_blr_br_data)[1]
        (brightest_pixel_Hb_blr_br_x,brightest_pixel_Hb_blr_br_y,brightest_pixel_OIII_br_x,brightest_pixel_OIII_br_y,brightest_pixel_OIII_nr_x,brightest_pixel_OIII_nr_y) = brightest_pixel_flux_map(Hb_blr_br_data,OIII_br_data,OIII_nr_data) 
        (Hb_par,Hb_error,Hb_model,blr_res) = moffat_fit(Hb_blr_br_data,Hb_blr_br_err,box_length,amp_Hb_blr_br,brightest_pixel_Hb_blr_br_x,brightest_pixel_Hb_blr_br_y,MC_loops,None,None)
        (gamma_fix,alpha_fix) = (Hb_par[3],Hb_par[4])#these two are gamma and alpha
        (OIII_br_par,OIII_br_error,OIII_br_model,wing_res) = moffat_fit(OIII_br_data,OIII_br_err,box_length,amp_OIII_br,brightest_pixel_OIII_br_x,brightest_pixel_OIII_br_y,MC_loops,gamma_fix,alpha_fix)   
        (OIII_nr_par,OIII_nr_error,OIII_nr_model,core_res) = moffat_fit(OIII_nr_data,OIII_nr_err,box_length,amp_OIII_nr,brightest_pixel_OIII_nr_x,brightest_pixel_OIII_nr_y,MC_loops,gamma_fix,alpha_fix) 
        
        (Hb_cen_x,Hb_cen_y) = (Hb_par[1],Hb_par[2])
        (OIII_br_cen_x,OIII_br_cen_y) = (OIII_br_par[1],OIII_br_par[2])
        (OIII_nr_cen_x,OIII_nr_cen_y) = (OIII_nr_par[1],OIII_nr_par[2])
        
        (Hb_amp,OIII_br_amp,OIII_nr_amp) = (Hb_par[0],OIII_br_par[0],OIII_nr_par[0])
        (y_m,x_m) = ndimage.measurements.maximum_position(Hb_blr_br_data)
        #print (x_m,y_m)
        Hb_res = res_filter(blr_res)
        OIII_br_res = res_filter(wing_res)
        OIII_nr_res = res_filter(core_res)
    
        (Hb_data,Hb_model,Hb_res,Hb_err,Hb_x,Hb_y) = red_data_err(Hb_blr_br_data,Hb_model,Hb_res,Hb_cen_x,Hb_cen_y,x_m,y_m,muse_sampling_size,box_size=4)
        (OIII_br_data,OIII_br_model,OIII_br_res,OIII_br_err,OIII_br_x,OIII_br_y) = red_data_err(OIII_br_data,OIII_br_model,OIII_br_res,OIII_br_cen_x,OIII_br_cen_y,x_m,y_m,muse_sampling_size,box_size=4)
        (OIII_nr_data,OIII_nr_model,OIII_nr_res,OIII_nr_err,OIII_nr_x,OIII_nr_y) = red_data_err(OIII_nr_data,OIII_nr_model,OIII_nr_res,OIII_nr_cen_x,OIII_nr_cen_y,x_m,y_m,muse_sampling_size,box_size=4)
        (y_m,x_m) = ndimage.measurements.maximum_position(Hb_data)
        #print (x_m,y_m)
        chi_squared_Hb = chi_squared(Hb_data,Hb_model,Hb_err)
        chi_squared_OIII_br = chi_squared(OIII_br_data,OIII_br_model,OIII_br_err)
        chi_squared_OIII_nr = chi_squared(OIII_nr_data,OIII_nr_model,OIII_nr_err)

        red_chi_squared_Hb = red_chi_squared(Hb_data,Hb_model,Hb_err,5)
        red_chi_squared_OIII_br = red_chi_squared(OIII_br_data,OIII_br_model,OIII_br_err,3)
        red_chi_squared_OIII_nr = red_chi_squared(OIII_nr_data,OIII_nr_model,OIII_nr_err,3)

        normalized_chi_squared_OIII_br = (red_chi_squared_OIII_br/red_chi_squared_Hb)
        normalized_chi_squared_OIII_nr = (red_chi_squared_OIII_nr/red_chi_squared_Hb)
    
        flux_wing_data = np.sum(OIII_br_data)
        flux_wing_model = np.sum(OIII_br_model)
        flux_ratio = (flux_wing_data/flux_wing_model)
        
        (offset_wing_x,offset_wing_y) = (OIII_br_x - Hb_x,OIII_br_y - Hb_y)
        (offset_core_x,offset_core_y) = (OIII_nr_x - Hb_x,OIII_nr_y - Hb_y)
        popt = [red_chi_squared_Hb,red_chi_squared_OIII_br,red_chi_squared_OIII_nr,normalized_chi_squared_OIII_br,normalized_chi_squared_OIII_nr,Hb_amp,OIII_br_amp,OIII_nr_amp,flux_ratio,gamma_fix,alpha_fix,Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y,offset_wing_x,offset_wing_y,offset_core_x,offset_core_y]
        print (popt)
        popt_MC[:,l] = popt
    parm_err = np.std(popt_MC,1)
    print ('norm chi-sq err and flux_err is', parm_err)
    #plt.imshow((OIII_br_data - OIII_br_model)/OIII_br_err,origin='lower')
    #plt.show()
    central_table(obj,parm_err)
    


# In[ ]:




z = {"HE1237-0504":0.009,"HE1248-1356":0.01465,"HE1330-1013":0.022145,"HE1353-1917":0.035021,"HE1417-0909":0.044,"HE2128-0221":0.05248
    ,"HE2211-3903":0.039714,"HE2222-0026":0.059114,"HE2233+0124":0.056482,"HE2302-0857":0.046860}

objs = z.keys()

broad2= {'HE1237-0504':False,'HE1248-1356':False,'HE1330-1013':True,'HE1353-1917':True,'HE1417-0909':False,'HE2128-0221':False
        ,'HE2211-3903':False,'HE2222-0026':True,'HE2233+0124':True,'HE2302-0857':False}

p_init= {'HE1237-0504':[1,10,2700.0,80.0,0.5,2,2700.0,200,2,0.5,2700.0,1500.0,0,0,2700.0,1000.0,-0.001,0.002]
        ,'HE1248-1356':[0.1,1.5,4395.0,50.0,1.0,3.0,4195.0,100.0,1.0,1.0,4395,1000.0,0,0,0,0.0,-0.001,2.0]
        ,'HE1330-1013':[0.15,0.6,6643,90,0.05,0.15,6543,200,0.12,0.04,6643,1500,0.3,0.04,6643,500,-0.02,0.3]
        ,'HE1353-1917':[0.07,0.7,10490.0,80.0,0.39,0.05,10306.0,490.0,0.1,0.02,8600.0,1500.0,0.14,0.02,12326.0,1500.0,-0.001,0.002]
        ,'HE1417-0909':[1,12.5,13200,50.0,3,3,13000,100.0,2,2,13200,1000.0,0,0,13200,1000.0,-0.001,0.1]
        ,'HE2128-0221':[0.1,1.5,15744,50.0,1.0,3.0,15544,100.0,1.0,1.0,15744,1000.0,0,0,15744,1000.0,-0.001,2.0]
        ,'HE2211-3903':[0.6,2.4,11914,50.0,0.1,0.1,11714,100.0,0.1,0.1,11914,200.0,0,0,11914,100.0,-0.001,0.2]
        ,'HE2222-0026':[0.08,0.4,17400.0,140.0,0.04,0.05,17150.0,300.0,0.4,0.1,18500,650.0,0.02,0.01,17460,1750.0,-0.001,0.002]
        ,'HE2233+0124':[0.1,1.2,16944.0,100.0,1.0,3.0,17044.0,300.0,1.0,1.0,16944,1200.0,1.0,1.0,16944,4000.0,-0.001,2.0]
        ,'HE2302-0857':[1,8,14058,200,0.1,2,14258,300,1,0.1,14058,1000,0,0,0,0,0.01,0.2]}

for obj in objs:
     algorithm_script(obj,z[obj],broad2[obj],p_init[obj]) 


# In[ ]:




