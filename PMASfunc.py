import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from numpy import exp
from astropy.modeling import models, fitting
from scipy import ndimage
import pyneb as pn


def loadCube(filename):
    hdu = fits.open(filename)
    cube = hdu[0].data
    global err
    try:
        err = hdu[1].data
    except IndexError:
        err = 0
    header = hdu[0].header
    hdu.close()
    wavestart = header['CRVAL3']
    try:
        wavint = header['CD3_3']
    except KeyError:
        wavint = header['CDELT3']  
    wave = wavestart+np.arange(cube.shape[0])*wavint
    return cube,err,wave,header

#If you do define store_cube as 
def store_cube(filename,mini_cube_data,wave,mini_cube_err=None,header=None): #
    if mini_cube_err is None:
        hdu_out = fits.PrimaryHDU(mini_cube_data)
    else:
        hdu_out = fits.HDUList([fits.PrimaryHDU(mini_cube_data),fits.ImageHDU(mini_cube_err)])
    if header is not None:
        hdu_out[0].header = header
    hdu_out[0].header['CRPIX3'] = wave[1]
    hdu_out[0].header['CRVAL3'] = wave[0]
    hdu_out[0].header['CDELT3'] = (wave[1]-wave[0])
    hdu_out.writeto(filename,overwrite=True)
    
def loadmap(filename):
    hdu = fits.open(filename)
    (OIII_nr,OIII_br,Hb1_blr_br,Hb2_blr_br) = (hdu[2].data,hdu[3].data,hdu[5].data,hdu[6].data)
    (amp_OIII_nr,amp_OIII_br,amp_Hb1_blr_br,amp_Hb2_blr_br) = (np.max(OIII_nr),np.max(OIII_br),np.max(Hb1_blr_br),np.max(Hb2_blr_br))
    if amp_Hb1_blr_br > amp_Hb2_blr_br:
        (Hb_blr_br,amp_Hb_blr_br) = (Hb1_blr_br,amp_Hb1_blr_br)
    else:
        (Hb_blr_br,amp_Hb_blr_br) = (Hb2_blr_br,amp_Hb2_blr_br)
    return Hb_blr_br,OIII_br,OIII_nr,amp_Hb_blr_br,amp_OIII_br,amp_OIII_nr
    
   
def loadplot(filename):
    hdu = fits.open(filename)
    (Hb_data,OIII_br_data,OIII_nr_data)=(hdu[1].data,hdu[2].data,hdu[3].data)
    (Hb_model,OIII_br_model,OIII_nr_model) = (hdu[4].data,hdu[5].data,hdu[6].data)
    (Hb_res,OIII_br_res,OIII_nr_res) = (hdu[7].data,hdu[8].data,hdu[9].data)
    return Hb_data,Hb_model,Hb_res,OIII_br_data,OIII_br_model,OIII_br_res,OIII_nr_data,OIII_nr_model,OIII_nr_res

def loadblr(filename):
    hdu = fits.open(filename)
    (Hb1_blr_br_data,Hb2_blr_br_data) = (hdu[5].data,hdu[6].data)
    return Hb1_blr_br_data,Hb2_blr_br_data
    
def loadwing(filename):
    hdu = fits.open(filename)
    OIII_br_model = hdu[5].data
    return OIII_br_model

def loadwing_err(filename):
    hdu = fits.open(filename)
    OIII_br_err_model = hdu[3].data
    return OIII_br_err_model

def loadHbwing(filename):
    hdu = fits.open(filename)
    Hb_br = hdu[4].data
    return Hb_br
    
def loadtab(filename):
    hdu = fits.open(filename)
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    amp_Hb = central_tab.field('amp_Hb')[0]
    amp_OIII5007 = central_tab.field('amp_OIII5007')[0] 
    
    amp_Hb_br = central_tab.field('amp_Hb_br')[0]
    amp_OIII5007_br = central_tab.field('amp_OIII5007_br')[0]
    
    amp_Hb1 = central_tab.field('amp_Hb1')[0]
    amp_Hb2 = central_tab.field('amp_Hb2')[0]

    amp_Fe5018_1 = central_tab.field('amp_Fe5018_1')[0]
    amp_Fe5018_2 = central_tab.field('amp_Fe5018_2')[0]

    
    vel_OIII = central_tab.field('vel_OIII')[0]
    vel_sigma_OIII = central_tab.field('vel_sigma_OIII')[0]  
    vel_OIII_br = central_tab.field('vel_OIII_br')[0]
    vel_sigma_OIII_br = central_tab.field('vel_sigma_OIII_br')[0]
    
    vel_Hb1 = central_tab.field('vel_Hb1')[0]
    vel_Hb2 = central_tab.field('vel_Hb2')[0]
    vel_sigma_Hb1 = central_tab.field('vel_sigma_Hb1')[0]
    vel_sigma_Hb2 = central_tab.field('vel_sigma_Hb2')[0]
    
    m = central_tab.field('m')[0]
    c = central_tab.field('c')[0]
    
    return amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c

def loadspectab(filename):
    hdu = fits.open(filename)
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    amp_Hb = central_tab.field('amp_Hb')[0]
    amp_OIII5007 = central_tab.field('amp_OIII5007')[0] 
    
    amp_Hb_br = central_tab.field('amp_Hb_br')[0]
    amp_OIII5007_br = central_tab.field('amp_OIII5007_br')[0]
    
    amp_Hb1 = central_tab.field('amp_Hb1')[0]
    amp_Hb2 = central_tab.field('amp_Hb2')[0]

    amp_Fe5018_1 = central_tab.field('amp_Fe5018_1')[0]
    amp_Fe5018_2 = central_tab.field('amp_Fe5018_2')[0]
    
    m = central_tab.field('m')[0]
    c = central_tab.field('c')[0]
    
    return amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c

def create_wo_absorption_cube(obj,orig_cube,orig_err,orig_header,cont_cube,cont_wave,difference):
    if np.shape(orig_cube)[0] > np.shape(cont_cube)[0]:
        (orig_cube,orig_err) = (orig_cube[:-difference,:,:],orig_err[:-difference,:,:]) 
    else:
        (orig_cube,orig_err) = (orig_cube,orig_err) 
    (wo_cube,wo_err,wo_wave,wo_header) = (orig_cube - cont_cube, orig_err, cont_wave,orig_header)
    return wo_cube,wo_err,wo_wave,wo_header   

def create_eline_cube(wo_cube,QSO_cube,difference_em):
    if np.shape(QSO_cube)[0] > np.shape(wo_cube)[0]:
        new_QSO_cube = QSO_cube[:-difference_em,:,:]
    else:
        new_QSO_cube = QSO_cube
    em_cube = wo_cube - new_QSO_cube
    return em_cube      

def redshift(vel):
    return vel/300000.0
    #This function also represent the line dispersion in A through a velocity dispersion in km/s also taking into account 
    # that the spectrograph itself already broadens the emission lines. This way you automatically fit for the intrinsic line dispersion
def line_width(vel_sigma,rest_line,inst_res_fwhm):
    sigma = vel_sigma/(300000.0-vel_sigma)*rest_line
    return np.sqrt(sigma**2+(inst_res_fwhm/3.0)**2)

def line_width_recons(vel_sigma,rest_line,inst_res_fwhm=0):
    sigma = vel_sigma/(300000.0-vel_sigma)*rest_line
    return np.sqrt(sigma**2+(inst_res_fwhm/3.0)**2)

def gauss(wave,amplitude,vel,vel_sigma, rest_wave,inst_res_fwhm):
    line = (amplitude)*exp(-(wave-(rest_wave*(1+redshift(vel))))**2/(2*(line_width(vel_sigma, rest_wave,inst_res_fwhm))**2))
    return line

def gauss_recons(wave,amplitude,vel,vel_sigma, rest_wave):
    line = (amplitude)*exp(-(wave-(rest_wave*(1+redshift(vel))))**2/(2*(line_width_recons(vel_sigma, rest_wave))**2))
    return line

def double_BLR(wave,amp_Hb1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,vel_Hb2,vel_sigma_Hb2):
    Hb1 = gauss_recons(wave,amp_Hb1,vel_Hb1,vel_sigma_Hb1,4861.33)
    Hb2 = gauss_recons(wave,amp_Hb2,vel_Hb2,vel_sigma_Hb2,4861.33)
    return Hb1+Hb2

    # Here we couple the HB and OIII doublet together using the gaussian function defined before
def Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel,vel_sigma):
    Hb = gauss(wave,amp_Hb,vel,vel_sigma,4861.33,3.1)
    OIII_4959 = (0.33)*gauss(wave,amp_OIII5007,vel,vel_sigma,4958.9,3.1)
    OIII_5007 = gauss(wave,amp_OIII5007,vel,vel_sigma,5006.8,3.1)
    return Hb + OIII_4959 + OIII_5007

    # Same as before but fore the Fe doublet
def Fe_doublet_gauss(wave,amp_Fe4923,amp_Fe5018,vel,vel_sigma):
    Fe_4923 = gauss(wave,amp_Fe4923,vel,vel_sigma,4923,3.1)
    Fe_5018 = gauss(wave,amp_Fe5018,vel,vel_sigma,5018,3.1)
    return Fe_4923+Fe_5018

def Hb_Fe_doublet_gauss(wave,amp_Hb,amp_Fe5018,vel,vel_sigma):
    Hb = gauss(wave,amp_Hb,vel,vel_sigma,4861.33,3.1)
    Fe_4923 = 0.81*gauss(wave,amp_Fe5018,vel,vel_sigma,4923,3.1)
    Fe_5018 = gauss(wave,amp_Fe5018,vel,vel_sigma,5018,3.1)
    return Hb+Fe_4923+Fe_5018

def OIII_wo_cont(wave,amp_OIII5007,amp_OIII5007_br,vel,vel_sigma,vel_br,vel_sigma_br):  
    OIII_5007_core = gauss_recons(wave,amp_OIII5007,vel,vel_sigma,5006.8)
    OIII_5007_wing = gauss_recons(wave,amp_OIII5007_br,vel_br,vel_sigma_br,5006.8)
    return OIII_5007_core + OIII_5007_wing
 
def SII_doublet_gauss(wave,amp_SII6716,amp_SII6731,vel,vel_sigma):
    SII_6716 = gauss(wave,amp_SII6716,vel,vel_sigma,6716.44,3.1)
    SII_6731 = gauss(wave,amp_SII6731,vel,vel_sigma,6730.82,3.1)
    return SII_6716+SII_6731

def nlr_gauss_decoupled(wave,amp_Ha,amp_NII6583,vel,vel_sigma):
    Ha = gauss(wave,amp_Ha,vel,vel_sigma,6562.85,2.5)
    NII_6548 = 0.33*gauss(wave,amp_NII6583,vel,vel_sigma,6548.05,2.5)
    NII_6583 = gauss(wave,amp_NII6583,vel,vel_sigma,6583.45,2.5)
    return Ha + NII_6548 + NII_6583 

def blr_gauss(wave,amp_Ha,vel,vel_sigma):
    Ha = gauss(wave,amp_Ha,vel,vel_sigma,6562.85,2.5)
    return Ha
    
def nlr_gauss_coupled(wave,amp_Ha,amp_NII6583,amp_SII6716,amp_SII6731,vel,vel_sigma):
    Ha = gauss(wave,amp_Ha,vel,vel_sigma,6562.85,2.5)
    NII_6548 = 0.33*gauss(wave,amp_NII6583,vel,vel_sigma,6548.05,2.5)
    NII_6583 = gauss(wave,amp_NII6583,vel,vel_sigma,6583.45,2.5)
    SII_6716 = gauss(wave,amp_SII6716,vel,vel_sigma,6716.44,2.5)
    SII_6731 = gauss(wave,amp_SII6731,vel,vel_sigma,6730.82,2.5)
    return Ha + NII_6548 + NII_6583 + SII_6716 + SII_6731
    
#==============================================================================
def SII_coupled_gauss(p,wave,data,error,fixed_param):
    (amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,m,c) = p
    [vel_sigma_OIII_core,vel_sigma_OIII_wing] = fixed_param
    SII_core = SII_doublet_gauss(wave,amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,vel_sigma_OIII_core)
    SII_wing = SII_doublet_gauss(wave,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,vel_sigma_OIII_wing)
    cont = (wave/1000.0)*m+c
    return (SII_core+SII_wing+cont-data)/error

def complex_gauss_coupled(p,wave,data,error):
     #(amp_Ha_core1,amp_NII6583_core1,amp_SII6716_core1,amp_SII6731_core1,vel_core1,vel_sigma_core1,amp_Ha_core2,amp_NII6583_core2,amp_SII6716_core2,amp_SII6731_core2,vel_core2,vel_sigma_core2,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,amp_Ha_blr4,vel_blr4,vel_sigma_blr4,m,c)= p
     (amp_Ha_core,amp_NII6583_core,amp_SII6716_core,amp_SII6731_core,vel_core,vel_sigma_core,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,m,n,c)= p  
     nlr_core = nlr_gauss_coupled(wave,amp_Ha_core,amp_NII6583_core,amp_SII6716_core,amp_SII6731_core,vel_core,vel_sigma_core)
#     nlr_core2 = nlr_gauss(wave,amp_Ha_core2,amp_NII6583_core2,amp_SII6716_core2,amp_SII6731_core2,vel_core2,vel_sigma_core2)
     nlr_wing = nlr_gauss_coupled(wave,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing)
#     nlr_wing = 0
     blr1 = blr_gauss(wave,amp_Ha_blr1,vel_blr1,vel_sigma_blr1)
     #blr2 = 0
     blr2 = blr_gauss(wave,amp_Ha_blr2,vel_blr2,vel_sigma_blr2)
     blr3 = blr_gauss(wave,amp_Ha_blr3,vel_blr3,vel_sigma_blr3)    
#     blr4 = blr_gauss(wave,amp_Ha_blr4,vel_blr4,vel_sigma_blr4)
     #blr3 = 0
     cont = m*((wave/1000.0)**2)+n*(wave/1000.0)+c
     return (nlr_core++nlr_wing+blr1+blr2+blr3+cont-data)/error
#============================================================================== 
    
def complex_gauss_decoupled(p,wave,data,error):
    (amp_Ha_core,amp_NII6583_core,vel_Ha_core,vel_sigma_Ha_core,amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,vel_sigma_SII6731_core,amp_Ha_wing,amp_NII6583_wing,vel_Ha_wing,vel_sigma_Ha_wing,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,vel_sigma_SII6731_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,m,n,c)= p
    Ha_NII_core = nlr_gauss_decoupled(wave,amp_Ha_core,amp_NII6583_core,vel_Ha_core,vel_sigma_Ha_core)
    SII_core = SII_doublet_gauss(wave,amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,vel_sigma_SII6731_core)
    Ha_NII_wing = nlr_gauss_decoupled(wave,amp_Ha_wing,amp_NII6583_wing,vel_Ha_wing,vel_sigma_Ha_wing)
    SII_wing = SII_doublet_gauss(wave,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,vel_sigma_SII6731_wing)
    blr1 = blr_gauss(wave,amp_Ha_blr1,vel_blr1,vel_sigma_blr1)
    #blr2 = 0
    blr2 = blr_gauss(wave,amp_Ha_blr2,vel_blr2,vel_sigma_blr2)
    blr3 = blr_gauss(wave,amp_Ha_blr3,vel_blr3,vel_sigma_blr3)
    #blr3 = 0
    cont = m*((wave/1000.0)**2)+n*(wave/1000.0)+c
    return (Ha_NII_core+SII_core+Ha_NII_wing+SII_wing+blr1+blr2+blr3+cont-data)/error

def fixed_par_vel_sigma(obj):
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    vel_sigma_OIII = central_tab.field('vel_sigma_OIII')[0]
    vel_sigma_OIII_br = central_tab.field('vel_sigma_OIII_br')[0]
    vel_sigma_OIII_err = central_tab.field('vel_sigma_OIII_err')[0]
    vel_sigma_OIII_br_err = central_tab.field('vel_sigma_OIII_br_err')[0]
    fixed_param = [vel_sigma_OIII,vel_sigma_OIII_br]  
    fixed_param_err = [vel_sigma_OIII_err,vel_sigma_OIII_br_err] 
    return fixed_param,fixed_param_err

def vel_sigma_coupled_gauss(p,wave,data,error,fixed_param):
    (amp_Ha_core,amp_NII6583_core,vel_Ha_core,amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,amp_Ha_wing,amp_NII6583_wing,vel_Ha_wing,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,m,n,c)= p
    [vel_sigma_OIII_core,vel_sigma_OIII_wing] = fixed_param
    Ha_NII_core = nlr_gauss_decoupled(wave,amp_Ha_core,amp_NII6583_core,vel_Ha_core,vel_sigma_OIII_core)
    SII_core = SII_doublet_gauss(wave,amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,vel_sigma_OIII_core)
    Ha_NII_wing = nlr_gauss_decoupled(wave,amp_Ha_wing,amp_NII6583_wing,vel_Ha_wing,vel_sigma_OIII_wing)
    SII_wing = SII_doublet_gauss(wave,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,vel_sigma_OIII_wing)
    blr1 = blr_gauss(wave,amp_Ha_blr1,vel_blr1,vel_sigma_blr1)
    #blr2 = 0
    blr2 = blr_gauss(wave,amp_Ha_blr2,vel_blr2,vel_sigma_blr2)
    blr3 = blr_gauss(wave,amp_Ha_blr3,vel_blr3,vel_sigma_blr3)
    #blr3 = 0
    cont = m*((wave/1000.0)**2)+n*(wave/1000.0)+c
    return (Ha_NII_core+SII_core+Ha_NII_wing+SII_wing+blr1+blr2+blr3+cont-data)/error
    
def full_gauss_SII(wave,amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br,m,c):
    narrow_SII = SII_doublet_gauss(wave,amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731)
    broad_SII = SII_doublet_gauss(wave,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br)
    cont = (wave/1000.0)*m+c
    return narrow_SII+ broad_SII+cont

def test_gauss_SII(p,wave,data,error):
    (amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br,m,n,c)=p
    narrow_SII = SII_doublet_gauss(wave,amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731)
    broad_SII = SII_doublet_gauss(wave,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br)
    #broad_SII = 0
    cont = m*((wave/1000.0)**2)+n*(wave/1000.0)+c
    return (narrow_SII+ broad_SII+cont-data)/error

def single_gauss_SII(p,wave,data,error):
    (amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br,m,c)=p
    narrow_SII = SII_doublet_gauss(wave,amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731)
    #broad_SII = SII_doublet_gauss(wave,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br)
    broad_SII = 0
    cont = (wave/1000.0)*m+c
    return (narrow_SII+ broad_SII+cont-data)/error


def full_gauss1(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    #broad_OIII = 0
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = 0 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error


def full_gauss2(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    #broad_OIII = 0
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error
    
def full_gauss3(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,amp_Hb3,amp_Fe5018_3,vel_Hb3,vel_sigma_Hb3,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    #broad_OIII = 0
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    Hb_broad3 = Hb_Fe_doublet_gauss(wave,amp_Hb3,amp_Fe5018_3,vel_Hb3,vel_sigma_Hb3) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error


def continuum(wave,m,c):
    slope = (wave/1000.0)*m
    const = c
    return slope + const

def continuum_qua(wave,m,n,c):
    curve = m*((wave/1000.0)**2)+n*(wave/1000.0)
    const = c
    return curve + const


def full_gauss1_fixkin(p,wave,data,error,fixed_param):
    (amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c) = p 
    [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2] = fixed_param 
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1)
    Hb_broad2 = 0 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error

def full_gauss2_fixkin(p,wave,data,error,fixed_param):
    (amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c) = p 
    [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2] = fixed_param 
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error
   
def brightest_pixel(wo_cube):
    wo_slice = wo_cube[0,:,:]
    [guess_y,guess_x] = ndimage.measurements.maximum_position(wo_slice)
    return guess_x, guess_y

    
def fixed_parameters(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/PMAS"):
    hdu = fits.open('%s/%s/%s_central_fit.fits'%(destination_path_cube,obj,obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    vel_OIII = central_tab.field('vel_OIII')[0]
    vel_sigma_OIII = central_tab.field('vel_sigma_OIII')[0]
    vel_OIII_br = central_tab.field('vel_OIII_br')[0]
    vel_sigma_OIII_br = central_tab.field('vel_sigma_OIII_br')[0]
    vel_Hb1 = central_tab.field('vel_Hb1')[0]
    vel_Hb2 = central_tab.field('vel_Hb2')[0]
    vel_sigma_Hb1 = central_tab.field('vel_sigma_Hb1')[0]
    vel_sigma_Hb2 = central_tab.field('vel_sigma_Hb2')[0]
    fixed_param = [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2]  
    return fixed_param
    
def light_weighted_centroid(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/PMAS"):
    hdu = fits.open('%s/Flux Maps/%s/13_arcsec_subcube_par_%s.fits'%(destination_path_cube,obj,obj))
    (OIII_nr_data,OIII_br_data,Hb1_br_data,Hb2_br_data) = (hdu[2].data,hdu[3].data,hdu[5].data,hdu[6].data)
    centroid_OIII_nr = ndimage.measurements.center_of_mass(OIII_nr_data)
    centroid_OIII_br = ndimage.measurements.center_of_mass(OIII_br_data)
    centroid_Hb1_br = ndimage.measurements.center_of_mass(Hb1_br_data)
    centroid_Hb2_br = ndimage.measurements.center_of_mass(Hb2_br_data)
    if np.max(Hb1_br_data) > np.max(Hb2_br_data):
        return centroid_Hb1_br,centroid_OIII_br,centroid_OIII_nr
    else:
        return centroid_Hb2_br,centroid_OIII_br,centroid_OIII_nr

def brightest_pixel_flux_map(Hb_blr_br_data,OIII_br_data,OIII_nr_data):
    [brightest_pixel_Hb_blr_br_y,brightest_pixel_Hb_blr_br_x] = ndimage.measurements.maximum_position(Hb_blr_br_data)
    [brightest_pixel_OIII_br_y,brightest_pixel_OIII_br_x] = ndimage.measurements.maximum_position(OIII_br_data)
    [brightest_pixel_OIII_nr_y,brightest_pixel_OIII_nr_x] = ndimage.measurements.maximum_position(OIII_nr_data)
    return brightest_pixel_Hb_blr_br_x,brightest_pixel_Hb_blr_br_y,brightest_pixel_OIII_br_x,brightest_pixel_OIII_br_y,brightest_pixel_OIII_nr_x,brightest_pixel_OIII_nr_y
    


def centers(obj):
    hdu = fits.open('moffat_table_%s.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    (Hb_x,Hb_y) = (central_tab.field('x0_Hb_Blr')[0],central_tab.field('y0_Hb_Blr')[0])
    (OIII_br_x,OIII_br_y) = (central_tab.field('x0_wing')[0],central_tab.field('y0_wing')[0])
    (OIII_nr_x,OIII_nr_y) = (central_tab.field('x0_core')[0],central_tab.field('y0_core')[0])
    return Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y  

def offset(Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y,muse_sampling_size):
    offset_OIII_br_pixel= [(OIII_br_x - Hb_x),(OIII_br_y - Hb_y)]
    offset_OIII_nr_pixel= [(OIII_nr_x - Hb_x),(OIII_nr_y - Hb_y)]
    offset_OIII_br_arcsec = np.asarray(offset_OIII_br_pixel)*muse_sampling_size 
    offset_OIII_nr_arcsec = np.asarray(offset_OIII_nr_pixel)*muse_sampling_size 
    return offset_OIII_br_pixel,offset_OIII_nr_pixel,offset_OIII_br_arcsec,offset_OIII_nr_arcsec
    
def ranges(Hb_x,Hb_y,muse_sampling_size,asymmetry=False):
    sampling_size = 0.66
    if asymmetry:
        size = 6
    else:
        size = 5
    (x_min,x_max) = (-(Hb_x+0.5)*sampling_size,(size-1-Hb_x+0.5)*sampling_size)
    (y_min,y_max) = (-(Hb_y+0.5)*sampling_size,(size-1-Hb_y+0.5)*sampling_size)
    return x_min,x_max,y_min,y_max

def ranges_talk(Hb_x,Hb_y,muse_sampling_size,asymmetry=False):
    if muse_sampling_size == 0.2:
        size = 17
    else:
        size = 9
    (x_min,x_max) = (-(Hb_x+0.5)*muse_sampling_size,(size-1-Hb_x+0.5)*muse_sampling_size)
    (y_min,y_max) = (-(Hb_y+0.5)*muse_sampling_size,(size-1-Hb_y+0.5)*muse_sampling_size)
    return x_min,x_max,y_min,y_max

def par(obj):
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (amp_Hb1,amp_Hb2) = (central_tab.field('amp_Hb1')[0],central_tab.field('amp_Hb2')[0])
    (vel_Hb1,vel_Hb2) = (central_tab.field('vel_Hb1')[0],central_tab.field('vel_Hb2')[0])
    (vel_sigma_Hb1,vel_sigma_Hb2) = (central_tab.field('vel_sigma_Hb1')[0],central_tab.field('vel_sigma_Hb2')[0])
    (amp_OIII5007,amp_OIII5007_br) = (central_tab.field('amp_OIII5007')[0],central_tab.field('amp_OIII5007_br')[0])
    (vel_sigma_OIII,vel_sigma_OIII_br) = (central_tab.field('vel_sigma_OIII')[0],central_tab.field('vel_sigma_OIII_br')[0])
    (vel_OIII,vel_OIII_br) = (central_tab.field('vel_OIII')[0],central_tab.field('vel_OIII_br')[0])
    (m,c) = (central_tab.field('m')[0],central_tab.field('c')[0])
    vel_offset = vel_OIII - vel_OIII_br
    
    return amp_Hb1,amp_Hb2,vel_Hb1,vel_Hb2,vel_sigma_Hb1,vel_sigma_Hb2,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,vel_offset,m,c

def par_nonspectro(obj):
    hdu = fits.open('%s_non_spectro_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (amp_Hb1,amp_Hb2) = (central_tab.field('amp_Hb1')[0],central_tab.field('amp_Hb2')[0])
    (vel_Hb1,vel_Hb2) = (central_tab.field('vel_Hb1')[0],central_tab.field('vel_Hb2')[0])
    (vel_sigma_Hb1,vel_sigma_Hb2) = (central_tab.field('vel_sigma_Hb1')[0],central_tab.field('vel_sigma_Hb2')[0])
    (amp_OIII5007,amp_OIII5007_br) = (central_tab.field('amp_OIII5007')[0],central_tab.field('amp_OIII5007_br')[0])
    (vel_sigma_OIII,vel_sigma_OIII_br) = (central_tab.field('vel_sigma_OIII')[0],central_tab.field('vel_sigma_OIII_br')[0])
    (vel_OIII,vel_OIII_br) = (central_tab.field('vel_OIII')[0],central_tab.field('vel_OIII_br')[0])
    (m,c) = (central_tab.field('m')[0],central_tab.field('c')[0])
    vel_offset = vel_OIII - vel_OIII_br
    
    return amp_Hb1,amp_Hb2,vel_Hb1,vel_Hb2,vel_sigma_Hb1,vel_sigma_Hb2,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,vel_offset,m,c

def par_spectro(obj):
    hdu = fits.open('%s_spectro_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (amp_Hb1,amp_Hb2) = (central_tab.field('amp_Hb1')[0],central_tab.field('amp_Hb2')[0])
    (amp_OIII5007,amp_OIII5007_br) = (central_tab.field('amp_OIII5007')[0],central_tab.field('amp_OIII5007_br')[0])
    (m,c) = (central_tab.field('m')[0],central_tab.field('c')[0])
    hdu.close()
    
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (vel_Hb1,vel_Hb2) = (central_tab.field('vel_Hb1')[0],central_tab.field('vel_Hb2')[0])
    (vel_sigma_Hb1,vel_sigma_Hb2) = (central_tab.field('vel_sigma_Hb1')[0],central_tab.field('vel_sigma_Hb2')[0])
    (vel_sigma_OIII,vel_sigma_OIII_br) = (central_tab.field('vel_sigma_OIII')[0],central_tab.field('vel_sigma_OIII_br')[0])
    (vel_OIII,vel_OIII_br) = (central_tab.field('vel_OIII')[0],central_tab.field('vel_OIII_br')[0])
    vel_offset = vel_OIII - vel_OIII_br
    hdu.close()
    
    return amp_Hb1,amp_Hb2,vel_Hb1,vel_Hb2,vel_sigma_Hb1,vel_sigma_Hb2,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,vel_offset,m,c

def par_spectro_err(obj):
    hdu = fits.open('%s_spectro_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (amp_Hb1_err,amp_Hb2_err) = (central_tab.field('amp_Hb1_err')[0],central_tab.field('amp_Hb2_err')[0])
    (amp_OIII5007_err,amp_OIII5007_br_err) = (central_tab.field('amp_OIII5007_err')[0],central_tab.field('amp_OIII5007_br_err')[0])
    (m,c) = (central_tab.field('m')[0],central_tab.field('c')[0])
    hdu.close()
    
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (vel_Hb1_err,vel_Hb2_err) = (central_tab.field('vel_Hb1_err')[0],central_tab.field('vel_Hb2_err')[0])
    (vel_sigma_Hb1_err,vel_sigma_Hb2_err) = (central_tab.field('vel_sigma_Hb1_err')[0],central_tab.field('vel_sigma_Hb2_err')[0])
    (vel_sigma_OIII_err,vel_sigma_OIII_br_err) = (central_tab.field('vel_sigma_OIII_err')[0],central_tab.field('vel_sigma_OIII_br_err')[0])
    (vel_OIII_err,vel_OIII_br_err) = (central_tab.field('vel_OIII_err')[0],central_tab.field('vel_OIII_br')[0])
    hdu.close()
    
    return amp_Hb1_err,amp_Hb2_err,vel_Hb1_err,vel_Hb2_err,vel_sigma_Hb1_err,vel_sigma_Hb2_err,amp_OIII5007_err,vel_OIII_err,vel_sigma_OIII_err,amp_OIII5007_br_err,vel_OIII_br_err,vel_sigma_OIII_br_err,m_err,c_err

def wavlim(vel_OIII,vel_OIII_br):
    c = 300000 # km/s
    k_OIII = 1+(vel_OIII/c)
    k_OIII_br = 1+(vel_OIII_br/c)
    wav_min = (5007*k_OIII_br) - 100
    wav_max = (5007*k_OIII) + 100
    return wav_min,wav_max

def int_spec(mini_cube,mini_err):
    shape = mini_cube.shape[1]*mini_cube.shape[2]
    int_spectrum = sum(mini_cube[:,i,j] for i in range(mini_cube.shape[1]) for j in range(mini_cube.shape[2]))
    int_error = np.sqrt(sum((mini_err[:,i,j])**2 for i in range(mini_cube.shape[1]) for j in range(mini_cube.shape[2])))
    return int_spectrum,int_error

def SII_fix_par(obj):
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (vel_sigma_SII6731,vel_sigma_SII6731_br) = (central_tab.field('vel_sigma_OIII')[0],central_tab.field('vel_sigma_OIII_br')[0])
    (vel_SII6731,vel_SII6731_br) = (central_tab.field('vel_OIII')[0],central_tab.field('vel_OIII_br')[0])
    
    fixed_param = [vel_SII6731,vel_sigma_SII6731,vel_SII6731_br,vel_sigma_SII6731_br]
    return fixed_param

def SII_test_par(obj):
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (vel_sigma_SII6731_br) = (central_tab.field('vel_sigma_OIII_br')[0])
    (vel_SII6731,vel_SII6731_br) = (central_tab.field('vel_OIII')[0],central_tab.field('vel_OIII_br')[0])
    vel_off = vel_SII6731 - vel_SII6731_br
    fixed_param = [vel_SII6731,vel_SII6731_br,vel_sigma_SII6731_br,vel_off]
    return fixed_param

def central_pixel(obj):
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (x_0,y_0) = (central_tab.field('central_x')[0],central_tab.field('central_y')[0])
    return x_0,y_0

def artificial_brightest_pixel(wo_cube):
    (y0,x0) = ndimage.measurements.maximum_position(wo_cube[0,:,:])
    return x0,y0

def artifical_fixed_param(obj,x1,y1):
    hdu = fits.open('%s_x%sy%s_artifica_center_fit.fits'%(obj,x1,y1))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    vel_OIII = central_tab.field('vel_OIII')[0]
    vel_sigma_OIII = central_tab.field('vel_sigma_OIII')[0]
    vel_OIII_br = central_tab.field('vel_OIII_br')[0]
    vel_sigma_OIII_br = central_tab.field('vel_sigma_OIII_br')[0]
    vel_Hb1 = central_tab.field('vel_Hb1')[0]
    vel_Hb2 = central_tab.field('vel_Hb2')[0]
    vel_sigma_Hb1 = central_tab.field('vel_sigma_Hb1')[0]
    vel_sigma_Hb2 = central_tab.field('vel_sigma_Hb2')[0]
    fixed_param = [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2]  
    return fixed_param

def electron_density_wing(obj):
    hdu = fits.open('%s_central_fitSII.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    n_e = central_tab.field('n_e_wing')[0]
    return n_e

def outflow_radius(obj):
    hdu = fits.open('%s_center_table.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    d = central_tab.field('off_wing_parsec')[0]
    return d
    
def Hb_OIII_wing_ratio(obj):
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    (Hb_br,OIII_br) = (central_tab.field('amp_Hb_br')[0],central_tab.field('amp_OIII5007_br')[0])
    ratio = Hb_br/OIII_br
    return ratio

def OIII_br_lum(obj):
    hdu = fits.open('%s_compare_flux_table.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    log_OIII_wing = central_tab.field('log_L_OIII_br_moffat')[0]
    L_OIII = 10**log_OIII_wing 
    return L_OIII
    
def flux_ratio(amp_SII6716_core,amp_SII6731_core,amp_SII6716_wing,amp_SII6731_wing):
    ratio_core = (amp_SII6716_core/amp_SII6731_core)
    ratio_wing = (amp_SII6716_wing/amp_SII6731_wing)
    return ratio_core,ratio_wing 

def electron_density(ratio_core,ratio_wing):
    S2 = pn.Atom('S',2)
    Ne_core = S2.getTemDen(int_ratio=ratio_core,tem=1e4,wave1=6717,wave2=6731)
    Ne_wing = S2.getTemDen(int_ratio=ratio_wing,tem=1e4,wave1=6717,wave2=6731)
    return Ne_core, Ne_wing
    
def SII_plot(obj,wo_wave,data,error,z,popt_full_fit,destination_path_cube):
    (amp_SII6716_fit,amp_SII6731_fit,vel_SII6731_fit,vel_sigma_SII6731_fit,amp_SII6716_br_fit,amp_SII6731_br_fit,vel_SII6731_br_fit,vel_sigma_SII6731_br_fit,m_fit,n_fit,c_fit) = popt_full_fit
    offset_fit=0.0
    k = 1+z
    select = (wo_wave>6680*k) & (wo_wave < 6750*k)
    
    fit = test_gauss_SII(popt_full_fit,wo_wave[select]+offset_fit,data[select],error[select])*(error[select])+data[select]
    residual = data[select] - fit
    
    fig = plt.gcf()
    fig.set_size_inches(10, 6)    
    plt.plot(wo_wave[select],data[select],color='gray',label='Data',ls='steps-mid',lw = 3)
    #plt.fill_between(wo_wave[select],data[select] - error[select],data[select] + error[select],color = 'gray')
    plt.plot(wo_wave[select],fit,'r-',label='Model')
    plt.plot(wo_wave[select], SII_doublet_gauss(wo_wave[select]+offset_fit,amp_SII6716_fit,amp_SII6731_fit,vel_SII6731_fit,vel_sigma_SII6731_fit),'g--',label='Core')
    plt.plot(wo_wave[select], SII_doublet_gauss(wo_wave[select]+offset_fit,amp_SII6716_br_fit,amp_SII6731_br_fit,vel_SII6731_br_fit,vel_sigma_SII6731_br_fit),'b:',label='Wing')
    
    plt.title('%s integrated [SII] complex modelling'%(obj))
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(r"Flux Density ($\times 10^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$) ")
    plt.legend()
    fig.savefig('%s/%s_[SII] model.png'%(destination_path_cube,obj),dpi = 600,overwrite=True)
    plt.show()     
    
    fig = plt.gcf()
    fig.set_size_inches(10, 2)  
    plt.plot(wo_wave[select],residual,color='black',label='residual',drawstyle='steps-mid')
    plt.axhline(y=0.0, color='r', linestyle='--')
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(r"Flux Density ($\times 10^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$) ")
    plt.title('%s integrated [SII] residual'%(obj))
    fig.savefig('%s/%s_[SII] residual.png'%(destination_path_cube,obj),dpi = 600,overwrite=True)
    plt.show()

    #continuum_qua(wo_wave[select],m_fit,n_fit,c_fit) +

def plot_coupled_kin(obj,wo_wave,data,error,z,popt_coupled_fit,destination_path_cube="/home/mainak/Downloads/PMAS_plots"):
    (amp_Ha_core,amp_NII6583_core,amp_SII6716_core,amp_SII6731_core,vel_core,vel_sigma_core,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,m,n,c)= popt_coupled_fit  
    k = 1+z
    select = (wo_wave>6350*k) & (wo_wave<6850*k) 
    fit = complex_gauss_coupled(popt_coupled_fit,wo_wave[select],data[select],error[select])*(error[select])+data[select]
    residual = data[select] - fit
      
    fig = plt.gcf()
    fig.set_size_inches(10, 6)    
    plt.plot(wo_wave[select],data[select],color='gray',label='Data',drawstyle='steps-mid',lw = 3)
    #plt.fill_between(wo_wave[select],data[select] - error[select],data[select] + error[select],color = 'gray')
    plt.plot(wo_wave[select],fit,'r-',label='Model')
    plt.plot(wo_wave[select],nlr_gauss_coupled(wo_wave[select],amp_Ha_core,amp_NII6583_core,amp_SII6716_core,amp_SII6731_core,vel_core,vel_sigma_core),color='blue',linestyle='-.',label='Core')
    plt.plot(wo_wave[select],nlr_gauss_coupled(wo_wave[select],amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing),'k--',label='Wing')
    plt.plot(wo_wave[select],blr_gauss(wo_wave[select],amp_Ha_blr1,vel_blr1,vel_sigma_blr1)+ blr_gauss(wo_wave[select],amp_Ha_blr2,vel_blr2,vel_sigma_blr2)+ blr_gauss(wo_wave[select],amp_Ha_blr3,vel_blr3,vel_sigma_blr3),'g:',label='blr')
    plt.title('%s coupled H-alpha+[NII]+[SII] complex modelling'%(obj))
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(r"Flux Density ($\times 10^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$) ")
    plt.legend()
    fig.savefig('%s/%s_H-alpha+[NII]+[SII] model.png'%(destination_path_cube,obj),dpi = 600,overwrite=True)
    plt.show()     
    
    fig = plt.gcf()
    fig.set_size_inches(10, 2)  
    plt.plot(wo_wave[select],residual,color='black',label='residual',drawstyle='steps-mid')
    plt.axhline(y=0.0, color='r', linestyle='--')
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(r"Flux Density ($\times 10^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$) ")
    plt.title('%s coupled H-alpha+[NII]+[SII]  residual'%(obj))
    fig.savefig('%s/%s_H-alpha+[NII]+[SII] residual.png'%(destination_path_cube,obj),dpi = 600,overwrite=True)
    plt.show()
 
    
def plot_decoupled_kin(obj,wo_wave,data,error,z,popt_decoupled_fit):
    (amp_Ha_core,amp_NII6583_core,vel_Ha_core,vel_sigma_Ha_core,amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,vel_sigma_SII6731_core,amp_Ha_wing,amp_NII6583_wing,vel_Ha_wing,vel_sigma_Ha_wing,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,vel_sigma_SII6731_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,m,n,c)= popt_decoupled_fit
    k = 1+z
    select = (wo_wave>6350*k) & (wo_wave<6850*k)   
    fit = complex_gauss_decoupled(popt_decoupled_fit,wo_wave[select],data[select],error[select])*(error[select])+data[select]
    residual = data[select] - fit
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.plot(wo_wave[select],data[select],label='data',color='black')
    plt.fill_between(wo_wave[select],data[select]-error[select],data[select]+error[select])
    
    plt.plot(wo_wave[select],fit,'r-',label='fit')
    plt.plot(wo_wave[select],residual,label='residual')  
    plt.plot(wo_wave[select],nlr_gauss_decoupled(wo_wave[select],amp_Ha_core,amp_NII6583_core,vel_Ha_core,vel_sigma_Ha_core),label='Ha + [NII] core',color='purple')
    plt.plot(wo_wave[select],SII_doublet_gauss(wo_wave[select],amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,vel_sigma_SII6731_core),label='SII core',color='orange')
    plt.plot(wo_wave[select],nlr_gauss_decoupled(wo_wave[select],amp_Ha_wing,amp_NII6583_wing,vel_Ha_wing,vel_sigma_Ha_wing),label='Ha + [NII] wing',color='brown')
    plt.plot(wo_wave[select],SII_doublet_gauss(wo_wave[select],amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,vel_sigma_SII6731_wing),label='SII wing',color='cyan')
    plt.plot(wo_wave[select],blr_gauss(wo_wave[select],amp_Ha_blr1,vel_blr1,vel_sigma_blr1),'m-',label='blr_1')
    plt.plot(wo_wave[select],blr_gauss(wo_wave[select],amp_Ha_blr2,vel_blr2,vel_sigma_blr2),'g-',label='blr_2')
    plt.plot(wo_wave[select],blr_gauss(wo_wave[select],amp_Ha_blr3,vel_blr3,vel_sigma_blr3),'b-',label='blr_3')

    plt.title('%s_brightest_pixel_fit'%(obj))
    plt.xlabel(r"Wavelength ($\AA$)")
    plt.ylabel(r"Flux Density ($\times 10^{-16}$ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$) ")
    plt.legend()
    plt.show()     
    
def agn_loc(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/PMAS"):
    hdu = fits.open('%s/%s/%s_AGNpix.fits'%(destination_path_cube,obj,obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    x0 = central_tab.field('x0')[0]
    y0 = central_tab.field('y0')[0]
    return x0,y0  
       
def aperture_size_PMAS(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/PMAS"):
    hdu = fits.open('%s/Flux Maps/%s/13_arcsec_subcube_par_%s.fits'%(destination_path_cube,obj,obj))
    (OIII_nr,OIII_br,Hb1_blr_br,Hb2_blr_br) = (hdu[2].data,hdu[3].data,hdu[5].data,hdu[6].data)
    box_size = np.shape(OIII_nr)[0]
    return box_size
    
def moffat_centers(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/PMAS"):
    hdu = fits.open('%s/%s/13_arcsec_moffat_table_%s.fits'%(destination_path_cube,obj,obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    (Hb_x,Hb_y) = (central_tab.field('x0_Hb_Blr')[0],central_tab.field('y0_Hb_Blr')[0])
    (OIII_br_x,OIII_br_y) = (central_tab.field('x0_OIII_br')[0],central_tab.field('y0_OIII_br')[0])
    (OIII_nr_x,OIII_nr_y) = (central_tab.field('x0_OIII_nr')[0],central_tab.field('y0_OIII_nr')[0])
    return Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y  
    
def agn_location(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/PMAS"):
    hdu = fits.open('%s/%s/%s_AGNpix.fits'%(destination_path_cube,obj,obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    x0 = central_tab.field('x0')[0]
    y0 = central_tab.field('y0')[0]
    return x0,y0 
