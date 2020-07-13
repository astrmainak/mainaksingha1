import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from numpy import exp
from astropy.modeling import models, fitting
from scipy import ndimage     

def redshift(vel):
    return vel/300000.0
    #This function also represent the line dispersion in A through a velocity dispersion in km/s also taking into account 
    # that the spectrograph itself already broadens the emission lines. This way you automatically fit for the intrinsic line dispersion
def line_width(vel_sigma,rest_line,inst_res_fwhm):
    sigma = vel_sigma/(300000.0-vel_sigma)*rest_line
    return np.sqrt(sigma**2+(inst_res_fwhm/2.354)**2)

def line_width_recons(vel_sigma,rest_line,inst_res_fwhm=0):
    sigma = vel_sigma/(300000.0-vel_sigma)*rest_line
    return np.sqrt(sigma**2+(inst_res_fwhm/2.354)**2)

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
    
def OIII_model(wave,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br):
    narrow_OIII = O3_gauss(wave,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = O3_gauss(wave,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    return narrow_OIII + broad_OIII 

def O3_gauss(wave,amp_OIII5007,vel,vel_sigma):
    OIII_4959 = (0.33)*gauss(wave,amp_OIII5007,vel,vel_sigma,4958.9,2.3)
    OIII_5007 = gauss(wave,amp_OIII5007,vel,vel_sigma,5006.8,2.3)
    return OIII_4959 + OIII_5007
    
    # Here we couple the HB and OIII doublet together using the gaussian function defined before
def Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel,vel_sigma):
    Hb = gauss(wave,amp_Hb,vel,vel_sigma,4861.33,2.8)
    OIII_4959 = (0.33)*gauss(wave,amp_OIII5007,vel,vel_sigma,4958.9,2.8)
    OIII_5007 = gauss(wave,amp_OIII5007,vel,vel_sigma,5006.8,2.8)
    return Hb + OIII_4959 + OIII_5007
    
def full_gauss_O3(p,wave,data,error):
    (amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,m,c)=p
    narrow_OIII = O3_gauss(wave,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = O3_gauss(wave,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+cont-data)/error

def full_gauss_Hb_O3(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+cont-data)/error

    # Same as before but fore the Fe doublet
def Fe_doublet_gauss(wave,amp_Fe4923,amp_Fe5018,vel,vel_sigma):
    Fe_4923 = gauss(wave,amp_Fe4923,vel,vel_sigma,4923,2.8)
    Fe_5018 = gauss(wave,amp_Fe5018,vel,vel_sigma,5018,2.8)
    return Fe_4923+Fe_5018

def Hb_Fe_doublet_gauss(wave,amp_Hb,amp_Fe5018,vel,vel_sigma):
    Hb = gauss(wave,amp_Hb,vel,vel_sigma,4861.33,2.948)
    Fe_4923 = 0.81*gauss(wave,amp_Fe5018,vel,vel_sigma,4923,2.8)
    Fe_5018 = gauss(wave,amp_Fe5018,vel,vel_sigma,5018,2.8)
    return Hb+Fe_4923+Fe_5018

def OIII_wo_cont(wave,amp_OIII5007,amp_OIII5007_br,vel,vel_sigma,vel_br,vel_sigma_br):  
    OIII_5007_core = gauss_recons(wave,amp_OIII5007,vel,vel_sigma,5006.8)
    OIII_5007_wing = gauss_recons(wave,amp_OIII5007_br,vel_br,vel_sigma_br,5006.8)
    return OIII_5007_core + OIII_5007_wing
 
def SII_doublet_gauss(wave,amp_SII6716,amp_SII6731,vel,vel_sigma):
    SII_6716 = gauss(wave,amp_SII6716,vel,vel_sigma,6716.44,2.5)
    SII_6731 = gauss(wave,amp_SII6731,vel,vel_sigma,6730.82,2.5)
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
     (amp_Ha_core,amp_NII6583_core,amp_SII6716_core,amp_SII6731_core,vel_core,vel_sigma_core,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,m,c)= p  
     nlr_core = nlr_gauss_coupled(wave,amp_Ha_core,amp_NII6583_core,amp_SII6716_core,amp_SII6731_core,vel_core,vel_sigma_core)
#     nlr_core2 = nlr_gauss(wave,amp_Ha_core2,amp_NII6583_core2,amp_SII6716_core2,amp_SII6731_core2,vel_core2,vel_sigma_core2)
     nlr_wing = nlr_gauss_coupled(wave,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing)
#     nlr_wing = 0
     blr1 = blr_gauss(wave,amp_Ha_blr1,vel_blr1,vel_sigma_blr1)
     #blr2 = 0
     blr2 = blr_gauss(wave,amp_Ha_blr2,vel_blr2,vel_sigma_blr2)
     blr3 = blr_gauss(wave,amp_Ha_blr3,vel_blr3,vel_sigma_blr3)    
#     blr4 = blr_gauss(wave,amp_Ha_blr4,vel_blr4,vel_sigma_blr4)
    # blr3 = 0
     cont = (wave/1000.0)*m+c
     return (nlr_core++nlr_wing+blr1+blr2+blr3+cont-data)/error
#============================================================================== 
    
def complex_gauss_decoupled(p,wave,data,error):
    (amp_Ha_core,amp_NII6583_core,vel_Ha_core,vel_sigma_Ha_core,amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,vel_sigma_SII6731_core,amp_Ha_wing,amp_NII6583_wing,vel_Ha_wing,vel_sigma_Ha_wing,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,vel_sigma_SII6731_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,m,c)= p
    Ha_NII_core = nlr_gauss_decoupled(wave,amp_Ha_core,amp_NII6583_core,vel_Ha_core,vel_sigma_Ha_core)
    SII_core = SII_doublet_gauss(wave,amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,vel_sigma_SII6731_core)
    Ha_NII_wing = nlr_gauss_decoupled(wave,amp_Ha_wing,amp_NII6583_wing,vel_Ha_wing,vel_sigma_Ha_wing)
    SII_wing = SII_doublet_gauss(wave,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,vel_sigma_SII6731_wing)
    blr1 = blr_gauss(wave,amp_Ha_blr1,vel_blr1,vel_sigma_blr1)
    #blr2 = 0
    blr2 = blr_gauss(wave,amp_Ha_blr2,vel_blr2,vel_sigma_blr2)
    blr3 = blr_gauss(wave,amp_Ha_blr3,vel_blr3,vel_sigma_blr3)
    #blr3 = 0
    cont = (wave/1000.0)*m+c
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
    (amp_Ha_core,amp_NII6583_core,vel_Ha_core,amp_SII6716_core,amp_SII6731_core,vel_SII6731_core,amp_Ha_wing,amp_NII6583_wing,vel_Ha_wing,amp_SII6716_wing,amp_SII6731_wing,vel_SII6731_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,m,c)= p
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
    cont = (wave/1000.0)*m+c
    return (Ha_NII_core+SII_core+Ha_NII_wing+SII_wing+blr1+blr2+blr3+cont-data)/error
    
def full_gauss_SII(wave,amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br,m,c):
    narrow_SII = SII_doublet_gauss(wave,amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731)
    broad_SII = SII_doublet_gauss(wave,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br)
    cont = (wave/1000.0)*m+c
    return narrow_SII+ broad_SII+cont

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

def continuum(wave,m,c):
    slope = (wave/1000.0)*m
    const = c
    return slope + const
    
def fixed_parameters(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
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
    
