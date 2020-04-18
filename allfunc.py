import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from numpy import exp
from astropy.modeling import models, fitting
from scipy import ndimage


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

def flux_data_and_error(obj,x1,y1,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    hdu = fits.open('%s/%s/Systematic Error/x%sy%s_artificial_fluxmap_%s.fits'%(destination_path_cube,obj,x1,y1,obj))
    (OIII_nr,OIII_br,Hb1_blr_br,Hb2_blr_br) = (hdu[2].data,hdu[3].data,hdu[5].data,hdu[6].data)
    (amp_Hb1_blr_br,amp_Hb2_blr_br) = (np.max(Hb1_blr_br),np.max(Hb2_blr_br))
    hdu.close()
    hdu = fits.open('%s/%s/Systematic Error/x%sy%s_artificial_fluxerrormap_%s.fits'%(destination_path_cube,obj,x1,y1,obj))
    (OIII_nr_err,OIII_br_err,Hb1_blr_br_err,Hb2_blr_br_err) = (hdu[2].data,hdu[3].data,hdu[5].data,hdu[6].data)
    if amp_Hb1_blr_br > amp_Hb2_blr_br:
        (Hb_blr_br,Hb_blr_br_err) = (Hb1_blr_br,Hb1_blr_br_err)
    else:
        (Hb_blr_br,Hb_blr_br_err) = (Hb2_blr_br,Hb2_blr_br_err)
    return Hb_blr_br,OIII_br,OIII_nr,Hb_blr_br_err,OIII_br_err,OIII_nr_err
   
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

def gauss_recons(wave,amplitude,vel,vel_sigma,rest_wave):
    line = (amplitude)*exp(-(wave-(rest_wave*(1+redshift(vel))))**2/(2*(line_width_recons(vel_sigma, rest_wave))**2))
    return line

def double_BLR(wave,amp_Hb1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,vel_Hb2,vel_sigma_Hb2):
    Hb1 = gauss_recons(wave,amp_Hb1,vel_Hb1,vel_sigma_Hb1,4861.33)
    Hb2 = gauss_recons(wave,amp_Hb2,vel_Hb2,vel_sigma_Hb2,4861.33)
    return Hb1+Hb2

    # Here we couple the HB and OIII doublet together using the gaussian function defined before
def Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel,vel_sigma):
    Hb = gauss(wave,amp_Hb,vel,vel_sigma,4861.33,2.8)
    OIII_4959 = (0.33)*gauss(wave,amp_OIII5007,vel,vel_sigma,4958.9,2.8)
    OIII_5007 = gauss(wave,amp_OIII5007,vel,vel_sigma,5006.8,2.8)
    return Hb + OIII_4959 + OIII_5007

def Hb_O3_gauss_model(wave,amp_Hb,amp_OIII5007,vel,vel_sigma):
    Hb = gauss_recons(wave,amp_Hb,vel,vel_sigma,4861.33)
    OIII_4959 = (0.33)*gauss_recons(wave,amp_OIII5007,vel,vel_sigma,4958.9)
    OIII_5007 = gauss_recons(wave,amp_OIII5007,vel,vel_sigma,5006.8)
    return Hb + OIII_4959 + OIII_5007

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

def Hb_Fe_doublet_gauss_model(wave,amp_Hb,amp_Fe5018,vel,vel_sigma):
    Hb = gauss_recons(wave,amp_Hb,vel,vel_sigma,4861.33)
    Fe_4923 = 0.81*gauss_recons(wave,amp_Fe5018,vel,vel_sigma,4923)
    Fe_5018 = gauss_recons(wave,amp_Fe5018,vel,vel_sigma,5018)
    return Hb+Fe_4923+Fe_5018

def OIII_wo_cont(wave,amp_OIII5007,amp_OIII5007_br,vel,vel_sigma,vel_br,vel_sigma_br):  
    OIII_5007_core = gauss_recons(wave,amp_OIII5007,vel,vel_sigma,5006.8)
    OIII_5007_wing = gauss_recons(wave,amp_OIII5007_br,vel_br,vel_sigma_br,5006.8)
    return OIII_5007_core + OIII_5007_wing
 
def SII_doublet_gauss(wave,amp_SII6716,amp_SII6731,vel,vel_sigma):
    SII_6716 = gauss(wave,amp_SII6716,vel,vel_sigma,6716.44,2.5)
    SII_6731 = gauss(wave,amp_SII6731,vel,vel_sigma,6730.82,2.5)
    return SII_6716+SII_6731

def nlr_gauss(wave,amp_Ha,amp_NII6583,amp_SII6716,amp_SII6731,vel,vel_sigma):
    Ha = gauss(wave,amp_Ha,vel,vel_sigma,6562.85,2.5)
    NII_6548 = 0.33*gauss(wave,amp_NII6583,vel,vel_sigma,6548.05,2.5)
    NII_6583 = gauss(wave,amp_NII6583,vel,vel_sigma,6583.45,2.5)
    SII_6716 = gauss(wave,amp_SII6716,vel,vel_sigma,6716.44,2.5)
    SII_6731 = gauss(wave,amp_SII6731,vel,vel_sigma,6730.82,2.5)
    return Ha + NII_6548 + NII_6583 + SII_6716 + SII_6731

def blr_gauss(wave,amp_Ha,vel,vel_sigma):
    Ha = gauss(wave,amp_Ha,vel,vel_sigma,6562.85,2.5)
    return Ha
    
def complex_gauss(p,wave,data,error):
    (amp_Ha_core1,amp_NII6583_core1,amp_SII6716_core1,amp_SII6731_core1,vel_core1,vel_sigma_core1,amp_Ha_core2,amp_NII6583_core2,amp_SII6716_core2,amp_SII6731_core2,vel_core2,vel_sigma_core2,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,amp_Ha_blr4,vel_blr4,vel_sigma_blr4,m,c)= p
    nlr_core1 = nlr_gauss(wave,amp_Ha_core1,amp_NII6583_core1,amp_SII6716_core1,amp_SII6731_core1,vel_core1,vel_sigma_core1)
    nlr_core2 = nlr_gauss(wave,amp_Ha_core2,amp_NII6583_core2,amp_SII6716_core2,amp_SII6731_core2,vel_core2,vel_sigma_core2)
    #nlr_wing = nlr_gauss(wave,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing)
    nlr_wing = 0
    blr1 = blr_gauss(wave,amp_Ha_blr1,vel_blr1,vel_sigma_blr1)
    #blr2 = 0
    blr2 = blr_gauss(wave,amp_Ha_blr2,vel_blr2,vel_sigma_blr2)
    blr3 = blr_gauss(wave,amp_Ha_blr3,vel_blr3,vel_sigma_blr3)    
    blr4 = blr_gauss(wave,amp_Ha_blr4,vel_blr4,vel_sigma_blr4)
    #blr3 = 0
    cont = (wave/1000.0)*m+c
    return (nlr_core1+nlr_core2+nlr_wing+blr1+blr2+blr3+cont-data)/error
    
def complex_gauss1(p,wave,data,error):
    (amp_Ha_core1,amp_NII6583_core,amp_SII6716_core,amp_SII6731_core,vel_core1,vel_sigma_core,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing,amp_Ha_blr1,vel_blr1,vel_sigma_blr1,amp_Ha_blr2,vel_blr2,vel_sigma_blr2,amp_Ha_blr3,vel_blr3,vel_sigma_blr3,m,c)= p
    nlr_core1 = nlr_gauss(wave,amp_Ha_core1,amp_NII6583_core1,amp_SII6716_core1,amp_SII6731_core1,vel_core1,vel_sigma_core1)
    nlr_wing = nlr_gauss(wave,amp_Ha_wing,amp_NII6583_wing,amp_SII6716_wing,amp_SII6731_wing,vel_wing,vel_sigma_wing)
    blr1 = blr_gauss(wave,amp_Ha_blr1,vel_blr1,vel_sigma_blr1)
    #blr2 = 0
    blr2 = blr_gauss(wave,amp_Ha_blr2,vel_blr2,vel_sigma_blr2)
    #blr3 = blr_gauss(wave,amp_Ha_blr3,vel_blr3,vel_sigma_blr3)
    blr3 = 0
    cont = (wave/1000.0)*m+c
    return (nlr_core1+nlr_core2+nlr_wing+blr1+blr2+blr3+cont-data)/error

def full_gauss_SII(wave,amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br,m,c):
    narrow_SII = SII_doublet_gauss(wave,amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731)
    broad_SII = SII_doublet_gauss(wave,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br)
    cont = (wave/1000.0)*m+c
    return narrow_SII+ broad_SII+cont

def test_gauss_SII(p,wave,data,error):
    (amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br,m,c)=p
    narrow_SII = SII_doublet_gauss(wave,amp_SII6716,amp_SII6731,vel_SII6731,vel_sigma_SII6731)
    broad_SII = SII_doublet_gauss(wave,amp_SII6716_br,amp_SII6731_br,vel_SII6731_br,vel_sigma_SII6731_br)
    cont = (wave/1000.0)*m+c
    return (narrow_SII+ broad_SII+cont-data)/error


def full_gauss1(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = 0 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error

def full_gauss1_lsq(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss_model(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss_model(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss_model(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = 0 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error


def full_gauss1_model(p,wave):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = 0 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont)

def full_gauss2(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error


def full_gauss2_lsq(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss_model(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss_model(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss_model(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss_model(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error

def full_gauss2_model(p,wave):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont)

def full_gauss3(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,amp_Hb3,amp_Fe5018_3,vel_Hb3,vel_sigma_Hb3,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    Hb_broad3 = Hb_Fe_doublet_gauss(wave,amp_Hb3,amp_Fe5018_3,vel_Hb3,vel_sigma_Hb3) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+Hb_broad3+cont-data)/error


def continuum(wave,m,c):
    slope = (wave/1000.0)*m
    const = c
    return slope + const


def full_gauss1_fixkin(p,wave,data,error,fixed_param):
    (amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c) = p 
    [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2] = fixed_param 
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1)
    Hb_broad2 = 0 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error

def full_gauss1_fixkin_lsq(p,wave,data,error,fixed_param):
    (amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c) = p 
    [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2] = fixed_param 
    narrow_OIII = Hb_O3_gauss_model(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss_model(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss_model(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1)
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

def full_gauss2_fixkin_lsq(p,wave,data,error,fixed_param):
    (amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c) = p 
    [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2] = fixed_param 
    narrow_OIII = Hb_O3_gauss_model(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss_model(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss_model(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss_model(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error

def difference_in_wavelength_dimension(orig_cube,cont_cube):
    difference_in_wavelength_dimension_source = np.shape(orig_cube)[0] - np.shape(cont_cube)[0]
    return difference_in_wavelength_dimension_source

def brightest_pixel(QSO_cube,wo_cube,wo_wave,z):
    k = 1 + z
    QSO_slice = QSO_cube[0,:,:]
    [guess_y,guess_x] = ndimage.measurements.maximum_position(QSO_slice)
    test_cube = wo_cube[:,guess_y-5:guess_y+5,guess_x-5:guess_x+5]
    select = (wo_wave >5006*k) & (wo_wave<5009*k) 
    test_cube = test_cube[select]
    (y0,x0) = ndimage.measurements.maximum_position(test_cube[0,:,:])
    (brightest_pixel_y,brightest_pixel_x) = (y0+guess_y-5,x0+guess_x-5)
    return brightest_pixel_x,brightest_pixel_y
    
def fixed_parameters(obj):
    hdu = fits.open('%s_central_fit.fits'%(obj))
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
    
def light_weighted_centroid(obj):
    hdu = fits.open('subcube_par_%s.fits'%(obj))
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
    
def sampling_size(cont_cube):
    single_dimension_shape = np.shape(cont_cube)[1] 
    if single_dimension_shape > 250:
        sampling_size = 0.2
    else:
        sampling_size = 0.4
    return sampling_size

def centers(obj):
    hdu = fits.open('moffat_table_%s.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    (Hb_x,Hb_y) = (central_tab.field('x0_Hb_Blr')[0],central_tab.field('y0_Hb_Blr')[0])
    (OIII_br_x,OIII_br_y) = (central_tab.field('x0_OIII_br')[0],central_tab.field('y0_OIII_br')[0])
    (OIII_nr_x,OIII_nr_y) = (central_tab.field('x0_OIII_nr')[0],central_tab.field('y0_OIII_nr')[0])
    return Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y  

def offset(Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y,muse_sampling_size):
    offset_OIII_br_pixel= [(OIII_br_x - Hb_x),(OIII_br_y - Hb_y)]
    offset_OIII_nr_pixel= [(OIII_nr_x - Hb_x),(OIII_nr_y - Hb_y)]
    offset_OIII_br_arcsec = np.asarray(offset_OIII_br_pixel)*muse_sampling_size 
    offset_OIII_nr_arcsec = np.asarray(offset_OIII_nr_pixel)*muse_sampling_size 
    return offset_OIII_br_pixel,offset_OIII_nr_pixel,offset_OIII_br_arcsec,offset_OIII_nr_arcsec
    
def ranges(Hb_x,Hb_y,muse_sampling_size,asymmetry=False):
    if asymmetry:
        size = 14
    else:
        size = 15
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
    
    (x_0,y_0) = (int(central_tab.field('central_x')[0]),int(central_tab.field('central_y')[0]))
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

def electron_density(obj):
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
    
def model(amp,x0,y0,A,B,C,alpha,muse_sampling_size):
    if muse_sampling_size == 0.4:
        y, x = np.mgrid[:7, :7]
    else:
        y, x = np.mgrid[:15, :15] 
    #p = models.Moffat2D(amp_Hb_blr,x0_Hb_Blr,y0_Hb_Blr,gamma,alpha)
    psf_data = amp*((1.0+A*(x-x0)**2+B*(y-y0)**2+C*(x-x0)*(y-y0))**(-alpha))
    amp_psf = np.max(psf_data)
    psf = (psf_data/amp_psf)
    return psf

def err_spec(mini_err,psf):
    error = mini_err*np.sqrt(psf)
    mu = np.mean(error)
    sigma=np.std(error)
    size = np.shape(error)
    noise = np.random.normal(mu,sigma,size)
    return noise,error

def pure_source(psf,fit):
    cube = (psf[np.newaxis,:,:]*fit[:,np.newaxis,np.newaxis])
    return cube

def scale_SN(flux,ref_flux,ref_SN):
    SN = np.sqrt(flux/ref_flux)*ref_SN
    return SN

def artf_source(psf,fit,ref_flux,ref_SN):
    noiseless = (psf[np.newaxis,:,:]*fit[:,np.newaxis,np.newaxis]) 
    SN = scale_SN(noiseless,ref_flux,ref_SN)
    err = (noiseless/SN)
    cube = np.random.normal(noiseless,err)
    return cube,err

def artf_cube(cube,err,filename):
    hdus=[]
    hdus.append(fits.PrimaryHDU(cube))
    hdus.append(fits.ImageHDU(err,name='error'))
    hdu = fits.HDUList(hdus)
    hdu.writeto(filename,overwrite=True)
    
def new_cube_cor(x_act,y_act,muse_sampling_size):
    if muse_sampling_size==0.4:
        (xl,yl,xr,yr)=(x_act-3,y_act-3,x_act+4,y_act+4)
    else:
        (xl,yl,xr,yr)=(x_act-7,y_act-7,x_act+8,y_act+8)
    return xl,xr,yl,yr
    
def new_minicube(z,wo_wave,wo_cube,wo_err,cube,err,x_l,x_r,y_l,y_r):
    k = 1+z
    select = (wo_wave>4750*k) & (wo_wave < 5090*k)
    (new_wo_cube,new_wo_err) = (wo_cube[select],wo_err[select])
    data = new_wo_cube[:,y_l:y_r,x_l:x_r] + cube
    error = np.sqrt(np.square(new_wo_err[:,y_l:y_r,x_l:x_r]) + np.square(err))
    wave = wo_wave[select]
    return data,error,wave

def test(z,wo_cube,wo_wave,brightest_pixel_x,brightest_pixel_y,x1,y1):
    k = 1+z
    select = (wo_wave>4750*k) & (wo_wave<6800*k)
    data = wo_cube[select]
    amp = np.max(data[:,brightest_pixel_y,brightest_pixel_x])
    cordlist =[]
    for i in range(16):
        if np.max(data[:,y1+i,x1]) >= 0.005*amp or np.max(data[:,y1+i,x1]) == 0.0 or np.max(data[:,y1,x1+i])==0.0 or np.max(data[:,y1,x1+i]) >= 0.005*amp or np.max(data[:,y1+i,x1+i]) == 0.0 or np.max(data[:,y1+i,x1+i]) >= 0.005*amp: 
            (x,y) = ('no','no')
        else:
            (x,y) = (x1,y1)
    cordlist.append([x,y])
    print (cordlist)
    return cordlist
   
    
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

def arb_brightest_pixel(arb_data):
    (y0,x0) = ndimage.measurements.maximum_position(arb_data[0,:,:])
    return x0,y0
      
def arb_fit_and_residual(wave,data,error,popt,broad2=False):
    if broad2:
        full_gauss = full_gauss2
    else:
        full_gauss = full_gauss1
    fit = full_gauss(popt,wave,data,error)*error+data
    residual = data - fit
    return fit,residual

def arb_sampling_size(wo_cube):
    single_dimension_shape = np.shape(wo_cube)[1] 
    if single_dimension_shape > 250:
        sampling_size = 0.2
    else:
        sampling_size = 0.4
    return sampling_size

def shape(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    hdu = fits.open('%s/%s/%s.wo_absorption.fits'%(destination_path_cube,obj,obj)) 
    data = hdu[0].data
    single_dimension_shape = np.shape(data)[1] 
    if single_dimension_shape > 250:
        box_length = 15
    else:
        box_length = 7
    
    (y_shape,x_shape) = np.shape(data[0,:,:])
    (x_min,y_min) = (box_length,box_length)
    (x_max,y_max) = (x_shape-box_length-1,y_shape-box_length-1)
    return x_min,y_min,x_max,y_max

def central_spectrum(wo_wave,popt,z,broad2=False):
    k = 1+z
    select = (wo_wave>4750*k) & (wo_wave < 5090*k)
    if broad2:
        full_gauss = full_gauss2_model
    else:
        full_gauss = full_gauss1_model
    fit = full_gauss(popt,wo_wave[select])
    return fit
    
def ellip_moffat_par(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    hdu = fits.open('%s/%s/9_arcsec_moffat_table_%s.fits'%(destination_path_cube,obj,obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    amp_Hb_blr = central_tab.field('amp_Hb_blr')[0]
    x0_Hb_Blr = central_tab.field('x0_Hb_blr')[0]
    y0_Hb_Blr = central_tab.field('y0_Hb_blr')[0]
    A = central_tab.field('A')[0]
    B = central_tab.field('B')[0]
    C = central_tab.field('C')[0]
    alpha = central_tab.field('alpha')[0]
    return amp_Hb_blr,x0_Hb_Blr,y0_Hb_Blr,A,B,C,alpha

def moffat_O3_par(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    hdu = fits.open('%s/%s/9_arcsec_moffat_table_%s.fits'%(destination_path_cube,obj,obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    amp_OIII_br = central_tab.field('amp_OIII_br')[0]
    x0_OIII_br = central_tab.field('x0_OIII_br')[0]
    y0_OIII_br = central_tab.field('y0_OIII_br')[0]
    amp_OIII_nr = central_tab.field('amp_OIII_nr')[0]
    x0_OIII_nr = central_tab.field('x0_OIII_nr')[0]
    y0_OIII_nr = central_tab.field('y0_OIII_nr')[0]
    return amp_OIII_br,x0_OIII_br,y0_OIII_br,amp_OIII_nr,x0_OIII_nr,y0_OIII_nr

def full_par_central(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    hdu = fits.open('%s/%s/%s_central_fit.fits'%(destination_path_cube,obj,obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (amp_Hb,amp_Hb_br) = (central_tab.field('amp_Hb')[0],central_tab.field('amp_Hb_br')[0])
    (amp_OIII5007,amp_OIII5007_br) = (central_tab.field('amp_OIII5007')[0],central_tab.field('amp_OIII5007_br')[0])
    (amp_Hb1,amp_Hb2) = (central_tab.field('amp_Hb1')[0],central_tab.field('amp_Hb2')[0])
    (amp_Fe5018_1,amp_Fe5018_2) = (central_tab.field('amp_Fe5018_1')[0],central_tab.field('amp_Fe5018_2')[0])
    (m,c) = (central_tab.field('m')[0],central_tab.field('c')[0])

    (vel_Hb1,vel_Hb2) = (central_tab.field('vel_Hb1')[0],central_tab.field('vel_Hb2')[0])
    (vel_sigma_Hb1,vel_sigma_Hb2) = (central_tab.field('vel_sigma_Hb1')[0],central_tab.field('vel_sigma_Hb2')[0])
    (vel_sigma_OIII,vel_sigma_OIII_br) = (central_tab.field('vel_sigma_OIII')[0],central_tab.field('vel_sigma_OIII_br')[0])
    (vel_OIII,vel_OIII_br) = (central_tab.field('vel_OIII')[0],central_tab.field('vel_OIII_br')[0])
    hdu.close()
    
    return amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c

def agn_location(obj,destination_path_cube="/home/mainak/Downloads/Outflow_paper1/MUSE"):
    hdu = fits.open('%s/%s/%s_AGNpix.fits'%(destination_path_cube,obj,obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    x0 = central_tab.field('x0')[0]
    y0 = central_tab.field('y0')[0]
    return x0,y0 