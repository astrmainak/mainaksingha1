
# coding: utf-8

# In[4]:

#!/usr/bin/env python
# -----------------------------------------------------------------------------
# VORONOI.ACCRETION
# Laura L Watkins [lauralwatkins@gmail.com]
# - converted from IDL code by Michele Cappellari (bin2d_accretion)
# -----------------------------------------------------------------------------
from numpy import *
from matplotlib.pyplot import *
from numpy import *


def accretion(x, y, signal, noise, targetsn, pixelsize=False, quiet=False):
    
    """
    Initial binning -- steps i-v of eq 5.1 of Cappellari & Copin (2003)
    
    INPUTS:
      x        : x coordinates of pixels to bin
      y        : y coordinates of pixels to bin
      signal   : signal associated with each pixel
      noise    : noise (1-sigma error) associated with each pixel
      targetsn : desired signal-to-noise ration in final 2d-binned data
    
    OPTIONS:
      pixelsize : pixel scale of the input data
      quiet     : if set, suppress printed outputs
    """
    
    
    n = x.size
    clas = zeros(x.size, dtype="<i8")   # bin number of each pixel
    good = zeros(x.size, dtype="<i8")   # =1 if bin accepted as good
    
    # for each point, find distance to all other points and select minimum
    # (robust but slow way of determining the pixel size of unbinned data)
    if not pixelsize:
        dx = 1.e30
        for j in range(x.size-1):
            d = (x[j] - x[j+1:])**2 + (y[j] - y[j+1:])**2
            dx = min(d.min(), dx)
        pixelsize = sqrt(dx)
    
    # start from the pixel with highest S/N
    sn = (signal/noise).max()
    currentbin = (signal/noise).argmax()
    
    # rough estimate of the expected final bin number
    # This value is only used to have a feeling of the expected
    # remaining computation time when binning very big dataset.
    wh = where(signal/noise<targetsn)
    npass = size(where(signal/noise >= targetsn))
    maxnum = int(round( (signal[wh]**2/noise[wh]**2).sum()/targetsn**2 ))+npass
    
    # first bin assigned CLAS = 1 -- with N pixels, get at most N bins
    for ind in range(1, n+1):
        
        if not quiet:
            print("  bin: {:} / {:}".format(ind, maxnum))
        
        # to start the current bin is only one pixel
        clas[currentbin] = ind
        
        # centroid of bin
        xbar = x[currentbin]
        ybar = y[currentbin]
        
        while True:
            
            # stop if all pixels are binned
            unbinned = where(clas == 0)[0]
            if unbinned.size == 0: break
            
            # find unbinned pixel closest to centroid of current bin
            dist = (x[unbinned]-xbar)**2 + (y[unbinned]-ybar)**2
            mindist = dist.min()
            k = dist.argmin()
            
            # find the distance from the closest pixel to the current bin
            mindist = ((x[currentbin]-x[unbinned[k]])**2                 + (y[currentbin]-y[unbinned[k]])**2).min()
            
            # estimate roundness of bin with candidate pixel added
            nextbin = append(currentbin, unbinned[k])
            roundness = bin_roundness(x[nextbin], y[nextbin], pixelsize)
            
            # compute sn of bin with candidate pixel added
            snold = sn
            sn = signal[nextbin].sum()/sqrt((noise[nextbin]**2).sum())
            
            # Test whether the CANDIDATE pixel is connected to the
            # current bin, whether the POSSIBLE new bin is round enough
            # and whether the resulting S/N would get closer to targetsn
            if sqrt(mindist) > 1.2*pixelsize or roundness > 0.3                 or abs(sn-targetsn) > abs(snold-targetsn):
                if (snold > 0.8*targetsn):
                    good[currentbin] = 1
                break
            
            # if all the above tests are negative then accept the CANDIDATE
            # pixel, add it to the current bin, and continue accreting pixels
            clas[unbinned[k]] = ind
            currentbin = nextbin
            
            # update the centroid of the current bin
            xbar = x[currentbin].mean()
            ybar = y[currentbin].mean()
        
        # get the centroid of all the binned pixels
        binned = where(clas != 0)[0]
        unbinned = where(clas == 0)[0]
        
        # stop if all pixels are binned
        if unbinned.size == 0: break
        xbar = x[binned].mean()
        ybar = y[binned].mean()
        
        # find the closest unbinned pixel to the centroid of all
        # the binned pixels, and start a new bin from that pixel
        k = ((x[unbinned]-xbar)**2 + (y[unbinned]-ybar)**2).argmin()
        currentbin = unbinned[k]    # the bin is initially made of one pixel
        sn = signal[currentbin] / noise[currentbin]
    
    # set to zero all bins that did not reach the target S/N
    clas = clas*good
    
    return clas



def bin2d(x, y, signal, noise, targetsn, cvt=True, wvt=False, quiet=True,
    graphs=True):
    
    """
    This is the main program that has to be called from external programs.
    It simply calls in sequence the different steps of the algorithms
    and optionally plots the results at the end of the calculation.
    
    INPUTS
      x        : x-coordinates of pixels
      y        : y-coordinates of pixels
      signal   : signal in pixels
      noise    : noise in pixels
      targetsn : target S/N required
    
    OPTIONS
      cvt      : use Modified-Lloyd algorithm [default True]
      wvt      : use additional modification by Diehl & Statler [default False]
      quiet    : supress output [default True]
      graphs   : show results graphically [default True]
    """
    
    
    npix = x.size
    if y.size != x.size or signal.size != x.size or noise.size != x.size:
        print("ERROR: input vectors (x, y, signal, noise) must have same size")
        return
    if any(noise < 0):
        print("ERROR: noise cannot be negative")
        return
    
    # prevent division by zero for pixels with signal=0 and
    # noise=sqrt(signal)=0 as can happen with X-ray data
    noise[noise==0] = noise[noise>0].min() * 1e-9
    
    # Perform basic tests to catch common input errors
    if signal.sum()/sqrt((noise**2).sum()) < targetsn:
        print("Not enough S/N in the whole set of pixels. "             + "Many pixels may have noise but virtually no signal. "             + "They should not be included in the set to bin, "             + "or the pixels should be optimally weighted."             + "See Cappellari & Copin (2003, Sec.2.1) and README file.")
        return
    if (signal/noise).min() > targetsn:
        print("EXCEPTION: all pixels have enough S/N -- binning not needed")
        return
    
    if not quiet: print("Bin-accretion...")
    clas = accretion(x, y, signal, noise, targetsn, quiet=quiet)
    if not quiet: print("{:} initial bins\n".format(clas.max()))
    
    if not quiet: print("Reassign bad bins...")
    xnode, ynode = reassign_bad_bins(x, y, signal, noise, targetsn, clas)
    if not quiet: print("{:} good bins\n".format(xnode.size))
    
    if cvt:
        if not quiet: print("Modified Lloyd algorithm...")
        scale, iters = cvt_equal_mass(x, y, signal, noise, xnode, ynode,
            quiet=quiet, wvt=wvt)
        if not quiet: print("  iterations: {:}".format(iters-1))
    else:
        scale = 1.
    
    if not quiet: print("Recompute bin properties...")
    clas, xbar, ybar, sn, area = bin_quantities(x, y, signal, noise, xnode,
        ynode, scale)
    unb = area==1
    binned = area!=1
    if not quiet: print("Unbinned pixels: {:} / {:}".format(sum(unb), npix))
    fracscat = ((sn[binned]-targetsn)/targetsn*100).std()
    if not quiet: print("Fractional S/N scatter (%):", fracscat)
    
    if graphs:
        
        # set up plotting
        rc("font", family="serif")
        rc("text", usetex=True)
        rc("xtick", labelsize="8")
        rc("ytick", labelsize="8")
        rc("axes", labelsize="10")
        rc("legend", fontsize="9")
        
        # pixel map
        fig = figure(figsize=(4,3))
        fig.subplots_adjust(left=0.13, bottom=0.13, top=0.97, right=0.98)
        rnd = random.rand(xnode.size).argsort()      # randomize bin colors
        scatter(x, y, lw=0, c=rnd[clas])
        plot(xnode, ynode, "k+", ms=2)
        xlim(x.min()-x.ptp()*0.05, x.max()+x.ptp()*0.05)
        ylim(y.min()-y.ptp()*0.05, y.max()+y.ptp()*0.05)
        xlabel("coordinate 1")
        ylabel("coordinate 2")
        show()
        
        # signal-to-noise profile
        fig = figure(figsize=(4,3))
        fig.subplots_adjust(left=0.12, bottom=0.13, top=0.97, right=0.97)
        rad = sqrt(xbar**2 + ybar**2)     # use centroids, NOT generators
        rmin = max(0., rad.min()-rad.ptp()*0.05)
        rmax = rad.max()+rad.ptp()*0.05
        plot([rmin, rmax], ones(2)*targetsn, c="k", lw=2, alpha=0.8)
        scatter(rad[binned], sn[binned], lw=0, c="b", alpha=0.8)
        if unb.size > 0: scatter(rad[unb], sn[unb], lw=0, c="r", alpha=0.8)
        xlim(rmin, rmax)
        ylim(0., sn.max()*1.05)
        xlabel(r"$R_{\rm bin}$")
        ylabel(r"$SN_{\rm bin}$")
        show()
    
    
    return clas, xnode, ynode, sn, area, scale

def bin_quantities(x, y, signal, noise, xnode, ynode, scale):
    
    """
    Recomputes (weighted) voronoi tessellation of the pixels grid to make
    sure that the clas number corresponds to the proper Voronoi generator.
    This is done to take into account possible zero-size Voronoi bins
    in output from the previous CVT (or WVT).
    
    INPUTS
      x      : x-coordinates of pixels
      y      : y-coordinates of pixels
      signal : signal in pixels
      noise  : noise in pixels
      xnode  : x-coordinates of bins
      ynode  : y-coordinates of bins
      scale  : bin scale
    """
    
    clas = np.zeros(x.size, dtype="int")   # will contain bin num of given pixels
    for j in range(x.size):
        clas[j] = (((x[j]-xnode)/scale)**2 + ((y[j]-ynode)/scale)**2).argmin()
    
    # At the end of the computation evaluate the bin luminosity-weighted
    # centroids (xbar,ybar) and the corresponding final S/N of each bin.
    
    area, lim = np.histogram(clas, bins=int(clas.ptp())+1,
        range=(clas.min()-0.5, clas.max()+0.5))
    cent = (lim[:-1]+lim[1:])/2.
    
    xb = np.zeros(xnode.size)
    yb = np.zeros(xnode.size)
    sn = np.zeros(xnode.size)
    for j in range(xnode.size):
        pix = np.where(clas == cent[j])[0]
        xb[j], yb[j] = weighted_centroid(x[pix], y[pix], signal[pix])
        sn[j] = signal[pix].sum()/np.sqrt((noise[pix]**2).sum())
    
    return clas, xb, yb, sn, area

def bin_roundness(x, y, pixelsize):
    
    """
    Computes roundness of a bin -- eq 5 of Cappellari & Copin (2003)
    
    INPUTS
      x         : x-coordinates of pixels in bin
      y         : y-coordinates of pixels in bin
      pixelsize : size of pixels
    """
    
    
    equivalentradius = sqrt(x.size/pi)*pixelsize
    
    # geometric centroid
    xbar = x.mean()
    ybar = y.mean()
    
    maxdistance = sqrt((x-xbar)**2 + (y-ybar)**2).max()
    roundness = maxdistance/equivalentradius - 1.
    
    return roundness

def cvt_equal_mass(x, y, signal, noise, xnode, ynode, quiet=True, wvt=False):
    
    """
    Modified Lloyd algorithm -- section 4.1 of Cappellari & Copin (2003).
    When the keyword wvt is set, the routine includes the modification
    proposed by Diehl & Statler (2006).
    
    INPUTS
      x      : x-coordinates of pixels
      y      : y-coordinates of pixels
      signal : signal in pixels
      noise  : noise in pixels
      xnode  : x-coordinates of bins
      ynode  : y-coordinates of bins
    
    OPTIONS
      quiet  : suppress output [default True]
      wvt    : use modification of Diehl & Statler (2006) [default False]
    """
    
    
    clas = np.zeros(len(signal))   # see beginning of section 4.1 of CC03
    if wvt: dens = np.ones(len(signal))
    else: dens = signal**2/noise**2
    scale = 1                   # start with the same scale length for all bins
    sn = np.zeros(len(xnode))
    
    iters = 1
    diff = 1
    while diff!=0:
        
        xold = xnode.copy()
        yold = ynode.copy()
        
        # computes (weighted) voronoi tessellation of the pixels grid
        for j in range(len(signal)):
            index = (((x[j]-xnode)/scale)**2+((y[j]-ynode)/scale)**2).argmin()
            clas[j] = index
        
        # Computes centroids of the bins, weighted by dens^2.
        # Exponent 2 on the density produces equal-mass Voronoi bins.
        # The geometric centroids are computed if /wvt keyword is set.
        
        area, lim = np.histogram(clas, bins=int(clas.ptp())+1,
            range=(clas.min()-0.5, clas.max()+0.5))
        cent = (lim[:-1]+lim[1:])/2
        
        nonzero = np.where(area>0)[0]       # check for zero-size voronoi bins
        for j in range(len(nonzero)):
            k = nonzero[j]                  # only loop over nonzero bins
            pix = clas==cent[k]
            xnode[k], ynode[k] = weighted_centroid(x[pix],y[pix],dens[pix]**2)
            sn[k] = signal[pix].sum()/np.sqrt((noise[pix]**2).sum())
        
        if wvt: scale = np.sqrt(area/sn)    # eq 4 of Diehl & Statler (2006)
        diff = ((xnode-xold)**2 + (ynode-yold)**2).sum()
        iters = iters + 1
        
        if not quiet:
            print("  iteration: {:}  difference: {:}".format(iters, diff))
    
    # only return the generators of the nonzero voronoi bins
    xnode = xnode[nonzero]
    ynode = ynode[nonzero]
    
    return scale, iters

def reassign_bad_bins(x, y, signal, noise, targetsn, clas):
    
    """
    Reassign bad bins -- steps vi-vii of eq 5.1 of Cappellari & Copin (2003)
    
    INPUTS
      x        : x-coordinates of pixels
      y        : y-coordinates of pixels
      signal   : signal in pixels
      noise    : noise in pixels
      targetsn : target signal/noise required
      clas     : bin number for each pixel
    """
    
    
    # get number of pixels in each bin (clas=0 are unassigned pixels)
    area, lim = histogram(clas, bins=int(clas.ptp()),
        range=(0.5,clas.max()+0.5))
    cent = (lim[:-1]+lim[1:])/2.
    
    # indices of good bins
    good = where(area > 0)[0]
    
    # centroids of good bins
    xnode = zeros(good.size)
    ynode = zeros(good.size)
    for j in range(good.size):
        p = where(clas == cent[good[j]])
        xnode[j] = x[p].mean()
        ynode[j] = y[p].mean()
    
    # reassign pixels of bins with S/N < targetSN to closest good bin
    bad = where(clas == 0)[0]
    for j in range(bad.size):
        index = ((x[bad[j]] - xnode)**2 + (y[bad[j]] - ynode)**2).argmin()
        clas[bad[j]] = good[index] + 1
    
    # recompute all centroids of the reassigned bins
    # these will be used as starting points for the CVT
    area, lim = histogram(clas, bins=int(clas.ptp())+1,
        range=(0.5,clas.max()+0.5))
    cent = (lim[:-1]+lim[1:])/2.
    
    good = where(area > 0)[0]
    for j in range(good.size):
        p = where(clas == cent[good[j]])
        xnode[j] = x[p].mean()
        ynode[j] = y[p].mean()
    
    return xnode, ynode

def weighted_centroid(x, y, density):
    
    """
    Computes weighted centroid of a bin -- eq 4 of Cappellari & Copin (2003).
    
    INPUTS
      x       : x-coordinate of pixels in bin
      y       : y-coordinate of pixels in bin
      density : pixel weights
    """
    
    
    mass = density.sum()
    xbar = (x*density).sum()/mass
    ybar = (y*density).sum()/mass
    
    return xbar, ybar


# In[ ]:



