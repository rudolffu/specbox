#!/usr/bin/env python
from astropy.io import fits
import numpy as np
from glob import glob
import re
import matplotlib.pyplot as plt
import os
from PyAstronomy import pyasl
from scipy.signal import savgol_filter
from pathlib import Path
from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from astropy import units as u
from specutils import Spectrum1D,SpectrumCollection
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
from .auxmodule import *


class Spiraf():
    def __init__(self, fname, redshift=None, ra=None, dec=None, 
                 telescope=None, side=None):
        hdu = fits.open(fname)
        header = hdu[0].header
        objname = header['object']
        self.hdu = hdu
        self.hducopy = hdu.copy()
        self.fname = os.path.basename(fname)
        self.fnametrim = re.sub('\.fits$', '_trim.fits', self.fname)
        if redshift is None:
            try:
                redshift = float(header['redshift'])
            except:
                print("Redshift not provided, setting redshift to zero.")
                redshift = 0
        if ra is None or dec is None:
            try:
                ra = float(header['ra'])
                dec = float(header['dec'])
            except:
                coord = SkyCoord(header['RA']+header['DEC'], 
                                 frame='icrs',
                                 unit=(u.hourangle, u.deg))
                ra = coord.ra.value
                dec = coord.dec.value
        if 'J' in objname:
            try:
                name = designation(ra, dec, telescope)
            except:
                name = objname
        else:
            name = objname
        if side is not None:
            name = name + side
        self.name = name
        CRVAL1 = hdu[0].header['CRVAL1']
        try:
            CD1_1 = hdu[0].header['CD1_1']
        except:
            CD1_1 = hdu[0].header['CDELT1']
        CRPIX1 = hdu[0].header['CRPIX1']
        self.CRVAL1 = CRVAL1
        self.CD1_1 = CD1_1
        self.CRPIX1 = CRPIX1
        W1 = (1-CRPIX1) * CD1_1 + CRVAL1
        data = hdu[0].data
        self.data = data
        dim = len(data.shape)
        self.dim = dim
        if dim==1:
            num_pt = len(data)
            self.len = num_pt
            self.wave = np.linspace(W1, 
                                    W1 + (num_pt - 1) * CD1_1, 
                                    num=num_pt)
            self.flux = data
            self.error = None
        elif dim==3:
            num_pt = len(data)
            self.len = num_pt
            self.wave = np.linspace(W1, 
                                    W1 + (num_pt - 1) * CD1_1, 
                                    num=num_pt)
            self.flux = data[0,0,:]
            self.error = data[3,0,:]
        else:
            print("Warning: format neither onedspec nor multispec (3d)!\n")
#         hdu.close()

    def smooth(self, window_length, polyorder, **kwargs):
        """
        Smooth the spectrum with scipy.signal.savgol_filter.
        Parameters:
        ----------
            window_length : int
                The length of the filter window (i.e., the number of coefficients).
                `window_length` must be a positive odd integer. If `mode` is 'interp',
                `window_length` must be less than or equal to the size of `x`.
            polyorder : int
                The order of the polynomial used to fit the samples.
                `polyorder` must be less than `window_length`.
        """
        self.flux_sm = savgol_filter(self.flux,
                                     window_length=window_length,
                                     polyorder=polyorder,
                                     **kwargs)            

    def plot(self, axlim='auto'):
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.wave, self.flux, lw=1)
        # plt.xlim(xrange)
        # plt.xlim(3800, 8300)
        # plt.ylim(bottom=0)
        plt.xlabel(r'$\mathrm{Wavelength(\AA})$')
        plt.ylabel(r'$\mathrm{Flux(erg/s/cm^{2}/\AA)}$')
        # plt.title('Spectrum of ' + self.name.replace('_', ' '))
        plt.title(self.name.replace('_', ' '))
#         plt.savefig(self.fname.strip(".fits") + '_spec.pdf', dpi=300)


    def trim(self, basepath='./', w1=None, w2=None, output=None):
        CRVAL1 = self.CRVAL1
        CD1_1 = self.CD1_1
        CRPIX1 = self.CRPIX1
        dim = self.dim
        tpath = basepath+'trimmed/'
        Path(tpath).mkdir(exist_ok=True)
        if output is None:
            outputfn = tpath + self.fnametrim
        else:
            outputfn = tpath + output        
        try:
            lower = float(w1)
            pix1=int((lower-CRVAL1)/CD1_1+CRPIX1-1)
        except:
            pix1=0
        try:
            upper = float(w2)
            pix2=int((upper-CRVAL1)/CD1_1+CRPIX1-1)
        except:
            pix2=self.len
        
        if dim==1:
            newdata = self.data[pix1:pix2]
            newlen = len(newdata)
            newflux = newdata
        elif dim==3:
            newdata = self.data[:,:,pix1:pix2]
            newlen = newdata.shape[2]
            newflux = newdata[0,0,:]
        else:
            print("Warning: format neither onedspec nor multispec (3d)!\n")
        self.newdata = newdata
        newCRVAL1 = self.wave[pix1]
        self.hducopy[0].header['CRVAL1'] = newCRVAL1
        try:
            self.hducopy[0].data = newdata
            self.hducopy.writeto(outputfn,overwrite=True)
        except:
            self.hducopy.writeto(outputfn,overwrite=True)
            print("Warning: format neither onedspec nor multispec (3d)!\n")
        newwave = np.linspace(newCRVAL1, 
                              newCRVAL1 + (newlen - CRPIX1) * CD1_1, 
                              newlen)
        
        plt.figure(figsize=(8, 6))
        plt.plot(newwave, newflux, lw=1)
        plt.xlabel(r'$\mathrm{Wavelength(\AA})$')
        plt.ylabel(r'$\mathrm{Flux(erg/s/cm^{2}/\AA)}$')
        plt.title('Spectrum of ' + self.name.replace('_', ' '))
            
    def close(self):
        self.hducopy.close()
        self.hdu.close()


class DoubleSpec():
    def __init__(self, spb=None, spr=None, spbfile=None, sprfile=None, varb=None, varr=None, instr=None):
        if spb is None:
            spb = Spiraf(spbfile)
        if spr is None:
            spr = Spiraf(sprfile)
        self.spb = spb
        self.spr = spr
        self.spbfile = spbfile
        self.sprfile = sprfile
        if varb is not None:
            self.varb = Spiraf(varb)
        if varr is not None:
            self.varr = Spiraf(varr)
#         self.objname = os.path.basename(spb.fname)[0:10]
#         self.objname = os.path.basename(spb.fname)
        self.objname = fits.getheader(spbfile)['object']
        if instr is not None:
            self.writename = self.objname + "_" + str(instr) + "_comb.fits"
        else:
            self.writename = self.objname + "_comb.fits"
    
    def combine(self,output=None):
        spb = self.spb
        spr = self.spr
        bwave = spb.wave
        rwave = spr.wave
        new_disp_grid = np.arange(bwave[0], rwave[-1], spb.CD1_1) * u.AA
        newdata = np.empty([4,1,len(new_disp_grid)])
        for i in range(4):
            spec1 = Spectrum1D(spectral_axis=bwave*u.AA, 
                               flux=spb.data[i,0,:]* u.Unit('erg cm-2 s-1 AA-1')) 
            spec2 = Spectrum1D(spectral_axis=rwave*u.AA, 
                               flux=spr.data[i,0,:]* u.Unit('erg cm-2 s-1 AA-1'))     
            resampler = LinearInterpolatedResampler(extrapolation_treatment='zero_fill')
            new_spec1 = resampler(spec1, new_disp_grid)
            new_spec2 = resampler(spec2, new_disp_grid)
            new_spec1.flux.value[spb.len:] = new_spec2.flux.value[spb.len:]
            idxleft2 = int((spr.CRVAL1-spb.CRVAL1)/spb.CD1_1)+2
            new_spec2.flux.value[:idxleft2] = new_spec1.flux.value[:idxleft2]
            new_spec_lin = (new_spec1 + new_spec2)/2
            newdata[i,0,:] = new_spec_lin.flux
        self.combined_data = newdata
        combined_hdu = self.spb.hducopy.copy()
        combined_hdu[0].data = newdata
        self.combined_hdu = combined_hdu
        combined_hdu.writeto(self.writename)
        
        
    def combine1D(self, basepath='./', normalize_left=False, output=None):
        spb = self.spb
        spr = self.spr
        bwave = spb.wave
        rwave = spr.wave
        cpath = basepath+'combined/'
        Path(cpath).mkdir(exist_ok=True)
        new_disp_grid = np.arange(bwave[0], rwave[-1], spb.CD1_1) * u.AA
        if self.varb:
            spec1 = Spectrum1D(spectral_axis=bwave*u.AA, 
                               flux=spb.data* u.Unit('erg cm-2 s-1 AA-1'),
                               uncertainty=StdDevUncertainty(np.sqrt(self.varb.data))) 
        else:
            spec1 = Spectrum1D(spectral_axis=bwave*u.AA, 
                               flux=spb.data* u.Unit('erg cm-2 s-1 AA-1')) 
        if self.varr:
            spec2 = Spectrum1D(spectral_axis=rwave*u.AA, 
                               flux=spr.data* u.Unit('erg cm-2 s-1 AA-1'),
                               uncertainty=StdDevUncertainty(np.sqrt(self.varr.data)))
        else:
            spec2 = Spectrum1D(spectral_axis=rwave*u.AA, 
                               flux=spr.data* u.Unit('erg cm-2 s-1 AA-1'))           
        resampler = LinearInterpolatedResampler(extrapolation_treatment='zero_fill')
        new_spec1 = resampler(spec1, new_disp_grid)
        new_spec2 = resampler(spec2, new_disp_grid)
        idxleft2 = int((spr.CRVAL1-spb.CRVAL1)/spb.CD1_1)+2
        meanjoin_left = np.mean(new_spec1.flux.value[idxleft2:spb.len])
        meanjoin_right = np.mean(new_spec2.flux.value[idxleft2:spb.len])
        if normalize_left == True:
            new_spec1 = new_spec1/meanjoin_left*meanjoin_right
        new_spec1.flux.value[spb.len:] = new_spec2.flux.value[spb.len:]
        new_spec2.flux.value[:idxleft2] = new_spec1.flux.value[:idxleft2]
        try:
            new_spec1.uncertainty.array[spb.len:] = new_spec2.uncertainty.array[spb.len:]
            new_spec2.uncertainty.array[:idxleft2] = new_spec1.uncertainty.array[:idxleft2]
        except:
            pass
        new_spec_lin = (new_spec1 + new_spec2)/2
        self.new_spec_lin = new_spec_lin
#         new_spec_lin.write(self.writename, format="myfits-writer")
        wvl = new_spec_lin.spectral_axis.value
        flux = new_spec_lin.flux.value
        try:
            uncert = new_spec_lin.uncertainty.array
            self.fluxerr = new_spec_lin.uncertainty.array
        except:
            pass
        # Write spectrum providing wavelength array
        if uncert.any():
            fluxerr= uncert
        else:
            fluxerr=None
        pyasl.write1dFitsSpec(cpath+self.writename, 
                              flux, 
                              wvl=wvl, 
                              clobber=True,
                              fluxErr=fluxerr,
                              refFileName=self.spbfile)
#         newdata = new_spec_lin.flux
#         self.combined_data = newdata.value
#         combined_hdu = self.spb.hducopy.copy()
#         combined_hdu[0].data = newdata.value
#         combined_hdu[0].header['NAXIS1']=len(self.combined_data)
#         self.combined_hdu = combined_hdu
#         combined_hdu.writeto(self.writename)  
    
        
    def close(self):
        self.spb.hducopy.close()
        self.spb.hdu.close()
        self.spr.hducopy.close()
        self.spr.hdu.close()