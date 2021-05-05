#!/usr/bin/env python
from astropy.io import fits
import numpy as np
from glob import glob
import re
import matplotlib.pyplot as plt
import os
from PyAstronomy import pyasl

from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from astropy import units as u
from specutils import Spectrum1D,SpectrumCollection
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler


class Spiraf():
    def __init__(self, fname, side=None):
        hdu = fits.open(fname)
        self.hdu = hdu
        self.hducopy = hdu.copy()
        name = fits.getheader(fname)['object']
        if side is not None:
            name = name + side
        self.name = name
        self.fname = name + ".fits"
        self.fnametrim = re.sub('\.fits$', 'trim.fits', self.fname)
        
        CRVAL1 = hdu[0].header['CRVAL1']
        CD1_1 = hdu[0].header['CDELT1']
        CRPIX1 = hdu[0].header['CRPIX1']
        self.CRVAL1 = CRVAL1
        self.CD1_1 = CD1_1
        self.CRPIX1 = CRPIX1
        data = hdu[0].data
        self.data = data
        dim = len(data.shape)
        self.dim = dim
        if dim==1:
            l = len(data)
            self.len = l
            self.wave = np.linspace(CRVAL1, 
                                    CRVAL1 + (l - CRPIX1) * CD1_1, 
                                    l)
            self.flux = data
        elif dim==3:
            l = data.shape[2]
            self.len = l
            self.wave = np.linspace(CRVAL1, 
                                    CRVAL1 + (l - CRPIX1) * CD1_1, 
                                    l)
            self.flux = data[0,0,:]
        else:
            print("Warning: format neither onedspec nor multispec (3d)!\n")
#         hdu.close()
            

    def plot(self, axlim='auto'):
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.wave, self.flux, lw=1)
        # plt.xlim(xrange)
        # plt.xlim(3800, 8300)
        # plt.ylim(bottom=0)
        plt.xlabel(r'$\mathrm{Wavelength(\AA})$')
        plt.ylabel(r'$\mathrm{Flux(erg/s/cm^{2}/\AA)}$')
        plt.title('Spectrum of ' + self.name.replace('_', ' '))
#         plt.savefig(self.fname.strip(".fits") + '_spec.pdf', dpi=300)


    def trim(self, w1='default', w2='default', output='default'):
        
        CRVAL1 = self.CRVAL1
        CD1_1 = self.CD1_1
        CRPIX1 = self.CRPIX1
        dim = self.dim
        if output == 'default':
            outputfn = self.fnametrim
        else:
            outputfn = output
        if not os.path.isfile(outputfn):
            pass
        else:
            raise OSError(f"File {outputfn!r} already exists.")        
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
        
        try:
            self.hducopy[0].data = newdata
            self.hducopy.writeto(outputfn)
        except:
            self.hducopy.writeto(outputfn)
            print("Warning: format neither onedspec nor multispec (3d)!\n")
        newwave = np.linspace(CRVAL1, 
                              CRVAL1 + (newlen - CRPIX1) * CD1_1, 
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
    def __init__(self, spb=None, spr=None, spbfile=None, sprfile=None, varb=None, varr=None):
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
        self.writename = self.objname + "_anu_comb.fits"
    
    def combine(self,output=None):
        spb = self.spb
        spr = self.spr
        bwave = spb.wave
        rwave = spr.wave
        new_disp_grid = np.arange(bwave[0], rwave[-1], spb.CD1_1) * u.AA
        linear = LinearInterpolatedResampler()
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
        
        
    def combine1D(self,output=None):
        spb = self.spb
        spr = self.spr
        bwave = spb.wave
        rwave = spr.wave
        new_disp_grid = np.arange(bwave[0], rwave[-1], spb.CD1_1) * u.AA
        linear = LinearInterpolatedResampler()
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
        new_spec1.flux.value[spb.len:] = new_spec2.flux.value[spb.len:]
        idxleft2 = int((spr.CRVAL1-spb.CRVAL1)/spb.CD1_1)+2
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
        pyasl.write1dFitsSpec(self.writename, 
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