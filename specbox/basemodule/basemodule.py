#!/usr/bin/env python
from astropy.io import fits
import numpy as np
import pandas as pd
from glob import glob
import re
import matplotlib.pyplot as plt
import os
from PyAstronomy import pyasl
from scipy.signal import savgol_filter
from pathlib import Path
from astropy.nddata import StdDevUncertainty,VarianceUncertainty,InverseVariance
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from specutils import Spectrum1D,SpectrumCollection,SpectrumList
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler, median_smooth
from ..auxmodule  import designation
import warnings
from astropy.units import Quantity
from astropy.stats import sigma_clip


class ConvenientSpecMixin():
    """A mixin class for convenient spectrum operations.
    """
    def __init__(self, wave=None, flux=None, err=None, 
                 wave_unit=None, flux_unit=None, *args, **kwargs):
        """
        Parameters
        ----------
        wave : array-like
            Wavelength array.   
        flux : array-like   
            Flux array.
        err : array-like
            Error array.
        wave_unit : astropy.units.Unit      
            Wavelength unit.
        flux_unit : astropy.units.Unit
            Flux unit.
        """
        super().__init__(*args, **kwargs)
        if wave_unit is not None:
            self.wave_unit = wave_unit
        else:
            self.wave_unit = u.angstrom
        if flux_unit is not None:
            self.flux_unit = flux_unit
        else:
            self.flux_unit = u.dimensionless_unscaled
        self.wave = wave
        self.flux = flux
        self.err = err

    @property
    def _length(self):
        return len(self.flux)
    
    @property
    def flux_unit(self):
        return self._flux_unit
    
    @flux_unit.setter
    def flux_unit(self, unit):
        try:
            self.flux = self.flux.to(unit)
        except:
            pass
        self._flux_unit = unit

    @property
    def flux(self):
        return self._flux
    
    @flux.setter
    def flux(self, flux):
        if flux is not None:
            self._flux = self._make_flux_quantity(flux)
        else:
            self._flux = None
    
    @property
    def wave_unit(self):
        return self._wave_unit
    
    @wave_unit.setter
    def wave_unit(self, unit):
        try:
            self.wave = self.wave.to(unit)
        except:
            pass
        self._wave_unit = unit

    @property
    def wave(self):
        return self._wave
    
    @wave.setter
    def wave(self, wave):
        if wave is not None:
            self._wave = self._make_wave_quantity(wave)
        else:
            self._wave = None

    @property
    def err(self):
        return self._err
    
    @err.setter
    def err(self, err):
        self._err = err

    @property
    def spec(self):
        return self._spec
    
    @spec.setter
    def spec(self, spec):
        self._spec = spec
        self.wave = spec.spectral_axis
        self.flux = spec.flux
        if self._has_uncertainty():
            self.err = spec.uncertainty.array
        
    def _make_flux_quantity(self, flux):
        if not isinstance(flux, Quantity):
            flux = flux * self.flux_unit
        return flux
    
    def _make_wave_quantity(self, wave):
        if not isinstance(wave, Quantity):
            wave = wave * self.wave_unit
        return wave
    
    def _has_uncertainty(self):
        return self.spec.uncertainty is not None
    
    def trim(self, wave_range):
        wave = self.wave.value
        idx = (wave >= wave_range[0]) & (wave <= wave_range[1])
        self.trimmed = True
        self.flux = self.flux[idx]
        self.wave = self.wave[idx]
        self.err = self.err[idx]
        self.spec = Spectrum1D(spectral_axis=self.wave, 
                               flux=self.flux, 
                               uncertainty=StdDevUncertainty(self.err)) 
        return self

    def flux_conserve_resample(self, wave, inplace=False):
        # if not inplace:
        #     return self.copy().flux_conserve_resample(wave, inplace=True)
        resampler = FluxConservingResampler()
        self.spec = resampler(self.spec, wave)
        self.wave = self.spec.spectral_axis
        self.flux = self.spec.flux
        return self
    
    def smooth(self, window_length, polyorder, plot=True, inplace=False, sigclip=False, **kwargs):
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
            plot : bool
                Plot the smoothed spectrum.
            inplace : bool
                Replace the original spectrum with the smoothed one.
            sigclip : bool
                Sigma clip the smoothed spectrum.
            **kwargs : dict
                Keyword arguments for scipy.signal.savgol_filter.
        Returns:
        -------
            self : ConvenientSpecMixin
                The smoothed spectrum.
        """
        if hasattr(self, 'flux_ori'):
            warnings.warn('The original spectrum is already replaced by the smoothed one. \
                           Smoothing not performed.', UserWarning)
            print('The original spectrum is already replaced by the smoothed one. \nSmoothing not performed.')
            return
        flux_sm = savgol_filter(self.flux.value,
                                window_length=window_length,
                                polyorder=polyorder,
                                **kwargs) 
        if sigclip == True:
            flux_sm = sigma_clip(flux_sm, sigma=6, maxiters=4)
            flux_sm = flux_sm.filled(0.0)
        flux_sm = flux_sm * self.flux.unit
        self.flux_sm = flux_sm
        if plot == True:
            ax = self.plot(label='original')
            ax.plot(self.wave, self.flux_sm, lw=1, c='r', label='smoothed')
            ax.legend()
        if inplace == True:
            self.flux_ori = self.flux
            self.flux = flux_sm
            self.spec = Spectrum1D(spectral_axis=self.wave, 
                                   flux=self.flux_sm, 
                                   uncertainty=StdDevUncertainty(self.err)) 
        return self
    
    def plot(self, ax=None, **kwargs):
        """
        Plot the spectrum.
        Parameters:
        ----------
            ax : matplotlib.axes.Axes
                The axes to plot on. If None, a new set of axes is created.
        Returns:
        -------
            ax : matplotlib.axes.Axes
                The axes that were plotted on.
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.wave, self.flux, lw=1, c='k', **kwargs)
        ax.set_xlabel("Wavelength ({})".format(self.wave.unit))
        ax.set_ylabel("Flux ({})".format(self.flux.unit))
        return ax
    
    def copy(self):
        return self.__class__(self.wave, self.flux, self.err)
    
    def to_restframe(self, z, inplace=False):
        """
        Convert the spectrum to the rest-frame.
        Parameters:
        ----------
            z : float
                Redshift of the spectrum.
            inplace : bool  
                If True, the spectrum is converted to the rest-frame in place.
                If False, only a new Spectrum1D object is created.
        """
        z = self.redshift
        if hasattr(self, 'spec_rest'):
            warnings.warn('Rest-frame spectrum already exisits. \
                           Spectrum unchanged.', UserWarning)
            return
        if z is not None:
            self.wave /= (1+z)
            self.flux *= (1+z)
            self.err *= (1+z)
            self.spec_rest = Spectrum1D(spectral_axis=self.wave, 
                                        flux=self.flux, 
                                        uncertainty=StdDevUncertainty(self.err))
            if inplace == True:
                self.spec = self.spec_rest
    

class SpecIOMixin():
    """
    Mixin class for reading and writing spectra.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read(self, filename, **kwargs):
        self.filename = filename
        self.basename = os.path.basename(filename)
        hdu = fits.open(filename)
        self.hdr = hdu[0].header
        self.data = hdu[0].data
        self.hdu = hdu

    def write(self, filename, **kwargs):
        hdu = fits.PrimaryHDU(self.data, header=self.hdr)
        hdu.writeto(filename, overwrite=True)


class SpecSDSS(SpecIOMixin, ConvenientSpecMixin):
    """
    Class for SDSS spectra.
    Parameters:
    ----------
        filename : str
            Name of the SDSS spectrum file.
    """
    def __init__(self, filename=None, *args, **kwargs):
        """
        Initialize the SpecSDSS class.
        Parameters:
        ----------
            filename : str
                Name of the SDSS spectrum file.
        """
        super().__init__(*args, **kwargs)
        self.wave_unit = u.Angstrom
        self.flux_unit = 1e-17 * u.erg / u.s / u.cm**2 / u.Angstrom
        if filename is not None:
            self.read(filename, **kwargs)

    def read(self, filename, **kwargs):
        """
        Read the SDSS spectrum file.
        Parameters:
        ----------
            filename : str
                Name of the SDSS spectrum file.
        """
        super().read(filename, **kwargs)
        header = self.hdr
        hdu = self.hdu
        if hdu[0].data:
            data = hdu[0].data
        else:
            data = hdu[1].data
        self.loglam = data['loglam']
        self.wave = 10**data['loglam'] * self.wave_unit
        self.flux = data['flux'] * self.flux_unit
        self.err = data['ivar']**-0.5 * 1e-17
        try:
            self.ra = header['plug_ra']          # RA 
            self.dec = header['plug_dec']        # DEC
        except:
            self.ra = header['ra']          # RA
            self.dec = header['dec']        # DEC
        self.plateid = header['PLATEID']
        self.mjd = header['MJD']
        self.fiberid = header['FIBERID']
        self.and_mask = data['AND_MASK']
        self.or_mask = data['OR_MASK']
        redshift = kwargs.get('redshift', None)
        if redshift is None:
            try: 
                redshift = self.hdu[2].data['Z'][0]
            except:
                redshift = header['Z']
        self.redshift = redshift
        # self.objname = self.hdr['OBJNAME']
        self.spec = Spectrum1D(spectral_axis=self.wave, flux=self.flux, 
                               uncertainty=StdDevUncertainty(self.err))
        self.filename = filename
        self.objname = designation(self.ra, self.dec)
        try:
            self.objid = header['SPEC_ID']
        except:
            self.objid = self.objname

    def write(self, filename, **kwargs):
        self.loglam = np.log10(self.wave.value)
        self.data = Table([self.loglam, self.flux, self.err], 
                          names=['loglam', 'flux', 'ivar'])
        self.hdr = fits.Header()
        self.hdr['Z'] = self.redshift
        self.hdr['RA'] = self.ra
        self.hdr['DEC'] = self.dec
        # self.hdr['OBJNAME'] = self.objname
        self.hdr['PLATEID'] = self.plateid
        self.hdr['MJD'] = self.mjd
        self.hdr['FIBERID'] = self.fiberid
        self.hdr['AND_MASK'] = self.and_mask
        self.hdr['OR_MASK'] = self.or_mask
        super().write(filename, **kwargs)

class SpecIRAF(ConvenientSpecMixin, SpecIOMixin):
    """
    Class for reading IRAF spectra.
    """
    def __init__(self, filename=None, *args, **kwargs):
        """
        Parameters:
        ----------
            filename : str
                Name of the file to read.    
        """
        super().__init__(*args, **kwargs)
        self.wave_unit=u.AA
        self.flux_unit=u.erg/u.s/u.cm**2/u.AA
        self.wave = None
        self.flux = None
        self.err = None
        self.telescope = kwargs.get('telescope', None)
        self.side = kwargs.get('side', None)
        if filename is not None:
            self.read(filename, **kwargs)

    def read(self, filename, **kwargs):
        """
        Read the spectrum.
        Parameters:
        ----------
            filename : str
                Name of the file to read.
            ra : float
                Right ascension of the object in degrees.
            dec : float
                Declination of the object in degrees.
            telescope : str
                Name of the telescope. Default is None.
            side : str
                Side (arm) of the spectrograph (e.g. blue, red). Default is None.
         """
        super().read(filename, **kwargs)
        header = self.hdr
        data = self.data
        ra = kwargs.get('ra', None)
        dec = kwargs.get('dec', None)
        objname = header['OBJECT']
        telescope = kwargs.get('telescope', self.telescope)
        side = kwargs.get('side', self.side)
        if ra is None or dec is None:
            try:
                ra = float(header['ra'])
                dec = float(header['dec'])
            except:
                coord = SkyCoord(header['RA']+' '+header['DEC'], 
                                 frame='icrs',
                                 unit=(u.hourangle, u.deg))
                ra = coord.ra.value
                dec = coord.dec.value
        self.ra = ra
        self.dec = dec
        if 'J' in objname:
            try:
                name = designation(ra, dec, telescope)
            except:
                name = objname
        else:
            name = objname
        if side is not None:
            name = name + side
        self.objname = name
        CRVAL1 = header['CRVAL1']
        try:
            CD1_1 = header['CD1_1']
        except:
            CD1_1 = header['CDELT1']
        CRPIX1 = header['CRPIX1']
        self.CRVAL1 = CRVAL1
        self.CD1_1 = CD1_1
        self.CRPIX1 = CRPIX1
        W1 = (1-CRPIX1) * CD1_1 + CRVAL1
        dim = len(data.shape)
        self.dim = dim
        if dim==1:
            num_pt = len(data)
            self.len = num_pt
            self.wave = np.linspace(W1, 
                                    W1 + (num_pt - 1) * CD1_1, 
                                    num=num_pt) * self.wave_unit
            self.flux = data * self.flux_unit
            self.err = np.zeros_like(self.flux)
        elif dim==3:
            num_pt = data.shape[2]
            self.len = num_pt
            self.wave = np.linspace(W1, 
                                    W1 + (num_pt - 1) * CD1_1, 
                                    num=num_pt) * self.wave_unit
            self.flux = data[0,0,:]  * self.flux_unit
            self.err = data[3,0,:]
        else:
            print("Warning: format neither onedspec nor multispec (3d)!\n")
        self.spec = Spectrum1D(spectral_axis=self.wave, 
                               flux=self.flux, 
                               uncertainty=StdDevUncertainty(self.err))


    
#     def trim_resample(self, wave_range, step, inplace=False):
#         new_disp_grid = np.arange(wave_range[0], wave_range[1], step) * u.AA
#         resampler = SplineInterpolatedResampler()
#         self.trimmed_spec = resampler(self.spec, new_disp_grid) 
#         if inplace == True:
#             self.spec = self.trimmed_spec
#             self._copy_spec_attr()

#     def slice(self, idx1, idx2, inplace=False):
#         self.trimmed_spec = self.spec[idx1:idx2]
#         if inplace == True:
#             self.spec = self.trimmed_spec
#             self.loglam = self.loglam[idx1:idx2]
#             self._copy_spec_attr()
        
#     def plot(self, fig_num=0):
#         plt.figure(num=fig_num, figsize=(16, 6))
#         plt.plot(self.spec.spectral_axis, self.spec.flux, lw=1, c='k')
#         # plt.plot(self.spec.spectral_axis, self.err)
#         plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
#         plt.ylabel(r'Flux [$\mathrm{erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$]')
#         if self.objname is not None:
#             plt.title(self.objname)
#         if self.objname is None and self.basename is not None:
#             plt.title(self.basename)
# #         plt.show()
        
#     def _copy_spec_attr(self):
#         self.wave = self.spec.spectral_axis
#         self.flux = self.spec.flux
#         self.err = self.spec.uncertainty.array

class SpecLAMOST(ConvenientSpecMixin, SpecIOMixin):
    """A class for LAMOST Low Resolution Spectral (LRS) data.
    """
    def __init__(self, filename=None, flux_calibrated=False, *args, **kwargs):
        """
        Parameters:
        ----------
        filename : str
            Name of the file to read.
        flux_calibrated : bool
            Whether the flux has been calibrated to absolute flux units.
            If True, the flux is in units of 1e-17 erg/s/cm^2/Angstrom.
            If False, the flux is in dimensionless units. Default is False.
        """
        super().__init__(*args, **kwargs)
        self.wave_unit=u.AA
        self.flux_calibrated = flux_calibrated
        if flux_calibrated:
            self.flux_unit = 1e-17 * u.erg/u.s/u.cm**2/u.AA 
        else:
            self.flux_unit = u.dimensionless_unscaled
        self.wave = None
        self.flux = None
        self.err = None
        if filename is not None:
            self.read(filename, **kwargs)

    def read(self, filename, **kwargs):
        """
        Read the data from a LAMOST LRS file.
        Parameters:
        ----------
        filename : str
            Name of the file to read.
        """
        super().read(filename, **kwargs)
        header = self.hdr
        hdu = self.hdu
        if hdu[0].data:
            data = hdu[0].data
        else:
            data = hdu[1].data
        self.data = data
        self.ra=header['RA']
        self.dec=header['DEC']
        self.plateid = header['OBSID']
        self.mjd = header['MJD'] 
        self.fiberid = header['FIBERID']
        self.objid = header['OBJNAME']
        self.objname = header['DESIG']
        if self.flux_calibrated:
            flux = data[0]
            ivar = data[1]
            wave = data[2]
            self.and_mask = data[3]
            self.or_mask = data[4] 
        else:
            flux = data['FLUX'].flatten()
            ivar = data['IVAR'].flatten()
            wave = data['WAVELENGTH'].flatten()
            self.and_mask = data['ANDMASK'].flatten()
            self.or_mask = data['ORMASK'].flatten()
        self.loglam = np.log10(wave)    
        z_pipe = header['z']
        self.redshift = z_pipe
        ivar[ivar == 0] = np.nan
        from scipy.interpolate import interp1d
        f = interp1d(wave, ivar, fill_value=np.nan)
        ivar = f(wave)
        err = np.sqrt(1/ivar)
        self.wave = wave * self.wave_unit
        self.flux = flux * self.flux_unit
        self.err = err
        self.spec = Spectrum1D(spectral_axis=self.wave, 
                               flux=self.flux, 
                               uncertainty=StdDevUncertainty(self.err))



#     def plot(self, axlim='auto'):
#         plt.figure(figsize=(8, 6))
#         plt.plot(self.wave, self.flux, lw=1)
#         # plt.xlim(xrange)
#         # plt.xlim(3800, 8300)
#         # plt.ylim(bottom=0)
#         plt.xlabel(r'$\mathrm{Wavelength(\AA})$')
#         plt.ylabel(r'$\mathrm{Flux(erg/s/cm^{2}/\AA)}$')
#         # plt.title('Spectrum of ' + self.name.replace('_', ' '))
#         plt.title(self.name.replace('_', ' '))
# #         plt.savefig(self.fname.strip(".fits") + '_spec.pdf', dpi=300)

    # def trim(self, basepath='./', w1=None, w2=None, output=None):
    #     CRVAL1 = self.CRVAL1
    #     CD1_1 = self.CD1_1
    #     CRPIX1 = self.CRPIX1
    #     dim = self.dim
    #     tpath = basepath+'trimmed/'
    #     Path(tpath).mkdir(exist_ok=True)
    #     if output is None:
    #         outputfn = tpath + self.fnametrim
    #     else:
    #         outputfn = tpath + output        
    #     try:
    #         lower = float(w1)
    #         pix1=int((lower-CRVAL1)/CD1_1+CRPIX1-1)
    #     except:
    #         pix1=0
    #     try:
    #         upper = float(w2)
    #         pix2=int((upper-CRVAL1)/CD1_1+CRPIX1-1)
    #     except:
    #         pix2=self.len
        
    #     if dim==1:
    #         newdata = self.data[pix1:pix2]
    #         newlen = len(newdata)
    #         newflux = newdata
    #     elif dim==3:
    #         newdata = self.data[:,:,pix1:pix2]
    #         newlen = newdata.shape[2]
    #         newflux = newdata[0,0,:]
    #     else:
    #         print("Warning: format neither onedspec nor multispec (3d)!\n")
    #     self.newdata = newdata
    #     newCRVAL1 = self.wave[pix1]
    #     self.hducopy[0].header['CRVAL1'] = newCRVAL1
    #     try:
    #         self.hducopy[0].data = newdata
    #         self.hducopy.writeto(outputfn,overwrite=True)
    #     except:
    #         self.hducopy.writeto(outputfn,overwrite=True)
    #         print("Warning: format neither onedspec nor multispec (3d)!\n")
    #     newwave = np.linspace(newCRVAL1, 
    #                           newCRVAL1 + (newlen - CRPIX1) * CD1_1, 
    #                           newlen)
        
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(newwave, newflux, lw=1)
    #     plt.xlabel(r'$\mathrm{Wavelength(\AA})$')
    #     plt.ylabel(r'$\mathrm{Flux(erg/s/cm^{2}/\AA)}$')
    #     plt.title('Spectrum of ' + self.name.replace('_', ' '))
            
    # def close(self):
    #     self.hducopy.close()
    #     self.hdu.close()


class DoubleSpec():
    def __init__(self, spb=None, spr=None, spbfile=None, sprfile=None, varb=None, varr=None, instr=None):
        if spb is None:
            spb = SpecIRAF(spbfile)
        if spr is None:
            spr = SpecIRAF(sprfile)
        self.spb = spb
        self.spr = spr
        self.spbfile = spbfile
        self.sprfile = sprfile
        if varb is not None:
            self.varb = SpecIRAF(varb)
        if varr is not None:
            self.varr = SpecIRAF(varr)
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


# class SdssSpec(SpecBase):
#     def __init__(self, fname, redshift=None, perform_rest=False):
#         hdu = fits.open(fname)
#         basename = os.path.basename(fname)
#         self.basename = basename
#         hdr = hdu[0].header
#         self.hdr = hdr
#         self.perform_rest = perform_rest
#         self.ra=hdr['plug_ra']          # RA 
#         self.dec=hdr['plug_dec']        # DEC
#         self.plateid = hdr['plateid']   # SDSS plate ID
#         self.mjd = hdr['mjd']           # SDSS MJD
#         self.fiberid = hdr['fiberid']   # SDSS fiber ID
#         if redshift is None:
#             try: 
#                 redshift = hdu[2].data['z']
#             except:
#                 print('Redshift not provided.')
#                 pass
#         self.redshift = redshift
#         data = hdu[1].data
#         hdu.close()
#         wave = 10**data['loglam'] * u.AA 
#         flux = data['flux'] * 10**-17 * u.Unit('erg cm-2 s-1 AA-1') 
#         ivar = pd.Series(data['ivar'])
#         ivar.replace(0, np.nan, inplace=True)
#         ivar_safe = ivar.interpolate()
#         err = 1./np.sqrt(ivar_safe.values) * 10**-17
#         self.wave = wave
#         self.loglam = data['loglam']
#         self.flux = flux
#         self.err = err    
#         self.spec = Spectrum1D(spectral_axis=wave, 
#                                flux=flux, 
#                                uncertainty=StdDevUncertainty(err))
#         if perform_rest == True:
#             self.to_restframe()
        
#     def plot_6sigma(self):
#         plt.figure()
#         mean = np.nanmean(self.spec.flux.value)
#         std = np.nanstd(self.spec.flux.value)
#         above_idx = np.where(abs(self.spec.flux.value)-6*std>0)
#         below_idx = np.where(abs(self.spec.flux.value)-6*std<0)
#         plt.step(self.spec.spectral_axis[below_idx], self.spec.flux[below_idx])
#         plt.step(self.spec.spectral_axis[above_idx], self.spec.flux[above_idx])
# #         plt.plot(self.spec.spectral_axis, self.spec.flux)
#         plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
#         plt.ylabel(r'Flux [$\mathrm{erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$]')
#         plt.title(self.basename)
        
#     def plot_err_sigma(self):
#         plt.figure()
#         med = np.nanmedian(self.err)
#         std = np.nanstd(self.err)
#         above_idx = np.where(abs(self.err)-6*std>0)
#         below_idx = np.where(abs(self.err)-6*std<0)
#         plt.step(self.spec.spectral_axis[below_idx], self.spec.flux[below_idx])
#         plt.step(self.spec.spectral_axis[above_idx], self.spec.flux[above_idx])
# #         plt.plot(self.spec.spectral_axis, self.spec.flux)
#         plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
#         plt.ylabel(r'Flux [$\mathrm{erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$]')
#         plt.title(self.basename)

class NIRSpecS3d():
    
    def __init__(self, fname):
        hdu = fits.open(fname)
        basename = os.path.basename(fname)
        self.basename = basename
        hdr = hdu[0].header
        sci = hdu[1].data
        sci_hdr = hdu[1].header
        # err = hdu[2].data
        # dq = hdu[3].data
        # wmap = hdu[4].data
        # hdrtab = Table(hdu[5].data)
        # data = {'hdr': hdr,
        #         'sci_hdr': sci_hdr,
        #         'sci': sci,
        #         'err': err,
        #         'dq': dq,
        #         'wmap': wmap,
        #         'hdrtab': hdrtab}
        # self.data = data
        sci = savgol_filter(sci,
                            window_length=25,
                            polyorder=3,
                            axis=0)
            #     for i in range(sci.shape[1]):
            # for j in range(sci.shape[2]):
            #     sci[:,i,j] = sigmaclip(sci[:,i,j])    
        sci_unit = u.Unit(sci_hdr['BUNIT'])
        wave_unit = u.Unit(sci_hdr['CUNIT3'])
        self.sci_unit = sci_unit
        self.wave_unit = wave_unit
        CRVAL3 = sci_hdr['CRVAL3']
        CDELT3 = sci_hdr['CDELT3']
        CRPIX3 = sci_hdr['CRPIX3']
        self.CRVAL3 = CRVAL3
        self.CDELT3 = CDELT3
        self.CRPIX3 = CRPIX3
        W1 = (1-CRPIX3) * CDELT3 + CRVAL3
        sci_shape = sci.shape
        dim = len(sci.shape)
        self.dim = dim
        if dim==3:
            num_pt = sci_shape[0]
            self.len = num_pt
            wave = np.linspace(W1, 
                               W1 + (num_pt - 1) * CDELT3, 
                               num=num_pt)
            self.wave = wave
        else:
            print('Not implemented.')
        # with asdf.open(fname) as af:
        #     wcslist = [af.tree["meta"]["wcs"]]
        # spectra = []
        # for hdu, wcs in zip(hdu, wcslist):
        sci[sci<=0]==np.nan
        flux_array = sci.T
        flux = Quantity(flux_array, unit=sci_unit)
        wavelength = Quantity(wave, unit=wave_unit)
        # grid = grid_from_bounding_box(wcs.bounding_box)[:, :, 0, 0]
        # _, _, wavelength_array = wcs(*grid)
        # _, _, wavelength_unit = wcs.output_frame.unit
        # Merge primary and slit headers and dump into meta
        slit_header = sci_hdr
        header = hdr.copy()
        header.extend(slit_header, strip=True, update=True)
        meta = {'header': header}

        # get uncertainty information
        ext_name = hdr.get("ERREXT", "ERR")
        err_type = hdu[ext_name].header.get("ERRTYPE", 'ERR')
        err_unit = hdu[ext_name].header.get("BUNIT", None)
        err_array = hdu[ext_name].data.T

        # ERRTYPE can be one of "ERR", "IERR", "VAR", "IVAR"
        # but mostly ERR for JWST cubes
        # see https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/science_products.html#s3d
        if err_type == "ERR":
            err = StdDevUncertainty(err_array, unit=err_unit)
        elif err_type == 'VAR':
            err = VarianceUncertainty(err_array, unit=err_unit)
        elif err_type == 'IVAR':
            err = InverseVariance(err_array, unit=err_unit)
        elif err_type == 'IERR':
            warnings.warn("Inverse error is not yet a supported astropy.nddata "
                          "uncertainty. Setting err to None.")
            err = None

        # get mask information
        mask_name = hdr.get("MASKEXT", "DQ")
        mask = hdu[mask_name].data.T
        spec = Spectrum1D(flux=flux, spectral_axis=wavelength, meta=meta,
                          uncertainty=err, mask=mask) 
        self.spec1d = spec