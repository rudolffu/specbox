#!/usr/bin/env python
from astropy.io import fits
import numpy as np
import pandas as pd
from glob import glob
import re
import matplotlib.pyplot as plt
import os
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
        if self.wave is not None and self.flux is not None:
            self.spec = Spectrum1D(spectral_axis=self.wave, 
                                   flux=self.flux, 
                                   uncertainty=StdDevUncertainty(self.err))

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
    
    def trim(self, wave_range, plot=True, inplace=False):
        """
        Trim the spectrum to a given wavelength range.
        Parameters:
        ----------
            wave_range : tuple
                The wavelength range to trim the spectrum.
            plot : bool
                Plot the trimmed spectrum.
            inplace : bool
                If True, the spectrum is trimmed in place.
                If False, a new ConvenientSpecMixin object is created.
        Returns:
        -------
            self (if inplace=True) or a trimmed copy : ConvenientSpecMixin
                The trimmed spectrum.
        """
        wave = self.wave.value
        idx = (wave >= wave_range[0]) & (wave <= wave_range[1])
        flux = self.flux[idx]
        wave = self.wave[idx]
        err = self.err[idx]
        if plot == True:
            fig, ax = plt.subplots()
            ax.plot(wave, flux, lw=1, c='k')
            ax.set_xlabel("Wavelength ({})".format(self.wave.unit))
            ax.set_ylabel("Flux ({})".format(self.flux.unit))
        if inplace == True:
            self.wave = wave
            self.flux = flux
            self.err = err
            self.spec = Spectrum1D(spectral_axis=self.wave, 
                                   flux=self.flux, 
                                   uncertainty=StdDevUncertainty(self.err))
            self.trimmed = True
            self.trimmed_idx = idx
            return self
        else:
            return self.__class__(wave=wave, flux=flux, err=err,
                                  wave_unit=self.wave_unit, flux_unit=self.flux_unit)
        
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
    
    def to_restframe(self, z=None, inplace=False):
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
        if z is None:
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

    def read(self, filename, ext=0, **kwargs):
        self.filename = filename
        self.basename = os.path.basename(filename)
        hdu = fits.open(filename)
        self.hdr = hdu[ext].header
        self.data = hdu[ext].data
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
    def __init__(self, filename=None, redshift=None, *args, **kwargs):
        """
        Initialize the SpecSDSS class.
        Parameters:
        ----------
            filename : str
                Name of the SDSS spectrum file.
            redshift : float
                Redshift of the object.
        """
        super().__init__(*args, **kwargs)
        self.wave_unit = u.Angstrom
        self.flux_unit = 1e-17 * u.erg / u.s / u.cm**2 / u.Angstrom
        self.wave = kwargs.get('wave', None)
        self.flux = kwargs.get('flux', None)
        self.err = kwargs.get('err', None)
        if filename is not None:
            self.read(filename, redshift=redshift, **kwargs)

    def read(self, filename, redshift=None, **kwargs):
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
        # convert ivar to error and replace 0 with NaN
        ivar = data['ivar']
        ivar[ivar == 0] = np.nan
        self.err = data['ivar']**-0.5 * 1e-17
        # replace np.nan with np.inf
        self.err[np.isnan(self.err)] = np.inf
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
        if redshift is None:
            redshift = self.hdu[2].data['Z'][0] if 'Z' in self.hdu[2].data.columns.names else header['Z']
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
        self.wave_unit=kwargs.get('wave_unit', u.AA)
        self.flux_unit=kwargs.get('flux_unit', u.erg/u.s/u.cm**2/u.AA)
        self.wave = kwargs.get('wave', None)
        self.flux = kwargs.get('flux', None)
        self.err = kwargs.get('err', None)
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
        # if 'J' in objname:
        #     try:
        #         name = designation(ra, dec, telescope)
        #     except:
        #         name = objname
        # else:
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
    
    def trim(self, wave_range, plot=True, inplace=False):
        """
        Trim the spectrum to a given wavelength range.
        Parameters:
        ----------
            wave_range : tuple
                The wavelength range to trim the spectrum.
            plot : bool
                Plot the trimmed spectrum.
            inplace : bool
                If True, the spectrum is trimmed in place.
                If False, a new SpecIRAF object is created.
        Returns:
        -------
            self (if inplace=True) or a trimmed copy : SpecIRAF
                The trimmed spectrum.
        """
        trimmed_copy = super().trim(wave_range, plot=plot, inplace=inplace)
        if inplace == True:
            self.hdr['CRVAL1'] = self.wave.value[0]
            self.hdr['CRPIX1'] = 1
            self.hdr['CDELT1'] = self.wave.value[1] - self.wave.value[0]
            self.data = self.data[:,:,self.trimmed_idx]
            return self
        else:
            return trimmed_copy
            

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
        self.objname = spb.objname
        if instr is not None:
            self.writename = self.objname + "_" + str(instr) + "_comb.fits"
        else:
            self.writename = self.objname + "_comb.fits"
    
    def combine(self, normalize_left=False, output=None, overwrite=True):
        spb = self.spb
        spr = self.spr
        bwave = spb.wave.value
        rwave = spr.wave.value
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
            idxleft2 = int((spr.wave.value[0]-spb.wave.value[0])/spb.CD1_1)+2
            meanjoin_left = np.mean(new_spec1.flux.value[idxleft2:spb.len])
            meanjoin_right = np.mean(new_spec2.flux.value[idxleft2:spb.len])
            if normalize_left:
                new_spec1 = new_spec1/meanjoin_left*meanjoin_right
            new_spec1.flux.value[spb.len:] = new_spec2.flux.value[spb.len:]
            new_spec2.flux.value[:idxleft2] = new_spec1.flux.value[:idxleft2]
            new_spec_lin = (new_spec1 + new_spec2)/2
            newdata[i,0,:] = new_spec_lin.flux
        self.combined_data = newdata
        combined_hdu = self.spb.hdu.copy()
        combined_hdu[0].header['NAXIS1']=len(new_disp_grid)
        combined_hdu[0].header['CRVAL1']=new_disp_grid[0].value
        combined_hdu[0].header['CRPIX1']=1
        combined_hdu[0].header['CDELT1']=new_disp_grid[1].value-new_disp_grid[0].value
        combined_hdu[0].data = newdata
        self.combined_hdu = combined_hdu
        combined_hdu.writeto(self.writename, overwrite=overwrite)
        
    def close(self):
        self.spb.hdu.close()
        self.spr.hdu.close()

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
        

class SpecCoadd1d(ConvenientSpecMixin, SpecIOMixin):
    """
    Class for reading 1D coadded spectra generated by PypeIt.
    """
    def __init__(self, filename=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wave_unit = u.Angstrom
        self.flux_unit = 1e-17 * u.erg / u.s / u.cm**2 / u.Angstrom
        if filename is not None:
            self.read(filename, **kwargs)

    def read(self, filename, **kwargs):
        super().read(filename, **kwargs)
        hdu = self.hdu
        data = hdu[1].data
        self.wave = data['wave_grid_mid'] * self.wave_unit
        self.flux = data['flux'] * self.flux_unit
        self.err = data['sigma'] * self.flux_unit
        self.telluric = data['telluric'] * self.flux_unit
        self.spec = Spectrum1D(
            spectral_axis=self.wave, 
            flux=self.flux, 
            uncertainty=StdDevUncertainty(self.err))
        self.telluric_spec = Spectrum1D(
            spectral_axis=self.wave,
            flux=self.telluric)
                                        
    def write_to_iraf(self, w1=None, w2=None, dw=1.0, bad_wave_ranges=None):
        if w1 is None:
            w1 = self.wave[1].value
        if w2 is None:
            w2 = self.wave[-2].value
        hdr = self.hdr
        new_disp_grid = np.arange(w1, w2, dw) * u.AA
        fluxcon = FluxConservingResampler()
        newspec = fluxcon(self.spec, new_disp_grid)
        # check if newspec.uncertainty is InverseVariance, if so, convert to StdDevUncertainty
        if isinstance(newspec.uncertainty, InverseVariance):
            newspec.uncertainty = StdDevUncertainty(1/np.sqrt(newspec.uncertainty.array))
        if bad_wave_ranges is not None:
            for rng in bad_wave_ranges:
                idx = np.where(
                    (new_disp_grid.value >= rng[0]) & (new_disp_grid.value <= rng[1]))
                newspec.flux.value[idx] = np.nan
        # newtell = fluxcon(self.telluric_spec, new_disp_grid)
        multispecdata = fake_multispec_data(
            (newspec.flux.value, np.zeros_like(newspec.flux.value), 
             np.zeros_like(newspec.flux.value), newspec.uncertainty.array))
        sp_hdu = fits.PrimaryHDU(data=multispecdata)
        hdrcopy = hdr.copy(strip=True)
        sp_hdu.header.extend(hdrcopy, strip=True, update=True,
                     update_first=False, useblanks=True, bottom=False)
        sp_hdr = sp_hdu.header
        sp_hdr['OBJECT'] = hdr['TARGET']
        sp_hdr['NAXIS'] = 3
        sp_hdr['NAXIS1'] = len(newspec.flux.value)
        sp_hdr['NAXIS2'] = 1
        sp_hdr['NAXIS3'] = 4
        sp_hdr['WCSDIM'] = 3
        sp_hdr['WAT0_001'] = 'system=equispec'
        sp_hdr['WAT1_001'] = 'wtype=linear label=Pixel'
        sp_hdr['WAT2_001'] = 'wtype=linear'
        sp_hdr['CRVAL1'] = w1
        sp_hdr['CRPIX1'] = 1
        sp_hdr['CD1_1'] = dw
        sp_hdr['CD2_2'] = dw
        sp_hdr['CD3_3'] = dw
        sp_hdr['LTM1_1'] = 1
        sp_hdr['LTM2_2'] = 1
        sp_hdr['LTM3_3'] = 1
        sp_hdr['WAT3_001'] = 'wtype=linear'
        sp_hdr['CTYPE1'] = 'PIXEL'
        sp_hdr['CTYPE2'] = 'LINEAR'
        sp_hdr['CTYPE3'] = 'LINEAR'
        sp_hdr['BANDID1'] = 'spectrum - background fit, weights variance, clean no'               
        sp_hdr['BANDID2'] = 'raw - background fit, weights none, clean no'                        
        sp_hdr['BANDID3'] = 'background - background fit'                                         
        sp_hdr['BANDID4'] = 'sigma - background fit, weights variance, clean no'  
        newname = self.basename.strip('.fits') + '_iraf.fits'
        sp_hdu.writeto(f'{newname}', overwrite=True)
        


def fake_multispec_data(arrlist):
# https://github.com/jrthorstensen/opextract/blob/master/opextract.py#L337
   # takes a list of 1-d numpy arrays, which are
   # to be the 'bands' of a multispec, and stacks them
   # into the format expected for a multispec.  As of now
   # there can only be a single 'aperture'.

   return np.expand_dims(np.array(arrlist), 1)


class SpecEuclid1d(ConvenientSpecMixin, SpecIOMixin):
    """
    Class for reading 1D spectra from Euclid.
    """
    def __init__(self, filename=None, ext=None, extname=None, clip=True, *args, **kwargs):
        """
        Parameters:
        ----------
            filename : str
                Name of the Euclid spectrum file.
            ext : int
                Extension number of the spectrum.
            extname : str
                Extension name of the spectrum.
            clip : bool
                If True, clip the spectrum to the useful range.  
        """
        super().__init__(*args, **kwargs)
        self.wave_unit=u.AA
        self.flux_unit=u.erg/u.s/u.cm**2/u.AA
        self.wave = kwargs.get('wave', None)
        self.flux = kwargs.get('flux', None)
        self.err = kwargs.get('err', None)
        self.telescope = 'Euclid'
        if filename is not None:
            self.read(filename, ext, extname, clip, **kwargs)
        
    def read(self, filename, ext=None, extname=None, clip=True, **kwargs):
        """
        Read the Euclid 1D spectrum.
        Parameters:
        ----------
            filename : str
                Name of the Euclid spectrum file.
            ext : int
                Extension number of the spectrum.
            extname : str
                Extension name of the spectrum.
            clip : bool
                If True, clip the spectrum to the useful range.
        """
        hdul = fits.open(filename)
        if extname is None and ext is None:
            print('No extension specified. Reading the first extension.')
            ext = 1
            hdu = hdul[ext].copy()
        elif extname is not None:
            hdu = hdul[f'{extname}'].copy()
        elif ext is not None:
            hdu = hdul[ext].copy()
        hdul.close()
        data = hdu.data
        if clip:
            data = data[11:511]
        self.hdu = hdu
        self.data = data
        fscale = hdu.header.get('FSCALE', 1.0)
        self.wave = data['WAVELENGTH'] * u.Angstrom
        self.flux = data['SIGNAL'] * fscale * u.erg / u.s / u.cm**2 / u.Angstrom
        variance = data['VAR']
        self.err = np.sqrt(variance) * fscale
        self.spec = Spectrum1D(spectral_axis=self.wave, 
                               flux=self.flux, 
                               uncertainty=StdDevUncertainty(self.err))
        self.objname = hdu.name
        try:
            self.objid = int(hdu.name)
        except ValueError:
            self.objid = hdu.name  # fallback if conversion fails
        self.filename = filename
        self.ext = ext
        self.ra = hdu.header.get('RA', 0.0)
        self.dec = hdu.header.get('DEC', 0.0)
        self.z_ph = hdu.header.get('Z_PH', 0.0)
        self.z_gaia = hdu.header.get('Z_GAIA', 0.0)
        self.z_vi = hdu.header.get('Z_VI', 0.0)
        self.z_temp = hdu.header.get('Z_TEMP', None)
        
        if self.z_temp is not None and self.z_temp > 0:
            if abs(self.z_vi - self.z_temp) < 0.01 or self.z_vi == 0:
                self.z_vi = self.z_temp

    @property
    def z_vi(self):
        return self._z_vi
        
    @z_vi.setter
    def z_vi(self, value):
        self._z_vi = value
