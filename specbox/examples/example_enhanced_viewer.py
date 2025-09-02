#!/usr/bin/env python
"""
Example script for using the enhanced spectrum viewer with image cutouts.
"""

# Prefer the enhanced thread if available, otherwise fall back to the standard one
try:
    from specbox.qtmodule import PGSpecPlotThreadEnhanced
except Exception:
    from specbox.qtmodule import PGSpecPlotThread as PGSpecPlotThreadEnhanced

from specbox.basemodule import SpecEuclid1d  # or SpecLAMOST, SpecSDSS, etc.

def run_enhanced_viewer():
    """Run the enhanced spectrum viewer."""
    
    # Example 1: Using a multi-extension FITS file
    specfile = 'COMBINED_SPECS.fits'  # Replace with your FITS file path
    
    viewer = PGSpecPlotThreadEnhanced(
        spectra=specfile,  # Multi-extension FITS file
        SpecClass=SpecEuclid1d,  # Spectrum class to use
        output_file='enhanced_vi_results.csv',  # Output CSV file
        z_max=5.0,  # Maximum redshift for slider
        load_history=True  # Load previous results if CSV exists
    )
    
    viewer.run()

def run_with_spectrum_list():
    """Run with a list of individual spectrum files."""
    
    # Example 2: Using a list of spectrum files
    from glob import glob
    
    basepath = 'lamost_spec/fits_files/'  # Replace with your path
    flist = glob(basepath + '*fits.gz')
    flist.sort()
    flist = flist[0:60]  # Process first 60 spectra
    
    from specbox.basemodule import SpecLAMOST
    
    viewer = PGSpecPlotThreadEnhanced(
        spectra=flist,  # List of individual FITS files
        SpecClass=SpecLAMOST,  # Use LAMOST spectrum class
        output_file='enhanced_lamost_vi.csv',
        z_max=6.0,
        load_history=True
    )
    
    viewer.run()

if __name__ == '__main__':
    # Run the enhanced viewer
    run_enhanced_viewer()