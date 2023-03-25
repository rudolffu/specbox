#!/usr/bin/env python
from math import log
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from uncertainties import ufloat
# from uncertainties.umath import *

def designation(ra, dec, telescope=None) -> str:
    c = SkyCoord(ra=ra*u.degree, 
                 dec=dec*u.degree,frame='icrs')
    srahms = c.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
    sdecdms = c.dec.to_string(sep='', precision=1, alwayssign=True, pad=True)
    if telescope is not None:
        newname = 'J'+srahms+sdecdms+'_'+str(telescope)
    else:
        newname = 'J'+srahms+sdecdms
    return newname