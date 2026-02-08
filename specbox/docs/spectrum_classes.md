# Spectrum classes

This page summarizes the main spectrum reader classes in `specbox.basemodule`.

All spectrum objects are designed to expose (at least) the following common fields:

- `wave`: wavelength array (usually an `astropy.units.Quantity`)
- `flux`: flux array (usually an `astropy.units.Quantity`)
- `err`: error array (may be `numpy.ndarray` or `Quantity`, depending on reader)
- `spec`: a `specutils.Spectrum` instance (when available)

Many classes also provide:

- `plot()` for quick matplotlib visualization
- `smooth()` for Savitzky–Golay smoothing
- `trim()` to select a wavelength range
- `to_restframe()` to shift to rest-frame using a redshift

## `SpecLAMOST`

Reader for LAMOST spectra in FITS format.

```python
from specbox import SpecLAMOST

spec = SpecLAMOST("path/to/lamost.fits")
spec.plot()
```

Notes:
- If you pass `flux_calibrated=True`, the class will interpret flux units as calibrated (see class docstring).

## `SpecSDSS`

Reader for SDSS spectra in FITS format.

```python
from specbox.basemodule import SpecSDSS

spec = SpecSDSS("path/to/sdss.fits")
spec.plot()
```

Notes:
- SDSS files typically store flux with an inverse-variance column; `SpecSDSS` converts this into `err`.

## `SpecIRAF`

Reader for IRAF-style 1D spectra stored in FITS. This is used for spectra reduced with IRAF
or stored in an IRAF-like header convention.

```python
from specbox.basemodule import SpecIRAF

spec = SpecIRAF("spec_J001554.18+560257.5_LJT.fits")
spec.plot()
spec.smooth(window_length=9, polyorder=3, plot=True, inplace=False)
trimmed = spec.trim((5000, 8000), plot=True, inplace=False)
```

Notes:
- `trim(inplace=True)` updates wavelength calibration keywords (`CRVAL1`, `CRPIX1`, `CDELT1`) and
  also trims the underlying data array accordingly.

## `SpecPandasRow`

Generic reader for “table-of-spectra” datasets readable by pandas where each row contains
array-like columns for wavelength/flux (and optionally error or inverse variance) plus scalar metadata.

Typical use cases include parquet/CSV/feather tables where each row is one spectrum.

```python
from specbox.basemodule import SpecPandasRow

spec = SpecPandasRow("spectra.parquet", ext=1)  # ext is a 1-based row index
spec.plot()
```

Notes:
- Column names and units depend on the specific table; `SpecSparcl` below is a specialized version
  for SPARCL tables.

## `SpecSparcl`

Specialized parquet/table reader for SPARCL-style “table-of-spectra” files.

```python
from specbox.basemodule import SpecSparcl

sp = SpecSparcl("sparcl_spectra.parquet", ext=1)
sp.plot()
```

Notes:
- Common metadata columns include `data_release` and `targetid`. If present, `euclid_object_id` can be
  used by the enhanced viewer for Euclid overlays.

## Common operations

### Plot

```python
ax = spec.plot()
ax.set_title(getattr(spec, "objname", "spectrum"))
```

### Smooth

```python
spec.smooth(window_length=9, polyorder=3, plot=True, inplace=False)
```

### Trim

```python
trimmed = spec.trim((5000, 8000), plot=True, inplace=False)
```

### Rest-frame conversion

```python
spec.to_restframe(z=2.0, inplace=True)
```

