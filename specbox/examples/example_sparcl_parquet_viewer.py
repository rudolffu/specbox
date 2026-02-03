#!/usr/bin/env python
"""
Example: inspect a multi-row SPARCL spectra parquet file.

This works with any dataframe-backed spectra file supported by SpecSparcl
as long as each row contains array columns for wavelength/flux/ivar.
"""

from specbox.basemodule import SpecSparcl
from specbox.qtmodule import PGSpecPlotThreadEnhanced


def main():
    specfile = "sparcl_spectra.parquet"  # repo example
    viewer = PGSpecPlotThreadEnhanced(
        spectra=specfile,
        SpecClass=SpecSparcl,
        output_file="sparcl_vi_results.csv",
        z_max=6.0,
        load_history=True,
    )
    viewer.run()


if __name__ == "__main__":
    main()

