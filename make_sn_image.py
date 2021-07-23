import os

import numpy as np
from astropy.io import fits

import context
from misc import array_from_header

def make_sn_images(cubename, outfile, redo=False, wmin=5590, wmax=5680):
    """ Collapse a MUSE data cube to produce a white-light image and a
    noise image.

    Input Parameters
    ----------------
    cubename : str
        Name of the MUSE data cube

    outfile : str
        Name of the output file

    redo : bool (optional)
        Redo calculations in case the outfile exists.
    """
    if os.path.exists(outfile) and not redo:
        return
    data = fits.getdata(cubename, 1)
    var = fits.getdata(cubename, 2)
    h0 = fits.getheader(cubename, 0)
    h = fits.getheader(cubename, 1)
    h2 = fits.getheader(cubename, 2)
    wave = array_from_header(cubename)
    idx = np.where((wave <= wmax) & (wave >= wmin))[0]
    h["NAXIS"] = 2
    h2["NAXIS"] = 2
    del_keys = ["NAXIS3", "CTYPE3", "CUNIT3", "CD3_3", "CRPIX3", "CRVAL3",
                "CRDER3", "CD1_3", "CD2_3", "CD3_1", "CD3_2"]
    for key in del_keys:
        del h2[key]
        del h[key]
    print("Starting collapsing process...")
    newdata = np.nanmean(data[idx,:,:], axis=0)
    noise = np.sqrt(np.nanmean(var[idx,:,:], axis=0))
    hdu0 = fits.PrimaryHDU()
    hdu0.header = h0
    hdu = fits.ImageHDU(newdata, h)
    hdu2 = fits.ImageHDU(noise, h2)
    hdulist = fits.HDUList([hdu0, hdu, hdu2])
    hdulist.writeto(outfile, overwrite=True)
    return

if __name__ == "__main__":
    for field in context.fields:
        wdir = os.path.join(context.home_dir, f"data/{field}")
        cubename = os.path.join(wdir, f"NGC3311_F"
                                      f"{field[1:]}_DATACUBE_COMBINED.fits")
        outfile = os.path.join(wdir, f"sn_{field}.fits")
        make_sn_images(cubename, outfile, redo=True, wmin=4800, wmax=7000)
