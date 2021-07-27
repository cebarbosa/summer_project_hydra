# -*- coding: utf-8 -*-
"""

Created on 28/10/2017

@author: Carlos Eduardo Barbosa

Detection of sources in data and separation of bins prior to Voronoi
tesselation

"""
from __future__ import division, print_function
import os

import pyregion
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

import sewpy

import context
from misc import array_from_header

def background_removed_data(imgname, redo=False, output=None, hdunum=1):
    """ Remove background from the image """
    data = fits.getdata(imgname, ext=1)
    output = "detection.fits"
    if os.path.exists(output) and not redo:
        return output
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (8, 8), filter_size=(5, 5),
                       sigma_clip=sigma_clip, bkg_estimator = bkg_estimator)
    outdata = data - bkg.background
    fits.writeto(output, outdata, overwrite=True)
    return output

def mask_from_regions(imgname, redo=False):
    """ Mask regions marked in file mask.reg made in ds9. """
    data = fits.getdata(imgname)
    filename = "mask.reg"
    outfile = "detection_masked.fits"
    if os.path.exists(outfile) and not redo:
        mask = fits.getdata(outfile)
        return mask
    r = pyregion.open(filename)
    for i, region in enumerate(r.get_filter()):
        mask = region.mask(data.shape)
        data[mask] = np.nan
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(outfile, overwrite=True)
    return outfile

def run_sextractor(img, redo=False, outfile=None):
    """ Produces a catalogue of sources in a given field. """
    if outfile is None:
        outfile = "source-catalog.fits"
    if os.path.exists(outfile) and not redo:
        return outfile
    params = ["NUMBER", "X_IMAGE", "Y_IMAGE", "KRON_RADIUS", "ELLIPTICITY",
               "THETA_IMAGE", "A_IMAGE", "B_IMAGE", "MAG_AUTO", "FLUX_RADIUS"]
    config = {"CHECKIMAGE_TYPE": "BACKGROUND",
                            "CHECKIMAGE_NAME": "background.fits",
                            "DETECT_THRESH" : 1.5}
    sew = sewpy.SEW(config=config, sexpath="source-extractor", params=params)
    cat = sew(img)
    cat["table"].write(outfile, format="fits", overwrite=True)
    return outfile

def mask_sources(img, cat, ignore=None, redo=False, output=None):
    """ Produces segmentation image with bins for detected sources using
    elliptical regions. """
    if output is None:
        output = "sources_mask.fits"
    if os.path.exists(output) and not redo:
        return output
    data = fits.getdata(img)
    ydim, xdim = data.shape
    xx, yy = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))
    table = Table.read(cat, 1)
    if ignore is not None:
        idx = np.array([i for i,x in enumerate(table["NUMBER"]) if x not in
                        ignore])
        table = table[idx]
    axratio = table["B_IMAGE"] / table["A_IMAGE"]
    # table = table[axratio > 0.4]
    mask = np.zeros_like(data)
    for source in table:
        R = calc_isophotes(xx, yy, source["X_IMAGE"], source["Y_IMAGE"], \
                           source["THETA_IMAGE"] - 90, source["B_IMAGE"] /
                           source["A_IMAGE"])
        Rmax = 1.5 * source["KRON_RADIUS"]
        mask += np.where(R <= Rmax, 1, 0)
    hdu = fits.PrimaryHDU(mask)
    hdu.writeto(output, overwrite=True)
    return output

def calc_isophotes(x, y, x0, y0, PA, q):
    """ Calculate isophotes for a given component. """
    x = np.copy(x) - x0
    y = np.copy(y) - y0
    shape = x.shape
    theta = np.radians(PA)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[s, c], [-c, s]])
    xy = np.dot(np.column_stack((x.flatten(), y.flatten())), rot).T
    x = np.reshape(xy[0], newshape=shape)
    y = np.reshape(xy[1], newshape=shape)
    return np.sqrt(np.power(x, 2) + np.power(y / q, 2))

def run_ngc3311(redo=False):
    data_dir = os.path.join(context.home_dir, "data")
    fields = context.fields
    for field in fields:
        os.chdir(os.path.join(data_dir, field))
        if field == "fieldA":
            imgname = "ellipse_model.fits"
        else:
            imgname = f"sn_field{field[-1]}.fits"
        detimg = background_removed_data(imgname, redo=redo)
        immasked = mask_from_regions(detimg, redo=redo)
        sexcat = run_sextractor(immasked, redo=redo)
        mask_sources(immasked, sexcat, redo=redo)

if __name__ == "__main__":
    run_ngc3311(redo=True)
