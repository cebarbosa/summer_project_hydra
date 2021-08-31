"""
Determination of optimal aperture and extraction of spectra of systems in the
halo of NGC 3311

"""

import os
import warnings

import numpy as np
from scipy.signal import argrelextrema
from astropy.io import fits
from astropy.table import Table, hstack
from astropy.stats import sigma_clipped_stats
from astropy.visualization import hist
from photutils.aperture import aperture_photometry
from photutils.aperture import EllipticalAperture, EllipticalAnnulus
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.utils.exceptions import AstropyWarning

import context
import misc

warnings.simplefilter('ignore', category=AstropyWarning)

def set_phot_aper(flux, fluxerr, mask, table, n_in=3, n_out=6, field=None,
                  redo=False):
    """ Performs aperture photometry to determine best the radius with
    highest S/N. """
    outfile = f"radius_extract_n{n_in}_{n_out}_rkron.fits"
    if os.path.exists(outfile) and not redo:
        otable = Table.read(outfile)
        return otable
    if os.path.exists(outfile):
        os.remove(outfile)
    plots_dir = os.path.join(os.getcwd(),
                          f"radius_extract_n{n_in}_{n_out}_rkron")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    field = "fieldA" if field is None else field
    r_extract = []
    desc = "Determining max S/N apertures"
    nkr = np.linspace(0.1, n_in, 50)
    for i, t in enumerate(tqdm(table, desc=desc)):
        objid = f"{field}_{t['NUMBER']:03d}"
        # Setting the geometry of ellipses
        x0 = t["X_IMAGE"] -1
        y0 = t["Y_IMAGE"] - 1
        a = t["A_IMAGE"]
        b = t["B_IMAGE"]
        ar = b / a
        rkron = t["KRON_RADIUS"]
        if rkron == 0:
            r_extract.append(np.nan)
            continue
        theta = np.deg2rad(t["THETA_IMAGE"])
        # Preparing several apertures to be extracted
        smas = nkr * rkron
        # Performing photometry of objects
        apertures = [EllipticalAperture((x0, y0), sma, sma * ar,
                                        theta=theta) for sma in smas]
        areas = np.array([aper.area for aper in apertures])

        # unmask the source itself from the general mask
        objmask = apertures[-1].to_mask(method="center").to_image(
                                          flux.shape).astype(np.float)
        source_mask = (mask - objmask == 0).astype(np.int)
        # Performing the photometry inside the source
        phot_table = aperture_photometry(flux, apertures, error=fluxerr,
                                         mask=source_mask)
        phot = np.array([phot_table[f"aperture_sum_{i}"].data[0] for i in
                         range(len(smas))])
        photerr_table = aperture_photometry(fluxerr**2, apertures,
                                            error=fluxerr, mask=source_mask)
        photerr = np.sqrt([photerr_table[f"aperture_sum_{i}"].data[0] for i in
                           range(len(smas))])
        # Performing photometry of halo annulus
        annulus_aperture = EllipticalAnnulus((x0, y0), a_in=n_in * rkron,
                            a_out=n_out * rkron, b_out=n_out * rkron * ar,
                            b_in=n_in * rkron * ar, theta=theta)
        annmask = annulus_aperture.to_mask(method='center')
        annulus_data = annmask.multiply(flux)
        annulus_mask = annmask.multiply(source_mask)
        # plt.subplot(1,2,1)
        # plt.imshow(annulus_data, origin="lower")
        # plt.subplot(1,2,2)
        # plt.imshow(annulus_mask, origin="lower")
        # plt.show()
        annulus_data_1d = annulus_data[annulus_mask==1]
        bg_mean, bg_med, bg_std = sigma_clipped_stats(annulus_data_1d)
        background = bg_med * areas
        source = phot - background
        noise = np.sqrt((bg_std * areas)**2 + photerr**2)
        aper_sn = source / noise
        idx_max = argrelextrema(aper_sn, np.greater)[0]
        r_sn_max = smas[idx_max][0] if len(idx_max) else np.nan
        r_extract.append(r_sn_max)
        plt.axvline(x=r_sn_max, c="k", ls="--")
        plt.plot(smas, aper_sn, "o-")
        plt.plot(smas[idx_max], aper_sn[idx_max], "xr", ms=10)
        plt.axvline(rkron, ls="--", c="y", label="$R=1R_{kron}$")
        # plt.axvline(n_in * rkron, ls="-.", c="y", label=f"$R_{{in}}={n_in}$")
        plt.title(objid)
        plt.xlabel("SMA (pixel)")
        plt.ylabel("S/N")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir,
                                 f"apphot_{objid}.png"), dpi=200)
        plt.close()
    t = Table([r_extract], names=["RADIUS_SN_MAX"])
    new_table = hstack([table, t])
    new_table.write(outfile, overwrite=True)
    return new_table

def plot_apertures(bkgimg, table, redo=False, n_in=3, n_out=6):
    """ Plot the apertures used for photometry in all cases on top of a
    background image. """
    outimg = f"apertures_n{n_in}_{n_out}_rkron.png"
    if os.path.exists(os.path.join(os.getcwd(), outimg)) and not redo:
        return
    vmin = np.nanpercentile(bkgimg, 10)
    vmax = np.nanpercentile(bkgimg, 98)
    # Impute median values for non-defined cases
    rsnmax = table["RADIUS_SN_MAX"]
    rsnmax_med = np.nanmedian(rsnmax)
    table["RADIUS_SN_MAX"][~np.isfinite(rsnmax)] = rsnmax_med
    # Impute values for the Kron Radius
    rkron = table["KRON_RADIUS"]
    rkron_med = np.nanmedian(rkron)
    table["KRON_RADIUS"][rkron <= 0] = rkron_med
    # Starting the figure with the background image
    fig = plt.figure(figsize=(8, 7.8))
    ax = plt.subplot(111)
    ax.grid(False)
    ax.imshow(bkgimg, vmin=vmin, vmax=vmax, origin="lower", cmap="viridis_r")
    # Plot all sources in catalog
    for t in table:
        # Setting the geometry of ellipses
        x0 = t["X_IMAGE"] -1
        y0 = t["Y_IMAGE"] - 1
        ar = t["B_IMAGE"] / t["A_IMAGE"]
        theta = np.deg2rad(t["THETA_IMAGE"])
        r_sn_max = t["RADIUS_SN_MAX"]
        rkron = t["KRON_RADIUS"]
        aper = EllipticalAperture((x0, y0), r_sn_max, r_sn_max * ar,
                                  theta=theta)
        aper_n_in = EllipticalAperture((x0, y0), n_in * rkron,
                                       rkron * ar * n_in, theta=theta)
        aper_n_out = EllipticalAperture((x0, y0), n_out * rkron,
                                       rkron * ar * n_out, theta=theta)
        aper.plot(lw=0.7, color="r")
        aper_n_in.plot(ls="--", lw=0.7, color="r")
        aper_n_out.plot(ls="--", lw=0.7, color="r")
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.99)
    plt.savefig(outimg, dpi=300)
    plt.show()
    plt.close()

def extract_spectra_from_cube(cubename, flux, table, mask, n_in=2.5,
                              n_out=4, field=None):
    """ Extract the spectra from the datacubes. """
    specs_dir = os.path.join(os.getcwd(),
                      f"spec1D_n{n_in}_{n_out}_rkron")
    if not os.path.exists(specs_dir):
        os.mkdir(specs_dir)
    field = "field" if field is None else field
    mask = mask.astype(int)
    # Impute median values for non-defined cases
    rsnmax = table["RADIUS_SN_MAX"]
    rsnmax_med = np.nanmedian(rsnmax)
    table["RADIUS_SN_MAX"][~np.isfinite(rsnmax)] = rsnmax_med
    # Impute values for the Kron Radius
    rkron = table["KRON_RADIUS"]
    rkron_med = np.nanmedian(rkron)
    table["KRON_RADIUS"][rkron <= 0] = rkron_med
    desc1 = f"Processing {field} "
    desc2 = f"Extracted source"
    wave = misc.array_from_header(cubename)
    outdata = np.zeros((2, len(table), 2, len(wave)))
    for i, extname in enumerate(tqdm(["DATA", "STAT"], desc1)):
        cube = fits.getdata(cubename, ext=i+1)
        for j, t in enumerate(tqdm(table, desc=desc2)):
            # Setting the geometry of ellipses
            x0 = t["X_IMAGE"] -1
            y0 = t["Y_IMAGE"] - 1
            ar = t["B_IMAGE"] / t["A_IMAGE"]
            rkron = t["KRON_RADIUS"]
            theta = np.deg2rad(t["THETA_IMAGE"])
            r_sn_max = t["RADIUS_SN_MAX"]
            # Defining apertures for the photometry without the correct mask
            aper = EllipticalAperture((x0, y0), r_sn_max, r_sn_max * ar,
                                      theta=theta)
            annulus_aperture = EllipticalAnnulus((x0, y0), a_in=n_in * rkron,
                        a_out=n_out * rkron, b_out=n_out * rkron * ar,
                        b_in=n_in * rkron * ar, theta=theta)
            # unmask the source itself from the general mask
            objmask = aper.to_mask(method="center").to_image(
                                              flux.shape).astype(np.float)
            source_mask = (mask - objmask == 0).astype(np.int)
            areas = []
            for k, region in enumerate([aper, annulus_aperture]):
                region_mask = region.to_mask(method='center').to_image(
                                              flux.shape).astype(np.float)
                # get x and y indices where the source is located
                idy, idx = np.where(region_mask == 1)
                areas.append(len(idx))
                # Extracting spectra from cube
                specs = cube[:, idy, idx]
                # Removing masked regions
                specmask = source_mask[idy, idx]
                specs = specs[:, specmask==1]
                spec1D = np.nansum(specs, axis=1)
                if k==1:
                    spec1D *= areas[0] / areas[1]
                outdata[i, j, k, :] = spec1D
    # Producing output tables
    print("Saving output tables")
    colnames = ["wave", "flux_source", "fluxerr_source", "flux_annulus",
                "fluxerr_annulus"]
    for j, t in enumerate(table):
        output = os.path.join(os.getcwd(), specs_dir,
                              f"{field}_{t['NUMBER']:03d}.fits")
        t = Table([wave, outdata[0, j, 0, :], np.sqrt(outdata[1, j, 0, :]),
                   outdata[0, j, 1, :], np.sqrt(outdata[1, j, 1, :])],
                  names=colnames)
        t.write(output, overwrite=True)


def run(n_in=2.5, n_out=4, redo=False):
    """ Pipeline to run all routines of extranction from data of NGC 3311. """
    for i, field in enumerate(context.fields):
        wdir = os.path.join(context.home_dir, f"data/{field}")
        os.chdir(wdir) # Changing to working directory
        table = Table.read("source-catalog.fits")
        snimage = f"sn_{field}.fits"
        flux = fits.getdata(snimage)
        fluxerr = fits.getdata(snimage, ext=2)
        # Load mask with all detected sources
        mask = fits.getdata("sources_mask.fits")
        # Load mask with objects masked by hand
        mask_regions = fits.getdata("mask_regions.fits")
        # Combine the two masks
        mask[np.isnan(mask_regions)] += 1
        # Calculate the best aperture for extraction
        table = set_phot_aper(flux, fluxerr, mask, table, redo=redo,
                              field=field, n_in=n_in, n_out=n_out)
        # Producing a finding chart for the sources
        detection = fits.getdata("detection_masked.fits")
        plot_apertures(detection, table, n_in=n_in, n_out=n_out, redo=redo)
        # Extracting the spectra in the aperturesn_in=n_in, n_out=n_out
        cubename = f"NGC3311_Field{field[-1]}_DATACUBE_COMBINED.fits"
        extract_spectra_from_cube(cubename, flux, table, mask,
                                  n_in=n_in, n_out=n_out, field=field)


if __name__ == "__main__":
    run()
