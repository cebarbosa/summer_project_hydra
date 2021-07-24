import os
import warnings

import numpy as np
from scipy.signal import argrelextrema
from astropy.io import fits
from astropy.table import Table, hstack
from astropy.stats import sigma_clipped_stats
from photutils.aperture import aperture_photometry
from photutils.aperture import EllipticalAperture, EllipticalAnnulus
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from astropy.utils.exceptions import AstropyWarning

import context
import misc

warnings.simplefilter('ignore', category=AstropyWarning)

def set_phot_aper(flux, fluxerr, mask, table, n_in=2, n_out=3, smas=None,
                  field=None, redo=False):
    if "RADIUS_EXTRACT" in table.columns and not redo:
        return table
    aper_phot_plots_dir = os.path.join(os.getcwd(), "extract_apertures")
    if not os.path.exists(aper_phot_plots_dir):
        os.mkdir(aper_phot_plots_dir)
    field = "fieldA" if field is None else field
    r_extract = []
    desc = "Determining max S/N apertures"
    nkr = np.linspace(0.1, 1, 50)
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
        source_mask = np.logical_or(mask - objmask > 0, np.isnan(flux))
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
        annmask_img = annmask.to_image(flux.shape)
        usemask = (mask == 0)
        goodpix = annmask.multiply(usemask)
        annulus_data = annmask.multiply(flux)
        annulus_data_1d = annulus_data[goodpix == 1]
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
        plt.savefig(os.path.join(aper_phot_plots_dir,
                                 f"apphot_{objid}.png"), dpi=200)
        plt.close()
    t = Table([r_extract], names=["r_extract"])
    new_table = hstack([table, t])
    return new_table

def select_candidates(table, seeing, dflux_radius=0.5, field=None, redo=False,
                      ellip_max=0.3):
    """ Narrow the number of GC candidates according to their morphology. """
    outtable = os.path.join(os.getcwd(), "gc_candidates.fits")
    if os.path.exists(outtable) and not redo:
        table = Table.read(outtable)
        return table
    title = f"Field {field[-1]}"
    table = table[table["FLUX_RADIUS"] > 0]
    table =  table[table["FLUX_RADIUS"] < 5]
    candidates = table[table["FLUX_RADIUS"] < 4]
    # Remove outliers in flux
    mags = candidates["MAG_AUTO"]
    mmean, mmedian, mstd = sigma_clipped_stats(mags)
    candidates = candidates[np.abs(mags - mmedian) <= 5 * mstd]
    # Use only round sources
    ellip = candidates["ELLIPTICITY"]
    candidates = candidates[ellip <= ellip_max]
    # Use only sources with flux_radius similar to point sources
    flux_radius = candidates["FLUX_RADIUS"]
    candidates = candidates[np.abs(flux_radius - seeing) <= dflux_radius]
    candidates.write(outtable, overwrite=True)
    # Plotting results
    fig = plt.figure(figsize=(context.fig_width, 2.5))
    ax = plt.subplot(111)
    ax.grid(True)
    idx = [i for i, n in enumerate(table["NUMBER"])
             if n not in candidates["NUMBER"]]
    ax.plot(table["FLUX_RADIUS"][idx], table["MAG_AUTO"][idx], "x",
             label=r"$\varepsilon > {}$".format(ellip_max))
    ax.plot(candidates["FLUX_RADIUS"], candidates["MAG_AUTO"], ".",
             label=r"$\varepsilon \leq {}$".format(ellip_max))
    # ymin = candidates["MAG_AUTO"].max() + 1
    # ymax = candidates["MAG_AUTO"].min() - 1
    # ax.set_ylim(ymin, ymax)
    ax.set_ylim(ax.get_ylim()[::-1])
    # Including rectangle
    ymin = candidates["MAG_AUTO"].min() - 0.5
    ymax = candidates["MAG_AUTO"].max() + 0.5
    xmin = seeing - dflux_radius
    rect=mpatches.Rectangle((xmin, ymin), 2 * dflux_radius, ymax - ymin,
                            fill=False, color ="purple", linewidth = 2)
    plt.gca().add_patch(rect)
    plt.legend(title=title)
    plt.xlabel("Flux radius (arcsec)")
    plt.ylabel("mag (arbitrary units)")
    plt.subplots_adjust(left=0.1, top=0.99, bottom=0.13, right=0.99)
    plt.savefig(f"gc_candidates_{field}.png", dpi=250)
    plt.close()
    return candidates

def plot_apertures(data, table, n_in=5, n_out=10, redo=True):
    outimg = f"apertures_nin{n_in}_nout{n_out}.png"
    if os.path.exists(os.path.join(os.getcwd(), outimg)) and not redo:
        return
    vmin = np.nanpercentile(data, 10)
    vmax = np.nanpercentile(data, 98)
    fig = plt.figure(figsize=(8, 7.8))
    ax = plt.subplot(111)
    ax.grid(False)
    ax.imshow(data, vmin=vmin, vmax=vmax, origin="lower")
    for t in table:
        # Setting the geometry of ellipses
        x0 = t["X_IMAGE"] -1
        y0 = t["Y_IMAGE"] - 1
        ar = t["B_IMAGE"] / t["A_IMAGE"]
        rkron = t["KRON_RADIUS"]
        theta = np.deg2rad(t["THETA_IMAGE"])
        r_sn_max = t["r_extract"]
        aper = EllipticalAperture((x0, y0), r_sn_max, r_sn_max * ar,
                                  theta=theta)
        aper_n_in = EllipticalAperture((x0, y0), n_in,
                                       n_in * ar, theta=theta)
        aper_n_out = EllipticalAperture((x0, y0), n_out,
                                       n_out * ar, theta=theta)
        aper.plot(lw=0.5)
        aper_n_in.plot(ls="--", lw=0.5)
        aper_n_out.plot(ls="--", lw=0.5)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.99)
    plt.savefig(outimg, dpi=300)
    plt.show()
    plt.close()

def extract_spectra(wave, flux, cube, table, mask, n_mask=1):
    """ Extract the spectra from the cubes. """
    mask = mask.astype(int)
    zdim, ydim, xdim = cube.shape
    xpix = np.arange(xdim) + 1
    ypix = np.arange(ydim) + 1
    for t in table:
        # Setting the geometry of ellipses
        x0 = t["X_IMAGE"] -1
        y0 = t["Y_IMAGE"] - 1
        ar = t["B_IMAGE"] / t["A_IMAGE"]
        rkron = t["KRON_RADIUS"]
        theta = np.deg2rad(t["THETA_IMAGE"])
        r_sn_max = t["r_extract"] if np.isfinite(t["r_extract"]) else rkron

        aper = EllipticalAperture((x0, y0), r_sn_max, r_sn_max * ar,
                                  theta=theta)
        annulus_aperture = EllipticalAnnulus((x0, y0), a_in=n_in * rkron,
                    a_out=n_out * rkron, b_out=n_out * rkron * ar,
                    b_in=n_in * rkron * ar, theta=theta)
        # unmask the source itself from the general mask
        aper_n_mask = EllipticalAperture((x0, y0), n_mask * rkron, n_mask *
                                       rkron * ar, theta=theta)
        source_mask = mask - aper_n_mask.to_mask(method="center").to_image(
                                          flux.shape).astype(np.float)
        for i, region in enumerate([aper, annulus_aperture]):
            phot_mask = region.to_mask(method="center").to_image(
                                         flux.shape).astype(np.float)
            # get x and y indices where the source is located
            idy, idx = np.where(phot_mask != 0)
            # Extracting spectra from cube
            specs = cube[:, idy, idx]
            # Removing masked regions
            specmask = source_mask[idy, idx]
            specs = specs[:, specmask==0]
            spec1D = np.nanmean(specs, axis=1)
            plt.plot(wave, spec1D)
        plt.show()

if __name__ == "__main__":
    redo = True
    stars_rh = [0.65, 0.66, 0.75, 1.]
    for i, field in enumerate(context.fields):
        seeing = context.seeing[i]
        wdir = os.path.join(context.home_dir, f"data/{field}")
        os.chdir(wdir) # Changing to working directory
        table = Table.read("sexcat.fits")
        # Converting table to arcsec
        table["FLUX_RADIUS"] *= context.PS
        # Offset for magnitude
        table["MAG_AUTO"] += 25
        snimage = f"sn_{field}.fits"
        flux = fits.getdata(snimage)
        fluxerr = fits.getdata(snimage, ext=2)
        # Load mask with all detected sources
        mask = fits.getdata("sources_mask.fits")
        table = set_phot_aper(flux, fluxerr, mask, table, redo=redo)

        # Select systems that are more likely GC candidates
        candidates = select_candidates(table, seeing / 2, field=field,
                                          redo=redo)


        # Calculate the best aperture for extraction
        # Producing a finding chart for the sources
        detection = fits.getdata("detection_masked.fits")
        plot_apertures(detection, candidates)
        continue
        # Extracting the spectra in the apertures
        cubename = "NGC3311_FieldA_DATACUBE_COMBINED.fits"
        wave = misc.array_from_header(cubename)
        cube = fits.getdata(cubename)
        extract_spectra(wave, flux, cube, table, mask)