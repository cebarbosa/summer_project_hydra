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
from tqdm import tqdm
from astropy.utils.exceptions import AstropyWarning

import context

warnings.simplefilter('ignore', category=AstropyWarning)

def calc_best_aper(flux, fluxerr, table, n_in=2, n_out=2.5, nkr=None,
                   field=None, wdir=None, redo=False):
    wdir = os.getcwd() if wdir is None else wdir
    outtable = os.path.join(wdir, f"maxsnr_nin{n_in}_nout{n_out}.fits")
    if os.path.exists(outtable) and not redo:
        t = Table.read(outtable)
        return t
    aper_phot_plots_dir = os.path.join(wdir, f"aperphot_nin{n_in}_nout{n_out}")
    if not os.path.exists(aper_phot_plots_dir):
        os.mkdir(aper_phot_plots_dir)
    nkr = np.linspace(0.1, n_in, 100)  if nkr is None else nkr # MUltiple of
                                                              # Kron Radius
    field = "fieldA" if field is None else field
    r_extract = []
    desc = "Determining max S/N apertures"
    for i, t in enumerate(tqdm(table, desc=desc)):
        objid = f"{field}_{t['NUMBER']:03d}"
        # Setting the geometry of ellipses
        x0 = t["X_IMAGE"] -1
        y0 = t["Y_IMAGE"] - 1
        ar = t["B_IMAGE"] / t["A_IMAGE"]
        rkron = t["KRON_RADIUS"]
        theta = np.deg2rad(t["THETA_IMAGE"])
        # Preparing several apertures to be extracted
        apertures = [EllipticalAperture((x0, y0), rkron * n, rkron * n * ar,
                                        theta=theta) for n in nkr]
        sma = np.array([aper.a for aper in apertures]) # semi-major axis
        areas = np.array([aper.area for aper in apertures])
        # Performing photometry of objects
        mask = np.isnan(flux)
        phot_table = aperture_photometry(flux, apertures, error=fluxerr,
                                         mask=mask)
        phot = np.array([phot_table[f"aperture_sum_{i}"].data[0] for i in
                         range(len(nkr))])
        photerr_table = aperture_photometry(fluxerr**2, apertures,
                                            error=fluxerr, mask=mask)
        photerr = np.sqrt([photerr_table[f"aperture_sum_{i}"].data[0] for i in
                            range(len(nkr))])
        # Performing photometry of halo annulus
        annulus_aperture = EllipticalAnnulus((x0, y0), a_in=n_in * rkron,
                            a_out=n_out * rkron, b_out=n_out * rkron * ar,
                            b_in=n_in * rkron * ar, theta=theta)
        annulus_masks = annulus_aperture.to_mask(method='center')
        mask = annulus_masks.data
        annulus_data = annulus_masks.multiply(flux)
        annulus_data_1d = annulus_data[mask > 0]
        bg_mean, bg_med, bg_std = sigma_clipped_stats(annulus_data_1d)
        background = bg_med * areas
        source = phot - background
        noise = np.sqrt((bg_std * areas)**2 + photerr**2)
        aper_sn = source / noise
        idx_max = argrelextrema(aper_sn, np.greater)[0]
        r_sn_max = sma[idx_max][0] if len(idx_max) else np.nan
        r_extract.append(r_sn_max)
        plt.axvline(x=r_sn_max, c="k", ls="--")
        plt.plot(sma, aper_sn, "o-")
        plt.plot(sma[idx_max], aper_sn[idx_max], "xr", ms=10)
        plt.axvline(rkron, ls="--", c="y", label="$R=1R_{kron}$")
        plt.axvline(n_in * rkron, ls="-.", c="y", label=f"$R={n_in}R_{{kron}}$")
        plt.axvline(n_out * rkron, ls=":", c="y", label=f"$R={n_out}R_{{"
                                                        f"kron}}$")
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


if __name__ == "__main__":
    wdir = os.path.join(context.home_dir, "data/fieldA")
    table_name = os.path.join(wdir, "source_cat.fits")
    table = Table.read(table_name)

    img_name = os.path.join(context.home_dir,
                            "data/fieldA/sn_fieldA.fits")
    flux = fits.getdata(img_name)
    fluxerr = fits.getdata(img_name, ext=2)
    table = calc_best_aper(flux, fluxerr, table, wdir=wdir)
    # fluxvar = fluxerr ** 2
    #
    # sn = flux / fluxerr
    # vmin = np.nanpercentile(sn, 2)
    # vmax = np.nanpercentile(sn, 99)
    # plt.imshow(sn, vmin=vmin, vmax=vmax, origin="lower")
    # plt.colorbar()
    # plt.show()
    # # Plot image of unsharp maks to visualize apertures
    # unsharp_mask = fits.getdata(os.path.join(wdir, "unsharp_mask.fits"))
    # vmin = np.nanpercentile(unsharp_mask, 50)
    # vmax = np.nanpercentile(unsharp_mask, 98)
    # fig = plt.figure()
    # # ax = plt.subplot(111)
    # # ax.grid(False)
    # # plt.imshow(unsharp_mask, vmin=vmin, vmax=vmax, origin="lower",
    # #            cmap="viridis")
    # # plt.show()
    # # Sky noise estimation
    #
    #     continue
    #     # err = np.sqrt(photerr**2 + (skynoise * areas)**2)
    #     # aper_sn = phot / err
    #     # Finding approximate location of max S/N
    #     idx_max = argrelextrema(aper_sn, np.greater)[0]
    #     if len(idx_max):
    #         r_sn_max = aper_a[idx_max][0]
    #         aper = EllipticalAperture((x0, y0), r_sn_max, r_sn_max * ar,
    #                                   theta=theta)
    #         # aper.plot(color="w", ls="-")
    #     else:
    #         r_sn_max = rkron
    #         aper = EllipticalAperture((x0, y0), r_sn_max, r_sn_max * ar,
    #                                   theta=theta)
    #         # aper.plot(color="r", ls="-")
    #     # annulus_aperture.plot(ls="--")
    #
    #     plt.plot(aper_a, phot / err, "o-")
    #     plt.plot(aper_a[idx_max], aper_sn[idx_max], "xr", ms=10)
    #     plt.show()
    # plt.colorbar()
    # plt.show()