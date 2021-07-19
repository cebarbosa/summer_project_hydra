import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from photutils.aperture import aperture_photometry
from photutils.aperture import EllipticalAperture
import matplotlib.pyplot as plt

import context


if __name__ == "__main__":
    table_name = os.path.join(context.home_dir, "data/fieldA/source_cat.fits")
    table = Table.read(table_name)
    radii = np.linspace(0.1, 1., 10)
    img_name = os.path.join(context.home_dir,
                            "data/fieldA/signal_noise.fits")
    signal = fits.getdata(img_name)
    noise = fits.getdata(img_name, ext=2)
    sn = signal / noise
    vmin = np.nanpercentile(sn, 2)
    vmax = np.nanpercentile(sn, 99)
    plt.imshow(sn, vmin=vmin, vmax=vmax, origin="lower")
    plt.colorbar()
    plt.show()
    for t in table:
        x0 = t["X_IMAGE"] -1
        y0 = t["Y_IMAGE"] - 1
        a = t["KRON_RADIUS"] * t["A_IMAGE"]
        b = t["KRON_RADIUS"] * t["B_IMAGE"]
        theta = np.deg2rad(t["THETA_IMAGE"])
        apertures = [EllipticalAperture((x0, y0), a * r, b * r, theta=theta)
                     for r in radii]
        # aper = EllipticalAperture((x0, y0), a, b, theta=theta)
        # aper.plot(color="w", ls="--")
        phot_table = aperture_photometry(signal, apertures)
        phot = np.array([phot_table[f"aperture_sum_{i}"].data[0] for i in
                          range(len(radii))])
        plt.plot(a * radii, phot, "o-")
        plt.show()