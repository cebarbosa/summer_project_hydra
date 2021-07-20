import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
from photutils.aperture import aperture_photometry
from photutils.aperture import EllipticalAperture, EllipticalAnnulus, CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import context

if __name__ == "__main__":
    
    table_name = os.path.join(context.home_dir, "fieldA/source_cat.fits")
    table = Table.read(table_name)
    radii = np.linspace(0.1, 1., 10) # try different apertures to find best S/N
    img_name = os.path.join(context.home_dir,"fieldA/signal_noise.fits")
    signal = fits.getdata(img_name)
    noise = fits.getdata(img_name, ext=2)
    sn = signal / noise
    vmin = np.nanpercentile(sn, 2)
    vmax = np.nanpercentile(sn, 99)
    aper_sum_bkgsubs = []
    
    plt.figure(figsize=(12, 12))
    plt.imshow(sn, vmin=vmin, vmax=vmax, origin="lower")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('S/N')
    
    for t in table:
        
        x0 = t["X_IMAGE"] -1 # offset in pixels
        y0 = t["Y_IMAGE"] -1
        a = t["KRON_RADIUS"] * t["A_IMAGE"] # A & B are normalized
        b = t["KRON_RADIUS"] * t["B_IMAGE"]
        theta = np.deg2rad(t["THETA_IMAGE"]) # orientation
        
        # Try apertures of different sizes to find best S/N
        #apertures = [EllipticalAperture((x0, y0), a * r, b * r, theta=theta) for r in radii]
        #phot_table = aperture_photometry(signal, apertures)
        #phot = np.array([phot_table[f"aperture_sum_{i}"].data[0] for i in range(len(radii))]) # signal
        #plt.plot(a * radii, phot, "o-")
        #plt.show()
        
        aper = EllipticalAperture((x0, y0), a, b, theta=theta)
        annulus_aperture = EllipticalAnnulus((x0, y0), a_in=a*1.1, a_out=a*1.3, 
                                             b_out=1.3*b, b_in=1.1*b, theta=theta)
        ap_patches = aper.plot(color='white', ls="--", lw=1, label='Photometry aperture')
        ann_patches = annulus_aperture.plot(color='red', ls="--", lw=1, label='Background annulus')
        
        # For now just use radii[8]
        aperture = EllipticalAperture((x0, y0), a*radii[8], b*radii[8], theta=theta)
        phot = aperture_photometry(signal, aperture)
        annulus_mask = annulus_aperture.to_mask(method='center')
        annulus_data = annulus_mask.multiply(signal)
        mask = annulus_mask.data
        annulus_data_1d = annulus_data[mask > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        aper_bkg = bkg_median * aperture.area
        aper_sum_bkgsub = phot['aperture_sum'] - aper_bkg
        aper_sum_bkgsubs.append(aper_sum_bkgsub)

    #print(aper_sum_bkgsubs)
    # There are some NaN values for those on the edge
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.savefig("NGC3311_Apertures.jpg", bbox_inches='tight')
