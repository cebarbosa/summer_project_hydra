""" Calculating SNR of the GC candidates """
import os

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from spectres import spectres
from tqdm import tqdm

import context
import misc

def der_snr(flux, axis=1, full_output=False):
    """ Calculates the S/N ratio of a spectra.

    Translated from the IDL routine der_snr.pro """
    signal = np.nanmean(flux, axis=axis)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.* flux - \
           np.roll(flux, 2, axis=axis) - np.roll(flux, -2, axis=axis)), \
           axis=axis)
    if full_output:
        return signal, noise, signal / noise
    return signal / noise

def snr_err_propagation(flux, fluxerr, axis=1):
    signal = np.nanmean(flux, axis=axis)
    noise = np.sqrt(np.nanmean(fluxerr**2, axis=axis))
    return signal / noise

def skylines_mask(wave, dw=4):
    """ Produces a mask for sky and telluric lines. """
    skylines = np.array([4792, 4860, 4923, 5071, 5239, 5268, 5577, 5889.99,
                         5895, 5888, 5990, 5998, 6300, 6363, 6386, 6562,
                         6583, 6717, 6730, 7246, 8286, 8344, 8430, 8737,
                         8747, 8757, 8767, 8777, 8787, 8797, 8827, 8836, 8919,
                         9310])
    mask = np.zeros(len(wave))
    for line in skylines:
        idx = np.where((wave >= line - dw) & (wave <= line + dw))
        mask[idx] = 1
    return mask

def load_field_data(cat, owave, r_inn=2.5, r_out=4):
    """ Load spectra from a catalog. """
    specs, specerrs, masks = [], [], []
    for i, obj in enumerate(tqdm(cat, desc="Preparing input spectra")):
        specfile = f"spec1D_n{r_inn}_{r_out}_rkron/{obj['FIELD']}_" \
                   f"{obj['NUMBER']:03d}.fits"
        t = Table.read(specfile)
        v = obj["velocity"] * u.km / u.s
        c = const.c.to(u.km / u.s)
        gamma = np.sqrt((1 + v / c) / (1 - v / c))
        if i == 0:
            wave = t["wave"].data
            mask = skylines_mask(wave)
        wrest = wave / gamma
        spec = np.column_stack([t["flux_source"].data,
                                t["flux_annulus"].data]).T
        specerr = np.column_stack([t["fluxerr_source"].data,
                                t["fluxerr_annulus"].data]).T
        # # Put spectrum to rest-frame and rebinning to common wavelength array
        spec, specerr = spectres(owave, wrest, spec, spec_errs=specerr, fill=0,
                                 verbose=False)
        omask = spectres(owave, wrest, mask, fill=0, verbose=False)
        specs.append(spec)
        specerrs.append(specerr)
        masks.append(omask)
    specs = np.stack(specs, axis=1) # 3D array
    specerrs = np.stack(specerrs, axis=1)
    masks = np.stack(masks, axis=0)
    return specs, specerrs, masks

def systems_with_emission_lines(field):
    """ List of spectra with emission lines.

    List is produced from visual classification. """
    if field == "fieldA":
        sources = ["002", "012", "016", "019", "033", "039", "041", "063",
                   "079", "113", "115", "161", "177", "181", "194", "215",
                   "225", "249", "250", "254", "263"]
    elif field == "fieldB":
        sources =  ["015","018","021","025","035","041","049","058","059","063",
               "064","073","077",
            "085","086","088","096","098","107","114","119","123","124",
                 "128", "131","140"]
    elif field == "fieldC":
        sources = ["020","022","049","063","091","095","100","109","114","116",
               "126","127"]
    else:
        return []
    return [int(_) for _ in sources]

def sn_stack(sn_wmin=5500, sn_wmax=7000, vmin=2292, vmax=5723):
    """ Pipeline for processing the data. """
    fields = context.fields[:-1]
    flux, fluxerr, catalog, masks = [], [], [], []
    # Reads field A cube to set output wavelength array
    cubename = os.path.join(context.home_dir, "data", "fieldA",
                            f"NGC3311_FieldA_DATACUBE_COMBINED.fits")
    bunit = np.power(10., -20) * u.erg / u.s / u.cm**2 / u.Angstrom
    wave = misc.array_from_header(cubename)
    print("=" * 80)
    print("Preparing spectra for stacking.")
    print("=" * 80)
    for i, field in enumerate(fields):
        wdir = os.path.join(context.home_dir, "data", field)
        os.chdir(wdir)
        cat = Table.read("gc-candidates-rvs.fits")
        ########################################################################
        # Use only GC candidates within cluster
        idx = np.where((cat["velocity"] >= vmin) & (cat["velocity"] <= vmax))[0]
        cat = cat[idx]
        ########################################################################
        # Remove systems with emission lines
        em_systems = systems_with_emission_lines(field)
        idx = np.in1d(cat["NUMBER"], em_systems)
        cat = cat[~idx]
        ########################################################################
        f, ferr, m = load_field_data(cat, wave)
        flux.append(f)
        fluxerr.append(ferr)
        catalog.append(cat)
        masks.append(m)
    print("=" * 80)
    print("Stacking spectra")
    print("=" * 80)
    # Arrays and catalog for all fields
    flux3D = np.concatenate(flux, axis=1)
    fluxerr3D = np.concatenate(fluxerr, axis=1)
    # Estimating halo-extracted spectrum for S/N
    flux = -np.diff(flux3D, axis=0)[0]
    fluxerr = np.sqrt(np.sum(fluxerr3D**2, axis=0))
    mask = np.concatenate(masks, axis=0)
    # Use only sector between wmin and wmax to estimate S/N
    wave_sn = np.arange(sn_wmin, sn_wmax + 1)
    flux_sn, fluxerr_sn = spectres(wave_sn, wave, flux,
                                   spec_errs=fluxerr ,
                                   fill=0, verbose=False)
    # # Sorting spectra by S/N
    snr = snr_err_propagation(flux_sn, fluxerr_sn, axis=1)
    idx = np.argsort(snr)[::-1]
    flux = flux[idx, :]
    fluxerr = fluxerr[idx, :]
    mask = mask[idx, :]
    fluxerr_sn = fluxerr_sn[idx, :]
    flux_sn = flux_sn[idx, :]
    # Choose stack with best S/N
    flux_cum = np.cumsum(flux, axis=0)  # Cumulative flux array
    fluxerr_cum = np.sqrt(np.cumsum(fluxerr ** 2, axis=0))
    flux_sn_cum = np.cumsum(flux_sn, axis=0)
    fluxerr_sn_cum = np.sqrt(np.cumsum(fluxerr_sn**2, axis=0))
    snc = snr_err_propagation(flux_sn_cum, fluxerr_sn_cum)
    imax = np.argmax(snc)
    stack = flux_cum[imax]
    stackerr = fluxerr_cum[imax]
    # Preparing mask
    stack_mask = np.where(mask.mean(axis=0) > 0.25, 1, 0)
    # Saving the results
    t = Table([wave, stack * bunit, stackerr * bunit, stack_mask],
              names=["wave", "flux", "fluxerr", "mask"])
    t.write(os.path.join(context.home_dir, "data",
                         "stacked_spectrum_gcs.fits"), overwrite=True)
    pmask = np.where(stack_mask == 1, np.nan, 1)
    # Make plot
    plt.plot(np.arange(len(snc))+1, snc, label="Error propagation")
    plt.xlabel("Number of spectra")
    plt.ylabel("S/N")
    plt.axvline(x=imax+1, c="r", ls="--")
    print(f"Maximum S/N: {snc[imax]:.1f}")
    print(f"Number of spectra stacked: {imax+1}")
    print("-" * 80)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(snc))+1, snc)
    plt.xlabel("Number of spectra")
    plt.ylabel("S/N")
    plt.axvline(x=imax+1, c="r", ls="--")
    plt.subplot(2, 1, 2)
    plt.plot(wave * pmask, stack, label="Stacked spectrum")
    plt.ylim(np.nanpercentile(stack, 2.5), None)
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Flux")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sn_stack()