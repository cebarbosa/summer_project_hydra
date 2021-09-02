""" Calculating SNR of the GC candidates """
import os

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from spectres import spectres
from tqdm import tqdm
import paintbox as pb

import context

def der_snr(flux, axis=0, full_output=False):
    """ Calculates the S/N ratio of a spectra.

    Translated from the IDL routine der_snr.pro """
    signal = np.nanmean(flux, axis=axis)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.* flux - \
           np.roll(flux, 2, axis=axis) - np.roll(flux, -2, axis=axis)), \
           axis=axis)
    if full_output:
        return signal, noise, signal / noise
    return signal / noise

def snr_err_propagation(flux, fluxerr):
    signal = np.nanmean(flux, axis=1)
    noise = np.sqrt(np.nanmean(fluxerr**2, axis=1))
    return signal / noise

def load_field_data(cat, r_inn=2.5, r_out=4):
    """ Load spectra from a catalog. """
    specs, specerrs = [], []
    for obj in tqdm(cat, desc="Preparing input spectra"):
        specfile = f"spec1D_n{r_inn}_{r_out}_rkron/{obj['FIELD']}_" \
                   f"{obj['NUMBER']:03d}.fits"
        t = Table.read(specfile)
        v = obj["velocity"] * u.km / u.s
        beta = np.sqrt((1 + v / const.c) / (1 - v / const.c))
        wave = t["wave"].data
        wrest = wave / beta
        spec = (t["flux_source"] - t["flux_annulus"]).data
        specerr = np.sqrt(t["fluxerr_source"]**2 + t["fluxerr_annulus"]**2).data
        # Put spectrum to rest-frame
        spec, specerr = spectres(wrest, wave, spec, spec_errs=specerr, fill=0,
                                 verbose=False)
        specs.append(spec)
        specerrs.append(specerr)
    specs = np.array(specs)
    specerrs = np.array(specerrs)
    return wave, specs, specerrs

def sn_stack(sn_wmin=5000, sn_wmax=6000, vmin=2292, vmax=5723):
    """ Pipeline for processing the data. """
    flux, fluxerr, catalog = [], [], []
    for i, field in enumerate(context.fields):
        print(f"Loading data for {field}")
        wdir = os.path.join(context.home_dir, "data", field)
        os.chdir(wdir)
        cat = Table.read("gc-candidates-rvs.fits")
        # Use only GC candidates withing cluster
        idx = np.where((cat["velocity"] >= vmin) & (cat["velocity"] <= vmax))[0]
        cat = cat[idx]
        w, f, ferr = load_field_data(cat)
        if i == 0:
            wave = w
            flux.append(f)
            fluxerr.append(ferr)
        else:
            newf, newferr = spectres(wave, w, f, spec_errs=ferr, fill=0,
                                     verbose=False)
            flux.append(newf)
            fluxerr.append(newferr)
        catalog.append(cat)
    # Arrays and catalog for all fields
    flux = np.vstack(flux)
    fluxerr = np.vstack(fluxerr)
    catalog = vstack(catalog)
    # Getting indices for wavelength range
    wave_sn = np.arange(sn_wmin, sn_wmax + 1)
    flux_sn, fluxerr_sn = spectres(wave_sn, wave, flux, spec_errs=fluxerr,
                                   fill=0, verbose=False)
    snr1 = der_snr(flux_sn, axis=1)
    snr2 = snr_err_propagation(flux_sn, fluxerr_sn)
    labels = ["DER_SNR", "Error propagation"]
    for i, snr in enumerate([snr1, snr2]):
        idx = np.argsort(snr)[::-1]
        sflux = flux[idx, :] # Sorted array
        sfluxerr = fluxerr[idx, :]
        cflux = np.cumsum(sflux, axis=0) # Cumulative flux array
        cfluxerr = np.sqrt(np.cumsum(sfluxerr**2, axis=0))
        cflux_sn, cfluxerr_sn = spectres(wave_sn, wave, cflux,
                                         spec_errs=cfluxerr,
                                   fill=0, verbose=False)
        if i == 1:
            snc = der_snr(cflux_sn, axis=1)
        else:
            snc = snr_err_propagation(cflux_sn, cfluxerr_sn)
        imax = np.argmax(snc)
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(snc))+1, snc, label=labels[i])
        plt.xlabel("Number of spectra")
        plt.ylabel("S/N")
        plt.axvline(x=imax+1, c=f"C{i}", ls="--")
        plt.subplot(2, 1, 2)
        plt.plot(wave, cflux[imax], label=labels[i])
        plt.ylim(20000, 40000)
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Flux")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sn_stack()