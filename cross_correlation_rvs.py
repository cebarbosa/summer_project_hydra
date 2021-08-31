""" Determination of RVs using cross correlation. """

import os
import glob
import warnings

from astropy.io import fits
from astropy.table import Table, hstack
import numpy as np
import ppxf as ppxf_package
from specutils import Spectrum1D
from astropy.nddata import StdDevUncertainty
from specutils.analysis import correlation
from specutils import SpectrumCollection
from astropy import units as u
from spectres import spectres
from tqdm import tqdm

import context

warnings.simplefilter('ignore', np.RankWarning)

def load_miles_templates():
    """ Load all templates of MILES library that comes with pPXF. """
    ppxf_dir = os.path.dirname(os.path.realpath(ppxf_package.__file__))
    filenames = sorted(glob.glob(ppxf_dir + '/miles_models/Mun1.30*.fits'))
    hdu = fits.open(filenames[0])
    h = hdu[0].header
    wave = h['CRVAL1'] + h['CDELT1'] * np.arange(h['NAXIS1'])
    idx = np.where(wave > 4800)[0]
    templates = []
    for i, filename in enumerate(tqdm(filenames)):
        flux_temp = fits.getdata(filename)
        flux_temp /= np.median(flux_temp)
        noise = StdDevUncertainty(0.01 * flux_temp * u.Jy)
        spec1d = Spectrum1D(spectral_axis=wave[idx] * u.Angstrom,
                            flux=flux_temp[idx] * u.Jy, uncertainty=noise[idx])
        templates.append(spec1d)
    return templates

if __name__ == "__main__":
    templates = load_miles_templates()
    for field in context.fields:
        field_dir = os.path.join(context.home_dir, "data", field)
        cat = Table.read(os.path.join(field_dir, "gc-candidates-catalog.fits"))
        outcat = os.path.join(field_dir, "gc-candidates-rvs.fits")
        specs_dir = os.path.join(field_dir, "spec1D_n2.5_4_rkron")
        rvs = []
        for j, source in enumerate(cat):
            name = f"{field}_{source['NUMBER']:03d}"
            specfile = os.path.join(specs_dir, f"{name}.fits")
            t = Table.read(specfile)
            t["flux"] = t["flux_source"] - t["flux_annulus"]
            t["fluxerr"] = np.sqrt(t["fluxerr_source"]**2 +
                                   t["fluxerr_annulus"]**2)
            if np.nanmedian(t["flux"]) <= 0:
                rvs.append(np.nan)
                continue
            twave = templates[0].spectral_axis
            flux = spectres(twave.value, t["wave"], t["flux"], fill=0,
                            verbose=False) * u.Jy
            fluxerr = spectres(twave.value, t["wave"], t["fluxerr"], fill=0,
                            verbose=False) * u.Jy
            ospec = Spectrum1D(spectral_axis=twave, flux=flux,
                               uncertainty=StdDevUncertainty(fluxerr))
            corrs = []
            for i, template in enumerate(tqdm(templates)):
                corr, lag = correlation.template_correlate(ospec, template)
                corrs.append(corr)
            corrs = np.array(corrs)
            idx = np.where(corrs == np.nanmax(corrs))
            rv = lag[idx[1]]
            rvs.append(rv.value[0])
        cat["velocity"] = rvs * u.km / u.s
        cat = cat[~np.isnan(rvs)]
        cat.write(outcat, overwrite=True)
