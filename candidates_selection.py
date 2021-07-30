import os
import warnings

import numpy as np
from astropy.table import Table, hstack
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astropy.utils.exceptions import AstropyWarning

import context

warnings.simplefilter('ignore', category=AstropyWarning)

def select_candidates(table, seeing, dflux_radius=0.5, field=None, redo=False,
                      ellip_max=0.3):
    """ Narrow the number of GC candidates according to their morphology. """
    outtable = os.path.join(os.getcwd(), "gc-candidates-catalog.fits")
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
    plt.show()
    plt.close()
    return candidates

if __name__ == "__main__":
    redo=True
    for i, field in enumerate(context.fields):
        wdir = os.path.join(context.home_dir, f"data/{field}")
        os.chdir(wdir) # Changing to working directory
        table = Table.read("source-catalog.fits")
        # Converting table to arcsec
        table["FLUX_RADIUS"] *= context.PS
        # Offset for magnitude
        table["MAG_AUTO"] += 25
        # Select systems that are more likely GC candidates
        candidates = select_candidates(table, context.seeing[i] / 2,
                                       field=field, redo=redo)