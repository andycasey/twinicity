# coding: utf-8

""" Top-secret twins project. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library.
import os
from itertools import combinations
from glob import glob
from math import factorial

# Third-party.
import numpy as np
from astropy.table import Table
from sklearn.svm import SVR


spectra_filename_format = "data/harps/RV_RES_{}.txt"
stars = Table.read("giants_harps_unique_goodPlx_sorted.dat", format="ascii",
    names="Object RA DEC B V J Ks parallax e_parallax B-V Nobs".split())

# Calculate intrinsic lumonsities from their parallax.
# A 10e-3 scale was quoted for the parallaxes, but that is wrong. 
# Distances in parsecs.
distance = 1./(stars["parallax"] * 10e-4)
mu = 5 * np.log10(distance) - 5
V_absolute = stars["V"] - mu
B_absolute = stars["B"] - mu
J_absolute = stars["J"] - mu
Ks_absolute = stars["Ks"] - mu

# We assume the same number of pixels in all spectra
num_stars = len(stars)
num_pixels = 282422 # Calculated externally.
num_combinations = factorial(num_stars)/(2 * factorial(num_stars - 2))


# For each star, compare it to all other stars.
print("Warning: not doing per-pixel interpolation (yet??)")

fluxes_filename = "fluxes.memmap"
if not os.path.exists(fluxes_filename):

    all_stellar_fluxes = np.memmap(fluxes_filename, dtype="float32", mode="w+",
        shape=(num_stars, num_pixels, ))
    for i, star in enumerate(stars):
        print("Loading star #{0:.0f}/{1:.0f} {2}".format(i + 1, num_stars,
            star["Object"]))

        all_stellar_fluxes[i, :] = Table.read(spectra_filename_format.format(
            star["Object"]), format="ascii")["flux"][:num_pixels]

else:
    all_stellar_fluxes = np.memmap(fluxes_filename, dtype="float32", mode="r")
    all_stellar_fluxes = all_stellar_fluxes.reshape((num_stars, num_pixels))


"""
flux_ratios_filename = "differences.memmap"
combinations_filename = "combinations.pickle"
if not os.path.exists(flux_ratios_filename) \
or not os.path.exists(combinations_filename):

    ratios = np.memmap(flux_ratios_filename, dtype="float32", mode="w+",
        shape=(num_combinations, num_pixels))

    all_combinations = combinations(range(num_stars), 2)
    with open(combinations_filename, "w") as fp:
        pickle.dump(all_combinations, fp, -1)

    for i, (index_a, index_b) in enumerate(all_combinations):

        # Load the stuff.
        star_a, star_b = stars[index_a], stars[index_b]
        print("Comparison {0:.0f}/{1:.0f} between {2} and {3}".format(
            i, num_combinations, star_a["Object"], star_b["Object"]))

        ratios[i, :] = \
            all_stellar_fluxes[index_a, :]/all_stellar_fluxes[index_b, :]

else:
    ratios = np.memmap(flux_ratios_filename, dtype="float32", mode="r")
    ratios = ratios.reshape((num_combinations, num_pixels))
    with open(combinations_filename, "r") as fp:
        all_combinations = pickle.load(fp)
"""


# Cross-validation of the sample.
#for i, star in enumerate(stars):

#    cv_indices = [_ for _ in all_combinations if i not in _]


# sum(pixels * coefficients) = luminosity
# pixels need to actually be ratios of pixels between one star and another
# especially due to the normalisation problem.

# training_star_luminosity * sum(pixels_of_new_star/pixels_of_training_star * coefficients) = luminosity

# but that means we get an estimate of the luminosity from every star in the training set
# luminosity_a/luminosity_b * sum(pixels_of_star_b / pixels_of_star_a  * coefficients) = 1
# coefficeints = (pixels_of_star_a / pixels_of_star_b) * luminosity_b/luminosity_a







del all_stellar_fluxes, ratios
