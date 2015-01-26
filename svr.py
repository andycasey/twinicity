# coding: utf-8

""" Top-secret twins project using SVR. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library.
import cPickle as pickle
import os
from math import factorial

# Third-party.
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from sklearn.svm import SVR

# Load in all the data
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

# All-but-one cross-validation
kernel = "linear"
predicted_distance = np.zeros(len(stars))
for i, star in enumerate(stars):

    
    mask = np.ones(len(stars), dtype=bool)
    mask[i] = False

    clf = SVR(kernel=kernel)
    clf.fit(all_stellar_fluxes[mask, :], V_absolute[mask])

    prediction = clf.predict(all_stellar_fluxes[i, :])
    
    # Convert predicted V_absolutes to distances
    predicted_distance[i] = 10**((stars["V"][i] - prediction + 5.)/5)

    print("Predicted distance for star {0} ({1:.0f}/{2:.0f}) is {3:.1f} parsec "
        "(expected {4:.1f} parsec; error of {5:.0f} percent)".format(
            star["Object"], i, num_stars, predicted_distance[i], distance[i],
            100. * (predicted_distance[i] - distance[i])/distance[i]))


fig, ax = plt.subplots(3)
ax[0].scatter(distance, 100. * (predicted_distance - distance)/distance, facecolor="k")
ax[1].scatter(distance, predicted_distance, facecolor="k")

ax[1].set_ylabel("Predicted (parsec)")
ax[2].set_xlabel("Distance (parsec)")
ax[2].set_ylabel("Predicted (parsec)")
ax[0].set_ylabel("Error (%)")

upper = max([ax[1].get_xlim()[1], ax[1].get_ylim()[1]])
ax[0].set_xlim(0, upper)
ax[1].set_xlim(0, upper)
ax[1].set_ylim(0, upper)

ax[2].scatter(distance, predicted_distance, facecolor="k")
ax[2].set_xlim(0, 500)
ax[2].set_ylim(0, 500)

fig.tight_layout()
fig.savefig("SVR-{}.pdf".format(kernel))

with open("SVR-{}-predictions.pickle".format(kernel), "w") as fp:
    pickle.dump((distance, predicted_distance), fp, -1)

del all_stellar_fluxes
