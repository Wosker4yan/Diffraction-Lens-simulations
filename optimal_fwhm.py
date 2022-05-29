# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests form Scalar_masks_XY"""

from diffractio import sp, nm, plt, np, mm, degrees, um
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from tqdm import tqdm
import math


# FIND THE OPTIMAL FWHM AT THE RIGHT DISTANCES

def normalize(vec):
    """Normalizes an electric field from its maximum, use for visualization of the field inside the index profile
    Args:
        vec (vector): Field to be normalized
    Returns:
        norm (Array): Normalized field
    """

    vec = np.asarray(vec)
    norm = np.squeeze(vec / vec.max())

    return norm


# Functions to calculate the FWHM
def fwhm(x, y, height=0.5):
    """FWHM of a given field.

    Args:
        x (list or array): x coordinate.
        y (list or array): y coordinate.
        height (float): height of the width.
    Returns:
        FWHM of the field (float)
    """

    height_half_max = np.max(y) * height
    index_max = np.argmax(y)

    x_low = np.interp(height_half_max, y[:index_max], x[:index_max])
    x_high = np.interp(height_half_max, np.flip(y[index_max:]), np.flip(x[index_max:]))

    return x_high - x_low


def lin_interp(x, y, i, half):
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def half_max_x(x, y):
    half = max(y) / 2
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


if __name__ == '__main__':

    x0 = np.linspace(-3.0 * mm, 6.0 * mm, 8000)
    z0 = np.linspace(0 * mm, 11.0 * mm, 8000)

    wavelength = 1.55 * um
    # angles = np.linspace(0, 25,6)
    angles = np.linspace(0, 28, 11)
    distances =  np.linspace(9.9,11,10)
    FWHM = []
    Z = []
    X = []
    intensity = []
    indexx = []
    detector_distance = np.linspace(5, 12, 50)
    phase = []
    plotting = True

    #### define lens shape #####
    lens = Scalar_mask_XZ(x0, z0, wavelength, n_background=1 + 0j, info='')

    f1 = "A1*(self.X-x0)**2+A2*(self.X-x0)+A3+y0"
    f2 = "A4+y0"
    v_globals = {
        'A1': 1E-07 * mm,
        'A2': 3E-07 * mm,
        'A3': 0 * mm,
        'A4': 2.7051 * mm,
        'x0': 0 * mm,
        'y0': 0.5 * mm
    }

    index = 1.51680 + 0j
    lens.mask_from_function(
        r0=(0 * mm, 0.5 * mm),
        refraction_index=index,
        f1=f1,
        f2=f2,
        z_sides=(-2.248 * mm, 2.248 * mm),
        angle=0 * degrees,
        v_globals=v_globals)

    lens.slit(r0=(0, 1000 * um),
              aperture=4.496 * mm,
              depth=100 * um,
              refraction_index=1 + 2j)

    lens.filter_refraction_index(type_filter=2, pixels_filtering=25)
    light = Scalar_source_X(x0, wavelength)
    for d in tqdm(distances):
        FWHM.append([])
        for angle in tqdm(angles):
            light.plane_wave(A=1, theta=angle * degrees)
            lens.incident_field(light)
            print("The incident angle is " + str(angle))

            lens.clear_field()
            lens.BPM(verbose=False)

        ##### search focus of the lens (probably it is a max-intensity search) #####

        #########

            lens_profile = lens.profile_transversal(kind='intensity', z0=d*mm, draw=False)
            #Add a monitor here to show where is the detector
            hmx = fwhm(x0, lens_profile, height=0.5)

            FWHM[-1].append(hmx)


            print("FWHM:{:.3f}".format(hmx))
            if plotting:
                lens.draw(kind='intensity', logarithm=1, normalize=None, draw_borders=True, colorbar_kind='vertical')

                plt.axvline(d*mm,
                            ymin=0,
                            ymax=1,
                            color='#ff4a4a',
                            label='source',
                            linewidth=4.0)

                plt.show()

    fig2, ax2 = plt.subplots()
    ax2.set_title('Optimal FWHM')
    for i, trans in enumerate(FWHM):
        ax2.plot( angles, trans, label=distances[i])

    ax2.set_xlabel('angles [degree]')
    ax2.legend()
    ax2.set_ylabel('fwhm [um]')

    fig2.tight_layout()
    fig2.show()









