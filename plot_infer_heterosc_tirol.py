"""
Elevation error map
===================

Digital elevation models have a precision that can vary with terrain and instrument-related variables. Here, we
rely on a non-stationary spatial statistics framework to estimate and model this variability in elevation error,
using terrain slope and maximum curvature as explanatory variables, with stable terrain as an error proxy for moving
terrain.

**Reference**: `Hugonnet et al. (2022) <https://doi.org/10.1109/jstars.2022.3188922>`_, Figs. 4 and S6–S9. Equations 7
or 8 can be used to convert elevation change errors into elevation errors.
"""
import geoutils as gu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# sphinx_gallery_thumbnail_number = 1
import xdem

# import os
# os.environ['USE_PYGEOS'] = '0'
import geopandas

# %%
# these are the warped, coregesiterd  versions:
fn1969_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1969_coreg.tif'
fn1997_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1997_coreg.tif'
fn2006_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/2006_coreg.tif'
fn2017_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'

dh_1969 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19691997.tif'
dh_1997 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19972006.tif'
dh_2006 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_20062017.tif'

# GI1 = 'mergedGI1_2.shp'
# GI2 = 'mergedGI2_2.shp'
# GI3 = 'mergedGI3_2.shp'
GI1 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/mergedOutlines/HEFtotal_mergedGI1_3.shp'
GI2 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/mergedOutlines/HEFtotal_mergedGI2_3.shp'
GI3 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/mergedOutlines/HEFtotal_mergedGI3_3.shp'
# GI5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/mergedOutlines/HEFtotal_mergedGI5_3.shp'

# %%
def heterosc(fn_dem, fn_dh, fn_outlines, figname, tablename):
    ref_dem = xdem.DEM(fn_dem)
    dh = xdem.DEM(fn_dh)
    glacier_outlines = gu.Vector(fn_outlines)
    # We derive the terrain slope and maximum curvature from the reference DEM.
    slope, maximum_curvature = xdem.terrain.get_terrain_attribute(ref_dem, attribute=["slope", "maximum_curvature"])

    # %%
    # Then, we run the pipeline for inference of elevation heteroscedasticity from stable terrain:
    errors, df_binning, error_function = xdem.spatialstats.infer_heteroscedasticity_from_stable(
        dvalues=dh, list_var=[slope, maximum_curvature], list_var_names=["slope", "maxc"], unstable_mask=glacier_outlines
    )

    # %%
    # The first output corresponds to the error map for the DEM (:math:`\pm` 1\ :math:`\sigma` level):
    fig, ax = plt.subplots(1,1)
    im = errors.show(vmin=0, vmax=7, cmap="Reds", ax=ax, )#cbar_title=r"Elevation error (1$\sigma$, m)")
    # cb = plt.colorbar(im)
    # cb.set_label(r"Elevation error (1$\sigma$, m)")
    fig.savefig('outputXdem/'+figname+'.png')
    # %%
    # The second output is the dataframe of 2D binning with slope and maximum curvature:
    print(df_binning)

    # %%
    # The third output is the 2D binning interpolant, i.e. an error function with the slope and maximum curvature
    # (*Note: below we divide the maximum curvature by 100 to convert it in* m\ :sup:`-1` ):
    dfheterosc = []
    for slope, maxc in [(0, 0), (30, 0), (40, 0), (0, 5), (30, 5), (40, 5)]:
        print(
            "Error for a slope of {:.0f} degrees and"
            " {:.2f} m-1 max. curvature: {:.4f} m".format(slope, maxc / 100, error_function((slope, maxc)))
        )
        dftemp = [slope, maxc / 100, error_function((slope, maxc))]
        dfheterosc.append(dftemp)

    df = pd.DataFrame(dfheterosc, columns=['slope', 'maxc', 'error'])
    print(df)
    df.to_csv('outputXdem/'+tablename+'.csv')




heterosc(fn1997_5al, dh_1969, GI1, '19691997_heterosc', '19691997_heterosc')
heterosc(fn2006_5al, dh_1997, GI1, '19972006_heterosc', '19972006_heterosc')
heterosc(fn2017_5al, dh_2006, GI1, '20062017_heterosc', '20062017_heterosc')

plt.show()

# %%
# This pipeline will not always work optimally with default parameters: spread estimates can be affected by skewed
# distributions, the binning by extreme range of values, some DEMs do not have any error variability with terrain (e.g.,
# terrestrial photogrammetry). **To learn how to tune more parameters and use the subfunctions, see the gallery example:**
# :ref:`sphx_glr_advanced_examples_plot_heterosc_estimation_modelling.py`!
