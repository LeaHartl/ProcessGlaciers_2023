"""
Standardization for stable terrain as error proxy
=================================================

Digital elevation models have both a precision that can vary with terrain or instrument-related variables, and
a spatial correlation of errors that can be due to effects of resolution, processing or instrument noise.
Accouting for non-stationarities in elevation errors is essential to use stable terrain as a proxy to infer the
precision on other types of terrain and reliably use spatial statistics (see :ref:`spatialstats`).

Here, we show an example of standardization of the data based on terrain-dependent explanatory variables
(see :ref:`sphx_glr_basic_examples_plot_infer_heterosc.py`) and combine it with an analysis of spatial correlation
(see :ref:`sphx_glr_basic_examples_plot_infer_spatial_correlation.py`) .

**Reference**: `Hugonnet et al. (2022) <https://doi.org/10.1109/jstars.2022.3188922>`_, Equation 12.
"""
import os
#os.environ['USE_PYGEOS'] = '0'
import geopandas
import geoutils as gu

# sphinx_gallery_thumbnail_number = 4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shapely.geometry
import xdem
from xdem.spatialstats import nmad

# %%
# We start by estimating the elevation heteroscedasticity and deriving a terrain-dependent measurement error as a function of both
# slope and maximum curvature, as shown in the :ref:`sphx_glr_basic_examples_plot_infer_heterosc.py` example.

# these are the warped, coregistered  versions:
fn1969_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1969_coreg.tif'
fn1997_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1997_coreg.tif'
fn2006_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/2006_coreg.tif'
fn2017_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'

dh_1969 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19691997.tif'
dh_1997 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19972006.tif'
dh_2006 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_20062017.tif'

GI1 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI1_3.shp'
GI2 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI2_3.shp'
GI3 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI3_3.shp'
GI5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI5_3.shp'


def stand_errors_all(demy1_fn, demy2_fn, dh_fn, outlines, statsfn, missing):

    ref_dem = xdem.DEM(demy2_fn)
    dh = xdem.DEM(dh_fn)
    glacier_outlines = gu.Vector(outlines)
    mask_glacier = glacier_outlines.create_mask(dh)

    # Compute the slope and maximum curvature
    slope, planc, profc = xdem.terrain.get_terrain_attribute(
        dem=ref_dem, attribute=["slope", "planform_curvature", "profile_curvature"]
    )

    # Remove values on unstable terrain
    dh_arr = dh[~mask_glacier].filled(np.nan)
    slope_arr = slope[~mask_glacier].filled(np.nan)
    planc_arr = planc[~mask_glacier].filled(np.nan)
    profc_arr = profc[~mask_glacier].filled(np.nan)
    maxc_arr = np.maximum(np.abs(planc_arr), np.abs(profc_arr))

    #Remove large outliers
    dh_arr[np.abs(dh_arr) > 4 * xdem.spatialstats.nmad(dh_arr)] = np.nan

    # Define bins for 2D binning
    custom_bin_slope = np.unique(
        np.concatenate(
            [
                np.nanquantile(slope_arr, np.linspace(0, 0.95, 20)),
                np.nanquantile(slope_arr, np.linspace(0.96, 0.99, 5)),
                np.nanquantile(slope_arr, np.linspace(0.991, 1, 10)),
            ]
        )
    )

    custom_bin_curvature = np.unique(
        np.concatenate(
            [
                np.nanquantile(maxc_arr, np.linspace(0, 0.95, 20)),
                np.nanquantile(maxc_arr, np.linspace(0.96, 0.99, 5)),
                np.nanquantile(maxc_arr, np.linspace(0.991, 1, 10)),
            ]
        )
    )

    # Perform 2D binning to estimate the measurement error with slope and maximum curvature
    df = xdem.spatialstats.nd_binning(
        values=dh_arr,
        list_var=[slope_arr, maxc_arr],
        list_var_names=["slope", "maxc"],
        statistics=["count", np.nanmedian, nmad],
        list_var_bins=[custom_bin_slope, custom_bin_curvature],
    )

    df.to_csv(statsfn+'df_stats2.csv')
    print('wrote df stats')
    # Estimate an interpolant of the measurement error with slope and maximum curvature
    slope_curv_to_dh_err = xdem.spatialstats.interp_nd_binning(
        df, list_var_names=["slope", "maxc"], statistic="nmad", min_count=30
    )
    maxc = np.maximum(np.abs(profc), np.abs(planc))

    # # Estimate a measurement error per pixel
    dh_err = slope_curv_to_dh_err((slope.data, maxc.data))
    # with open(statsfn+'dh_err2.pickle', 'wb') as handle:
    #     pickle.dump(dh_err, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print('wrote pickle')

    # # %%
    # # Using the measurement error estimated for each pixel, we standardize the elevation differences by applying
    # # a simple division:

    z_dh = dh.data / dh_err

    # %%
    # We remove values on glacierized terrain and large outliers.
    z_dh.data[mask_glacier.data] = np.nan
    z_dh.data[np.abs(z_dh.data) > 4] = np.nan

    # # # %%
    # # # We perform a scale-correction for the standardization, to ensure that the standard deviation of the data is exactly 1.
    print(f"Standard deviation before scale-correction: {nmad(z_dh.data):.1f}")
    scale_fac_std = nmad(z_dh.data)
    z_dh = z_dh / scale_fac_std
    print(f"Standard deviation after scale-correction: {nmad(z_dh.data):.1f}")
    # # these are just some checks and plots to see if it works, uncomment if needed.
    # plt.figure(figsize=(8, 5))
    # plt_extent = [
    #     ref_dem.bounds.left,
    #     ref_dem.bounds.right,
    #     ref_dem.bounds.bottom,
    #     ref_dem.bounds.top,
    # ]
    # ax = plt.gca()
    # glacier_outlines.ds.plot(ax=ax, fc="none", ec="tab:gray")
    # ax.plot([], [], color="tab:gray", label="2006 outlines")
    # plt.imshow(z_dh.squeeze(), cmap="RdYlBu", vmin=-3, vmax=3, extent=plt_extent)
    # cbar = plt.colorbar()
    # cbar.set_label("Standardized elevation differences (m)")
    # plt.legend(loc="lower right")
    # #plt.show()
    print('here now2 ')
    # # %%
    # Now, we can perform an analysis of spatial correlation as shown in the :ref:`sphx_glr_advanced_examples_plot_variogram_estimation_modelling.py`
    # example, by estimating a variogram and fitting a sum of two models.
    # df_vgm = xdem.spatialstats.sample_empirical_variogram(
    #     values=z_dh.data.squeeze(), gsd=dh.res[0], subsample=300, n_variograms=10, random_state=42
    # )
    # df_vgm.to_csv(statsfn+'variogram2.csv')
    print('variogram')
    df_vgm = pd.read_csv(statsfn+'variogram2.csv')
    # 

    func_sum_vgm, params_vgm = xdem.spatialstats.fit_sum_model_variogram(
        ["Gaussian", "Spherical"], empirical_variogram=df_vgm
        )
        
    # %%
    # With standardized input, the variogram should converge towards one. With the input data close to a stationary
    # variance, the variogram will be more robust as it won't be affected by changes in variance due to terrain- or
    # instrument-dependent variability of measurement error. The variogram should only capture changes in variance due to
    # spatial correlation.



    df2 = pd.DataFrame(columns=['slope', 'maxc', 'z_error', 'nrsamples', 'dh', 'dh_err'],index=['all', 'small'])

    # for gl in glacier_outlines.ds["nr"]:
    #     if missing == 'no':
        # if gl not in [2167, 2168, 2169, 2170, 2171]:
            # print(gl)
    gl_shp = gu.Vector(glacier_outlines.ds)
    gl_mask = gl_shp.create_mask(dh)

    nmeanslope = np.nanmean(slope[gl_mask])
    nmeanmaxc = np.nanmean(maxc[gl_mask])

    df2.loc['all', 'slope'] = nmeanslope
    df2.loc['all', 'maxc'] = nmeanmaxc

    gl_neff = xdem.spatialstats.neff_circular_approx_numerical(
        area=gl_shp.ds.area.values[0], params_variogram_model=params_vgm
        )

    df2.loc['all', 'nrsamples'] = gl_neff

    gl_z_err = 1 / np.sqrt(gl_neff)


    df2.loc['all', 'z_error'] = gl_z_err


    # Destandardize the uncertainty
    fac_gl_dh_err = scale_fac_std * np.nanmean(dh_err[gl_mask.data])
    gl_dh_err = fac_gl_dh_err * gl_z_err

    # Derive mean elevation change
    gl_dh = np.nanmean(dh.data[gl_mask.data])

    df2.loc['all', 'dh'] = gl_dh
    df2.loc['all', 'dh_err'] = gl_dh_err
    df2.to_csv(statsfn+'error_dataframe_all.csv')

    
    xdem.spatialstats.plot_variogram(
        df_vgm,
        xscale_range_split=[100, 1000, 10000],
        list_fit_fun=[func_sum_vgm],
        list_fit_fun_label=["Standardized double-range variogram"],
    )
    plt.savefig(statsfn+'variogram_all.png')
    # plt.show()


# Careful, this takes a long time to run. 'yes' / 'no' refers to whether a couple of very small 
# glaciers are missing in the DEMs, if so they need to be skipped in the function. They are missing
# in the 2007 and 2017 DEMs.

# #stand_errors(fn1969_5al, fn1997_5al, dh_1969, GI1, '19691997_', 'no')
# stand_errors(fn1997_5al, fn2006_5al, dh_1997, GI1, '19972006_', 'yes')
# # stand_errors(fn2006_5al, fn2017_5al, dh_2006, GI1, '20062017_', 'yes')



stand_errors_all(fn2006_5al, fn2017_5al, dh_2006, GI1, '20062017_', 'yes')

plt.show()


