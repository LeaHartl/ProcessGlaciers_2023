#! /usr/bin/env python3
import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib as mpl

import fiona
import os
import geopandas as gpd
import rasterio
import rioxarray


# import EAZ_vis as ev
#import EAZ_setup as st
import procDEMS_1 as proc
import meta as mt
import xdem 
import geoutils as gu


meta=mt.meta


# dem = xdem.DEM(meta['GI5']['f_dem'][1])

# # slope201718 = xdem.terrain.slope(dem)
# # aspect201718 = xdem.terrain.aspect(dem)

# # slope201718.save('xdem1/slope201718.tif')
# # aspect201718.save('xdem1/saspect201718.tif')

# slope = xdem.DEM('xdem1/slope201718.tif')
# aspect = xdem.DEM('xdem1/saspect201718.tif')


# def clip_dif_ma_raster(meta):
#     # meta = meta.meta
#     # dm = xdem.DEM('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif')
#     # #clip 
#     fn = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'
#     proc.clipRaster(fn, gpd.read_file(meta['GI3']['shp']).geometry, 'aligned_resampled_2_5m_tir1718_Clip')


# # def clip_dif_ma_raster(meta):
# #     # meta = meta.meta
# #     dz_ma = xdem.DEM('xdem1/dif_5m_20062017_ma.tif')
# #     #clip
# #     proc.clipRaster('xdem1/dif_5m_20062017_ma.tif', gpd.read_file(meta['GI3']['shp']).geometry, 'dif_5m_20062017_ma_Clip')

# clip_dif_ma_raster(meta)
# stop

def plot_attribute(attribute, cmap, label=None, vlim=None):

    add_cbar = True if label is not None else False

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    if vlim is not None:
        if isinstance(vlim, (int, float)):
            vlims = {"vmin": -vlim, "vmax": vlim}
        elif len(vlim) == 2:
            vlims = {"vmin": vlim[0], "vmax": vlim[1]}
    else:
        vlims = {}

    attribute.show(ax=ax, cmap=cmap, add_cbar=add_cbar, cbar_title=label, **vlims)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


plot_attribute(aspect, "twilight", "Aspect (°)")


plt.show()


