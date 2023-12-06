# import procDEMS as proc
# import EAZ_setup as st
import rasterio
from rasterio.plot import show
import numpy as np
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'

import geopandas as gpd
import matplotlib.pyplot as plt
import procDEMS_new as proc

import geoutils as gu
import xdem
import meta as mt
# supresse setting with copy warning, use carefully!
pd.options.mode.chained_assignment = None  # default='warn'

meta = mt.meta

def flightyearsraster():
    # # this makes raster of flight years:
    clippedflights = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/data/als_flugjahr_v22/clipped_flugjahr.shp'
    gd = gpd.read_file(clippedflights)
    gd.index = gd['year_int']-1

    refraster = xdem.DEM('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/2006_coreg.tif')
    vec = gu.Vector(gd)
    raster_yr = vec.rasterize(refraster)
    # print(vec)
    # print(raster_yr.transform)
    # raster_yr.save('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/data/als_flugjahr_v22/flightyearsRaster.tif', driver='GTiff', dtype=None, nodata=None, compress='deflate', tiled=False, blank_value=None, co_opts=None, metadata=None, gcps=None, gcps_crs=None)

    div = (raster_yr - 2006)
    print(np.unique(div))

    dz_ma = xdem.DEM('xdem1/dif_5m_20062017.tif') / div

    dz_ma.save('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_20062017_ma.tif', driver='GTiff', dtype=None, nodata=None, compress='deflate', tiled=False, blank_value=None, co_opts=None, metadata=None, gcps=None, gcps_crs=None)
    # print(dz_ma)


def dif_ma_raster():
    dz_maGI3 = xdem.DEM('xdem1/dif_5m_19972006.tif') / (2006-1997)
    dz_maGI3.save('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19972006_ma.tif', driver='GTiff', dtype=None, nodata=None, compress='deflate', tiled=False, blank_value=None, co_opts=None, metadata=None, gcps=None, gcps_crs=None)
    
    dz_maGI2 = xdem.DEM('xdem1/dif_5m_19691997.tif') / (1997-1969)
    dz_maGI2.save('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19691997_ma.tif', driver='GTiff', dtype=None, nodata=None, compress='deflate', tiled=False, blank_value=None, co_opts=None, metadata=None, gcps=None, gcps_crs=None)
    #

# flightyearsraster()
# dif_ma_raster()





# dem = xdem.DEM(meta['GI5']['f_dem'][1])

# # slope201718 = xdem.terrain.slope(dem)
# # aspect201718 = xdem.terrain.aspect(dem)

# # slope201718.save('xdem1/slope201718.tif')
# # aspect201718.save('xdem1/saspect201718.tif')

# slope = xdem.DEM('xdem1/slope201718.tif')
# aspect = xdem.DEM('xdem1/saspect201718.tif')


# def clip_dif_ma_raster(meta):
#     # meta = meta.meta
#     #dm = xdem.DEM('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif')
#     # #clip 
#     fn = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'
#     proc.clipRaster(fn, gpd.read_file(meta['GI3']['shp']).geometry, 'aligned_resampled_2_5m_tir1718_Clip')


def clip_dif_ma_raster(meta):
    # meta = meta.meta
    dz_ma = xdem.DEM('xdem1/dif_5m_20062017_ma.tif')
    #clip
    proc.clipRaster('xdem1/dif_5m_20062017_ma.tif', gpd.read_file(meta['GI3']['shp']).geometry, 'dif_5m_20062017_ma_Clip')

# clip_dif_ma_raster(meta)





# --------
#deal with HEF in GI5:
fldr = 'shapefiles'

Otz = {
    'GI1': {
    'shp' : fldr +'/Oetztal_GI1.shp',
    },
    'GI2': {
    'shp' : fldr +'/Oetztal_GI2.shp',
    },
    'GI3': {
    'shp' : fldr +'/Oetztal_GI3.shp',
    },
    'GI5': {
    # 'shp' : 'Newshapes/Oetztaler_Alpen_GI5_pangaea_LH.shp',
    'shp' : 'Newshapes/Pangaea/Oetztaler_Alpen_GI5_pangaea.shp',
    }
    }

Stb = {
    'GI1': {
    'shp' : fldr +'/Stubaier_Alpen_GI1.shp',
    },
    'GI2': {
    'shp' : fldr +'/Stubaier_Alpen_GI2.shp',
    },
    'GI3': {
    'shp' : fldr +'/Stubaier_Alpen_GI3.shp',
    },
    'GI5': {
    # 'shp' : 'Newshapes/Stubai_GI5_pangaea.shp',
    'shp' : 'Newshapes/Pangaea/Stubai_GI5_pangaea.shp',
    }
    }


# combine HEF ohne Toteis and Toteis in GI5: 
dat = gpd.read_file(Otz['GI5']['shp'])
datHEF = dat.loc[(dat.nr == 2125000) | (dat.nr == 0)]
datHEF['key'] = 'HEFtotal'
datHEF = datHEF[['geometry', 'key']]
total = datHEF.dissolve(by='key')
total['Gletschern'] = 'Hintereisferner'
total['nr'] = 2125
total['Year'] = 2017
total['Area'] = total.geometry.area

dat = pd.concat([dat, total])


sub = dat.loc[(dat.nr == 2125) | (dat.nr == 0) | (dat.nr == 2125000)]

# save to new file
dat.to_file(fldr+'/Oetztaler_Alpen_GI5_HEFadjusted.shp')



