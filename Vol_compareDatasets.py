import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import geoutils as gu # note: version 0.0.12
import xdem # note: version 0.0.10
import meta as mt
import os
import glob

import rioxarray 
import xarray
from rioxarray.merge import merge_arrays

# import dictionary with paths to files:
meta = mt.meta


def computeNMAD(outlines, dem):
    glacier_outlines = gu.Vector(outlines)
    inlier_mask = ~glacier_outlines.create_mask(dem)

    nmad = xdem.spatialstats.nmad(dem[inlier_mask])
    print(nmad)


# clip with ROI, get sum
def get_sum(raster, outline1, px_x, px_y):
    # reproject
    outline1.to_crs(raster.crs, inplace=True)
    outline = gu.Vector(outline1)

    # get sum of pixels
    gl_mask = outline.create_mask(raster)
    clippedice = raster.data[gl_mask.data]
    icesum = np.nansum(clippedice)
    # multiply by pix size
    icetotal = icesum * px_x *px_y
    print(icetotal)
    print(icetotal/1e9)

    print('divided by area:', icetotal/outline1.geometry.area.sum())




    # return 


#------get ice thickness------


# deal w farinotti data tiles:
def mergetiles_Farin():
    # get RGI ids in ROI: 
    rgiroi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI_ROI.shp')
    idlist = rgiroi.RGIId.values

    # get filelist farinotti:
    folder = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI60-11_far/'
    files = glob.glob(folder+'/*.tif')

    # get RGI ID
    ids = []
    rioars = []
    for file in files:
        nr = file[-28:-14]
        if nr in idlist:
            ids.append(file)
            rioars.append(rioxarray.open_rasterio(file, masked=True))

    # merge the relevant tifs and save to new raster
    merged = merge_arrays(rioars)
    merged.rio.to_raster('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/icedepthFarinotti.tif')


# deal w Hugonnet elevation change data tiles:
def mergetiles_Hugon():
    folderHugo = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/11_rgi60_2000-01-01_2020-01-01/dhdt/'

    filesHugo = ['N46E010_2000-01-01_2020-01-01_dhdt.tif', 'N46E011_2000-01-01_2020-01-01_dhdt.tif',
            'N47E010_2000-01-01_2020-01-01_dhdt.tif', 'N47E011_2000-01-01_2020-01-01_dhdt.tif']

    rioars = []
    for file in filesHugo:
        rioars.append(rioxarray.open_rasterio(folderHugo + file, masked=True))

    # merge the relevant tifs and save to new raster
    merged = merge_arrays(rioars)
    merged.rio.to_raster('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/volchangeHugonnet2000_2020.tif')

## uncomment to make tifs with merged tiles
# mergetiles_Farin()
# mergetiles_Hugon()


# load cook ROI
cookRoi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/cook_area/cook_area.shp')

# get relevant part of RGI 
rgiroi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI_ROI.shp')

clippedRGI = gpd.clip(rgiroi, cookRoi)
clippedRGI.to_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/clippedRGI.shp')


# uncomment to load and process the various rasters:
# regional ice thickness (Helfricht) clipped with ROI
# ice_crs = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/GI3_ice_thickness_clip_crs.tif'
# ice_Helfricht = xdem.DEM(ice_crs)
# print(ice_Helfricht.crs)

# get_sum(ice_Helfricht, cookRoi, 10, 10)

# Farinotti ice thickness clipped with ROI
# farin = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/icedepthFarinotti/icedepthFarinotti.tif'
# ice_farin = xdem.DEM(farin)

# get_sum(ice_farin, cookRoi, 25, 25)


# Millan  ice thickness clipped with ROI
# millan = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI-11_millan/THICKNESS_RGI-11_2021July09.tif'
# ice_millan = xdem.DEM(millan)
# get_sum(ice_millan, cookRoi, 50, 50)

# Hugonnet vol change clipped with ROI and RGI 
hugonnet = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/volchangeHugonnet2000_2020.tif'
volchange_hugonnet = xdem.DEM(hugonnet)
# get_sum(volchange_hugonnet, clippedRGI, 100, 100)


dz1969_97 = xdem.DEM(meta['GI2']['f_dif_ma'])
gg69 = gpd.read_file(meta['GI1']['shp'])

dz1997_06 = xdem.DEM(meta['GI3']['f_dif_ma'])
gg97 = gpd.read_file(meta['GI2']['shp'])

dz2006_17 = xdem.DEM(meta['GI5']['f_dif_ma'])
gg06 = gpd.read_file(meta['GI3']['shp'])

# get_sum(dz1969_97, gg69, 5, 5)
# get_sum(dz1997_06, gg97, 5, 5)
# get_sum(dz2006_17, gg06, 5, 5)


computeNMAD(gg69, dz1969_97)
computeNMAD(gg69, dz1997_06)
computeNMAD(gg69, dz2006_17)
computeNMAD(clippedRGI, volchange_hugonnet)


