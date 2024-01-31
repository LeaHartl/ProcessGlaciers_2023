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




    # return 


#------get ice thickness------

# deal w farinotti data:
# get RGI ids in ROI: 
# rgiroi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI_ROI.shp')
# idlist = rgiroi.RGIId.values

# # get filelist farinotti:
# folder = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI60-11_far/'
# files = glob.glob(folder+'/*.tif')

# # get RGI ID
# ids = []
# rioars = []
# for file in files:
#     nr = file[-28:-14]
#     if nr in idlist:
#         ids.append(file)
#         rioars.append(rioxarray.open_rasterio(file, masked=True))

# # merge the relevant tifs and save to new raster
# merged = merge_arrays(rioars)
# merged.rio.to_raster('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/icedepthFarinotti.tif')


# load cook ROI
cookRoi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/cook_area/cook_area.shp')



# load ice thickness rasters:
ice_crs = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/GI3_ice_thickness_clip_crs.tif'
ice_Helfricht = xdem.DEM(ice_crs)
print(ice_Helfricht.crs)


farin = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/icedepthFarinotti/icedepthFarinotti.tif'
ice_farin = xdem.DEM(farin)

millan = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI-11_millan/THICKNESS_RGI-11_2021July09.tif'
ice_millan = xdem.DEM(millan)


get_sum(ice_Helfricht, cookRoi, 10, 10)

get_sum(ice_farin, cookRoi, 25, 25)

get_sum(ice_millan, cookRoi, 50, 50)


