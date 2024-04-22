import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import geoutils as gu # note: version 0.0.12
import xdem # note: version 0.0.10


# import rioxarray 
# import xarray
# from rioxarray.merge import merge_arrays


# clip with ROI, get sum
def get_sum(raster, outline1, px_x, px_y):
    # reproject
    outline1.to_crs(raster.crs, inplace=True)
    print(outline1.geometry.area)

    print(raster.crs)
    outline = gu.Vector(outline1)

    # get sum of pixels
    gl_mask = outline.create_mask(raster)
    clippedice = raster.data[gl_mask.data]
    icesum = np.nansum(clippedice)
    # multiply by pix size
    icetotal = icesum * px_x *px_y

    #print(icetotal)
    print('pixel sum * pix size:', icetotal/1e9)
    print('mean thickness:', icetotal/outline1.geometry.area.sum())

    icetotalv2 = np.nanmean(clippedice)*outline1.geometry.area.sum()

    print('pixel mean * area:', icetotalv2/1e9)

    

#------get ice thickness------

# load RGI 6:
rgi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp')
# load Patrick's list of RGI IDs:
ids = pd.read_csv('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI_PIZ_OTZ_STB_2.csv')

clippedRGI = rgi.loc[rgi['RGIId'].isin(ids['rgi_id'].values)]
clippedRGIGEP = rgi.loc[rgi['RGIId']=='RGI60-11.00746']
# print(clippedRGIGEP.geometry.area)

# clippedRGI.to_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/clippedRGI.shp')


# Millan  ice thickness clipped with ROI
millan = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI-11_millan/THICKNESS_RGI-11_2021July09.tif'
ice_millan = xdem.DEM(millan)
get_sum(ice_millan, clippedRGIGEP, 50, 50)

