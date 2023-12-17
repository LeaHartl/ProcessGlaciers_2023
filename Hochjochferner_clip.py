import rasterio
from rasterio.plot import show
import numpy as np
import pandas as pd
# import os
# os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import matplotlib.pyplot as plt
import geoutils as gu
import xdem
import meta as mt


meta=mt.meta

GI2 = meta['GI2']['shp']
GI3 = meta['GI3']['shp']
GI5 = meta['GI5']['shp']

# difdemGI2 = meta['GI2']['f_dif']
# difdemGI3 = meta['GI3']['f_dif']
# difdemGI5 = meta['GI5']['f_dif']

# difdem_ma = meta['GI5']['f_dif_ma']

# load outlines and extract hochjochferner, save to news file for each GI. This is then clipped in
# Qgis to remove western most part that is not covered by DEMs.

# for i, GI in enumerate([GI2, GI3, GI5]):
#     outlines = gpd.read_file(GI)
#     hochjoch = outlines[outlines.nr == 2121]
#     print(hochjoch.crs)
#     hochjoch.to_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/HochjochGI_'+str(i+2)+'.shp')
#     print(hochjoch)

HochJoch_clipGI2 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/Hochjoch_clipGI_2.shp'
HochJoch_clipGI3 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/Hochjoch_clipGI_3.shp'
HochJoch_clipGI5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/Hochjoch_clipGI_5.shp'


HJGI2 = gpd.read_file(HochJoch_clipGI2)
HJGI2 = HJGI2.explode()
HJGI2 = HJGI2.dissolve(by='nr')

HJGI3 = gpd.read_file(HochJoch_clipGI3).explode()
HJGI3 = HJGI3.explode()
HJGI3 = HJGI3.dissolve(by='nr')

HJGI5 = gpd.read_file(HochJoch_clipGI5).explode()
HJGI5 = HJGI5.explode()
HJGI5 = HJGI5.dissolve(by='nr')

HJGI3_reproj = HJGI3.to_crs(31287)
ice_crs = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/GI3_ice_thickness_clip_crs.tif'        

def clipDEM(dem, outline):
    dh = xdem.DEM(dem)
    gl_shp = gu.Vector(outline)
    gl_mask = gl_shp.create_mask(dh)

    # Derive mean elevation change
    gl_dh = np.nanmean(dh.data[gl_mask.data])
    gl_dh_total = gl_dh * gl_shp.area.values
    return(gl_dh_total)

            # dfdV.loc[gl, 'dh'] = gl_dh
            # dfdV.loc[gl, 'dV'] = gl_dh_total[0]

ice2006 = clipDEM(ice_crs, HJGI3_reproj)

dif20061997 = clipDEM(meta['GI3']['f_dif'], HJGI2)
dif20172006 = clipDEM(meta['GI5']['f_dif'], HJGI3)

dat = pd.DataFrame(index=['1997', '2006', '2017'], columns=['area', 'areachange', 'volume', 'volchange'])
dat.loc['1997', 'area'] = HJGI2.geometry.area.sum()
dat.loc['2006', 'area'] = HJGI3.geometry.area.sum()
dat.loc['2017', 'area'] = HJGI5.geometry.area.sum()

dat.loc['2006', 'volume'] = ice2006

dat.loc['2006', 'volchange'] = dif20061997
dat.loc['2017', 'volchange'] = dif20172006

dat.loc['2006', 'areachange'] = HJGI3.geometry.area.sum() - HJGI2.geometry.area.sum()
dat.loc['2017', 'areachange'] = HJGI5.geometry.area.sum() - HJGI3.geometry.area.sum()

print(dat)
dat.to_csv('data/Hochjoch_clipped.csv')


fig, ax = plt.subplots(1, 1)
HJGI2.boundary.plot(label='1997_clipped', ax=ax, color='r')
HJGI3.boundary.plot(label='2006_clipped', ax=ax, color='k')
HJGI5.boundary.plot(label='2017_clipped', ax=ax)
ax.legend()
fig.savefig('data/Hochjoch_clipped.png')

plt.show()
