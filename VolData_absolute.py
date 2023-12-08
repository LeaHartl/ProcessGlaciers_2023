import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib as mpl

import geopandas as gpd
import rasterio
import rioxarray

import helpers as proc
import meta as mt
import xdem 
import geoutils as gu


# dat = pd.read_csv('data/volchange_constantrate_new.csv')
# print(dat)
# fig, ax = plt.subplots(1,1)
# ax.plot(dat.time, dat.volume)
# plt.show()
# stop

# -------------------------------------------
def getVol(ice, shapes, IDs):

    # loop over glacier IDs to get total V per glacier.
    Vol = pd.DataFrame(index=IDs, columns=['VolGI3', 'AreaGI3'])

    for ID in IDs:
        Vol.loc[ID, 'Vol'], Vol.loc[ID, 'count'] = proc.getDZ_ID(ice, shapes[IDs.index(ID):IDs.index(ID)+1])

    # pixel size is 10x10 m, multiply dz by 10x10 to get dV
    Vol['VolGI3'] = Vol['Vol']*100
    Vol['AreaGI3'] = Vol['count']*100

    return (Vol)


def getDZ(dif, shapes, yrs, IDs):
    # loop over glacier IDs to get total dV per glacier.
    dz = pd.DataFrame(index=yrs.ID.values, columns=['dV'])
    for ID in yrs.ID.values:
        dz.loc[ID, 'dZsum'], dz.loc[ID, 'count'] = proc.getDZ_ID(dif, shapes[IDs.index(ID):IDs.index(ID)+1])
    
    # pixel size is 5x5 m, multiply dz by 5x5 to get dV
    dz['dV'] = dz['dZsum'] * 25
    return (dz)
# -------------------------------------------
# set path to data
# read raster, set crs, save raster as tif with crs
# ice = '/Users/leahartl/Desktop/OGGM_Patrick/GI3_ice_thickness_clip.tif'
# proc.SetCRSIceThk(ice)

meta=mt.meta

# read raster file with proper crs settings
# ice_crs = '/Users/leahartl/Desktop/OGGM_Patrick/GI3_ice_thickness_clip_crs.tif'
ice_crs = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/GI3_ice_thickness_clip_crs.tif'



# read shapefile of glacier boundary 2006
GI3 = meta['GI3']['shp']
gpdGI3 = gpd.read_file(GI3)
gpdGI3_reproj = gpdGI3.to_crs(31287)
# make reprojected outline file and save - project to crs of the ice thickness raster
gpdGI3_reproj.to_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/Oetztaler_Alpen_GI3_31287.shp')

# read shapefile of glacier boundary 2017
GI5 = meta['GI5']['shp']
gpdGI5 = gpd.read_file(GI5)
gpdGI5_reproj = gpdGI5.to_crs(31287)


## see VolData.py in /v2 if tif needs to be reproduced


difdem = meta['GI5']['f_dif']

difdem_ma = meta['GI5']['f_dif_ma']

dem2017 = meta['GI5']['f_dem'][1]


# # # resample difdem to match clipped ice raster
# proc.warpDEMS([ice_crs, difdem], 'data/difdem_10m.tif')

# # # # resample difdem_ma to match clipped ice raster
# proc.warpDEMS([ice_crs, difdem_ma], 'data/difdem_10m_ma.tif')

# # # # # resample dem to match clipped ice raster
# proc.warpDEMS([ice_crs, dem2017], 'data/dem17_10m_resampled_to_ice.tif')


#dem2017resampled = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/dem17_10m_resampled_to_ice.tif'


# uncomment lines as needed to process the DEMs
# # subtract the difference from the ice thickness
# proc.addRaster('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/difdem_10m.tif', ice_crs, 'icethickness_2017')

# # #clip ice 2017 with GI3 boundary (reprojected to match crs of the ice DEM)
# proc.clipRaster('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/icethickness_2017.tif', gpdGI3_reproj.geometry, 'ice2017_clip_GI3')
ice_2017 = 'data/ice2017_clip_GI3.tif'

# # #clip ice 2006 with GI3 boundary (reprojected to match crs of the ice DEM)
# proc.clipRaster(ice_crs, gpdGI3_reproj.geometry, 'ice2006_clip_GI3')

ice_2006 = 'data/ice2006_clip_GI3.tif'

# # clip the resampled difference dem (total AND meter/year version) with GI3 boundary (reprojected to match crs of the ice DEM)
# proc.clipRaster('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/difdem_10m_ma.tif', gpdGI3_reproj.geometry, 'dif_ma_clip_GI3')

# proc.clipRaster('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/difdem_10m.tif', gpdGI3_reproj.geometry, 'dif_clip_GI3')

# # files produced by function in previous line.
difdem_ma_clip = 'data/dif_ma_clip_GI3.tif'
difdem_clip = 'data/dif_clip_GI3.tif'

# # divide ice thickness raster by difdem_clip, pixelwise. (--> divide volume by dz/a)
# div_Vol1 = proc.divideVol(ice_2017, difdem_ma_clip, 'division')

# # # clip the tif of the divided data dem with GI5 boundary (reprojected to match crs of the ice DEM)
# proc.clipRaster('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/division.tif', gpdGI5_reproj.geometry, 'division_clip_GI5')

# # resulting file
division_GI5 = 'data/division_clip_GI5.tif'


def yearstilmelt(division_GI5):
    # resulting file
    # division_GI5 = 'data/division_clip_GI5.tif'
    #make "years till melt" figure
    with rasterio.open(division_GI5) as src:
            div_Vol = src.read(1, masked=True) *-1

    # div_Vol[div_Vol<100]=np.nan
    fig, ax  = plt.subplots(1, 1, figsize =(10, 8))
    # im = ax.imshow(div_Vol, cmap='cividis', vmax=300, vmin=0)
    im = ax.contourf(div_Vol, cmap='cividis', levels = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260])#, vmax=300, vmin=0)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.05, 0.8])
    plt.colorbar(im, cax=cax)
    ax.set_xlim([0, 4500])
    ax.set_ylim([4800, 700])
    ax.set_title('years until ice gone, per pixel, starting from vol. 2017/18')
    fig.savefig('plots/yearstilmelt.png')


# uncomment to make the plot
# yearstilmelt(division_GI5)
# plt.show()
# stop



def getIceYears():
    #get ice thickness per year per pixel until 2100
    #write to nc file - reload older file unless something changed to save time!
    ice_ar = rioxarray.open_rasterio(ice_2017)
    # ice_ar = rioxarray.open_rasterio(ice_2006)
    print(ice_ar)
    ice_ar = ice_ar.drop_vars('band')
    dif_ice = rioxarray.open_rasterio(difdem_ma_clip) # use m/a file here!
    dif_a = dif_ice ###/ (2017-2006)
    # y1 = 2017
    y1 = 2017
    ice_xr1 = ice_ar.to_dataset(name = 'thickness')
    # ice_xr = ice_xr1.expand_dims(yrs=np.arange(2017, 2101))
    ice_xr = ice_xr1.expand_dims(yrs=np.arange(y1, 2101))

    forconcat = []
    for i, y in enumerate(np.arange(y1, 2101)):

        ice = ice_xr.thickness.sel(yrs=y1).values + (dif_a*i)
        # set negative values to np.nan 
        ice = ice.where(ice > 1)
        forconcat.append(ice)

    combined = xr.concat(forconcat, pd.Index(np.arange(y1, 2101), name='time'))
    combined_ice = combined.to_dataset(name='thickness')

    combined_ice = combined_ice.drop_vars('band')
    combined_ice = combined_ice.squeeze()
    ##print(combined_ice)
    combined_ice = combined_ice.sortby(["time"])#, "y", "x"])
    combined_ice=combined_ice.reindex(y=list(reversed(combined_ice.y)))
    combined_ice.rio.write_crs("epsg:31287", inplace=True)
    # ds.transpose()
    #combined_ice = combined_ice.drop_dims('band')
    combined_ice.to_netcdf('data/ice_thickness_new.nc')

# uncomment if the nc file needs to be reproduced (slow). otherwise load existing version.
# getIceYears()
print('bla')
#stop
# combined_ice = xr.open_dataset('data/ice_thickness.nc')
combined_ice = xr.open_dataset('data/ice_thickness_new.nc')


# load to check:
print(combined_ice)
print(combined_ice.sel(time=2050))
ice2050 = combined_ice.sel(time=2050)


def writetodisk(yr, combined_ice):
    fname2 = 'data/ice_'+str(yr)+'.tif'
    # write to disk
    print(fname2)
    ice = combined_ice.sel(time=yr)
    ice.rio.write_crs("epsg:31287", inplace=True)
    #combined_ice.rio.write_nodata(np.nan, inplace=True)
    ice.rio.to_raster(fname2, driver='GTiff')
# resample difdem to match clipped ice raster

def writewarped(yr, ice_crs):
    fname2 = 'data/ice_'+str(yr)+'.tif'
    proc.warpDEMS([ice_crs, fname2], 'data/ice_'+str(yr)+'_resamp.tif')

# # uncomment as needed
# writetodisk(2100, combined_ice)
# writewarped(2100, ice_crs)
# writetodisk(2030, combined_ice)
# writewarped(2030, ice_crs)
# writetodisk(2050, combined_ice)
# writewarped(2050, ice_crs)
# writetodisk(2075, combined_ice)
# writewarped(2075, ice_crs)

# # load and plot to check files:
# d17 = rioxarray.open_rasterio('data/ice_2100_resamp.tif', engine='netcdf4')
# print(d17)

# plt.imshow(d17.squeeze())
# plt.show()
# stop
# combined_ice['elev'] = xr.DataArray(NDTI2, dims=('y', 'x'), 
#                 coords={'x': x, 'y': np.flip(y)})


def forGIF(ice):
    for yr in ice.time.values:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ice.thickness.sel(time=yr).plot(cmap='plasma', vmin=0, vmax=150, ax=ax)
        fig.savefig('figs/thickness'+str(yr)+'.png')
        plt.close()

# forGIF(combined_ice)

total = combined_ice.sum(dim=['x', 'y'])
print(total)
print(total.to_dataframe())
# write total volume per year to csv:
totalVol = total.to_dataframe()
totalVol = totalVol.rename(columns={"thickness": "volume"})
totalVol.volume = totalVol.volume*10*10
totalVol = totalVol[['volume']]
totalVol.to_csv('data/volchange_constantrate_new.csv')

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(total.time, total.thickness*10*10)
ax.set_ylabel('vol in m3')
ax.set_xlabel('year')
ax.set_title('Ötztal+Stubai total volume')
fig.savefig('plots/totalvolume_constantchange.png')
plt.show()




