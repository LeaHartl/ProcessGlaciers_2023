import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib as mpl
from scipy.stats import circmean

import geopandas as gpd

import geoutils as gu
import xdem

import helpers as proc
import meta as mt
# -------------------------------------------

# -------------------------------------------
# set path to data (dictionary in file "meta.py")
meta = mt.meta

# ice thickness data:
# read raster file with proper crs settings
ice_crs = 'data/GI3_ice_thickness_clip_crs.tif'

# -------------------------------------------
# get area change per size class, make table
# GI1 = countGlaciers(meta['GI1']['shp'])  
GI2 = gpd.read_file(meta['GI2']['shp']) #1997
GI3 = gpd.read_file(meta['GI3']['shp']) #2006
GI5 = gpd.read_file(meta['GI5']['shp']) #2017/18


def VolChange(cl, dif, px):
    dz = xdem.DEM(dif)
    # print(cl)
    if cl.crs != dz.crs:
        print('crs do not match, reprojecting')
        cl.to_crs(dz.crs, inplace=True)

    gl_mask = gu.Vector(cl).create_mask(dz)

    # Derive mean elevation change
    gl_dh = np.nanmean(dz.data[gl_mask.data])
    gl_dh_total = gl_dh * cl.area.sum()

    gl_dh_total_px = np.nansum(dz.data[gl_mask.data])*px*px

    return(gl_dh_total, gl_dh_total_px, gl_dh)



# get volume change per size class
def getChange_byclass(dif, shp):#, rep, pxsize):
    mG = gpd.read_file(shp)
    mG['area'] = mG.geometry.area

    mG['class'] = np.nan
    mG['class'].loc[mG.area < 0.5e6] = 'v_small'
    mG['class'].loc[(mG.area >= 0.5e6) & (mG.area < 1e6)] = 'small'
    mG['class'].loc[(mG.area >= 1e6) & (mG.area < 5e6)] = 'mid'
    mG['class'].loc[(mG.area >= 5e6) & (mG.area < 10e6)] = 'big'
    mG['class'].loc[(mG.area >= 10e6)] = 'vbig'

    sz_cl = mG[['geometry', 'class', 'nr']].dissolve(by='class', aggfunc='count')
    sz_cl['area'] = sz_cl.geometry.area
    sz_cl['class'] = sz_cl.index
    # print(sz_cl)
    # print(sz_cl[['nr', 'area']].sum())
    # plot glaciers colored by size classes as reality check
    # fig, ax = plt.subplots(1, 1)
    # sz_cl.plot(ax=ax, column='class')
    # plt.show()



    dz_class = pd.DataFrame(columns=['dz_total', 'dz_total_px'], index = ['all', 'v_small', 'small', 'mid', 'big', 'vbig'])
    classes = [mG, sz_cl.loc[sz_cl.index=='v_small'], sz_cl.loc[sz_cl.index=='small'], sz_cl.loc[sz_cl.index=='mid'], sz_cl.loc[sz_cl.index=='big'], sz_cl.loc[sz_cl.index=='vbig']]

    total=[]
    total_px =[]
    dh_mean = []
    ar = []
    for i, cl in enumerate(classes):
        # call volchange function and pass geometries of glaciers in each size class, difdem, and pixelsize:
        # output is total change as mean * area and for comparison sum * pixel size
        gl_dh_total, gl_dh_total_px, gl_dh_mean = VolChange(cl, dif, 5)
        total.append(gl_dh_total)
        total_px.append(gl_dh_total_px)
        dh_mean.append(gl_dh_mean)
        ar.append(cl.area.sum())

    dz_class['dz_total'] = total
    dz_class['dz_total_px'] = total_px
    dz_class['dh_mean'] = dh_mean
    dz_class['area'] = ar


    return (dz_class)

# dz_class_total = getChange_byclass(meta['GI5']['f_dif'], meta['GI3']['shp'])
# dz_class_ma = getChange_byclass(meta['GI5']['f_dif_ma'], meta['GI3']['shp'])

# print(dz_class_total)
# # print(dz_class_ma)


#--- deal with errors -----
erdf_20062017_all = pd.read_csv('outputXdem/20062017_error_dataframe_classes.csv', index_col=0)
erdf_20062017_ma = pd.read_csv('outputXdem/20062017_maerror_dataframe_classes.csv', index_col=0)

# erdf_20062017_all = pd.read_csv('outputXdem/20062017_error_dataframe_classes.csv')
erdf_19972006_ma = pd.read_csv('outputXdem/19972006_maerror_dataframe_classes.csv', index_col=0)



erdf_20062017_all = erdf_20062017_all.sort_index()
# dz_class_total = dz_class_total.sort_index()
# merge20062017_all = pd.concat([dz_class_total, erdf_20062017_all], axis=1)

area_unc = pd.read_csv('tables/Uncertainties_Area.csv', index_col=0)
print(area_unc)
print(erdf_20062017_all)

merge20062017_all = area_unc.join(erdf_20062017_all[['dh', 'dh_err']])
merge20062017_ma = area_unc.join(erdf_20062017_ma[['dh', 'dh_err']])

merge19972006_ma = area_unc.join(erdf_19972006_ma[['dh', 'dh_err']])

# print(merge20062017_all)
# print(dz_class_total)

# volume change 2006-2017 total
merge20062017_all['uncVolchange'] = np.sqrt( (merge20062017_all['dh_err']*merge20062017_all['area2006'])**2 + (merge20062017_all['Unc2006']*merge20062017_all['dh'])) 
merge20062017_all['uncVolchangekm3'] = merge20062017_all['uncVolchange']/1e9

#reality check:
merge20062017_all['Volchange'] = merge20062017_all['area2006']*merge20062017_all['dh']

# volume change 2006-2017 per year (m/a)
merge20062017_ma['uncVolchange'] = np.sqrt( (merge20062017_ma['dh_err']*merge20062017_ma['area2006'])**2 + (merge20062017_ma['Unc2006']*merge20062017_ma['dh'])) 
merge20062017_ma['uncVolchangekm3'] = merge20062017_ma['uncVolchange']/1e9

#reality check:
merge20062017_ma['Volchange'] = merge20062017_ma['area2006']*merge20062017_ma['dh']


# volume change 1997-2006 per year (m/a)
merge19972006_ma['uncVolchange'] = np.sqrt( (merge19972006_ma['dh_err']*merge19972006_ma['area1997'])**2 + (merge19972006_ma['Unc1997']*merge19972006_ma['dh'])) 
merge19972006_ma['uncVolchangekm3'] = merge19972006_ma['uncVolchange']/1e9

#reality check:
merge19972006_ma['Volchange'] = merge19972006_ma['area1997']*merge19972006_ma['dh']


print(merge20062017_all)
print(merge20062017_ma)
print(merge19972006_ma)


# erdf_20062017_ma = erdf_20062017_ma.sort_index()

# erdf_19972006_ma = erdf_19972006_ma.sort_index()











