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


def getPolyCoords(row, geom, coord_type):
    # Returns the coordinates ('x' or 'y') of edges of a Polygon exterior

    # Parse the exterior of the coordinate
    exterior = row[geom].exterior
    if coord_type == 'x':
        # Get the x coordinates of the exterior
        return list(exterior.coords.xy[0])
    elif coord_type == 'y':
        # Get the y coordinates of the exterior
        return list(exterior.coords.xy[1])


# generate dataframe with dA for each glacier, input: gdf for two different years
def get_dA(gp1, gp2):
    gp1 = gp1.loc[gp1.nr != 2125]
    gp2 = gp2.loc[gp2.nr != 2125]

    gp1['area'] = gp1['geometry'].area
    gp2['area'] = gp2['geometry'].area

    both = gp1.merge(gp2, how='inner', on='nr', suffixes=('_1', '_2'))
    both['dA'] = both['area_2'] - both['area_1']

    return (both[['nr', 'dA']])


# get dz for individual glaciers
def getDZ(dif, shp):
    # read file, export list of shapes
    shapes, IDs = proc.listShapes(shp, 'nr')

    # loop over glacier IDs to get total dV per glacier.
    dz = pd.DataFrame(index=IDs, columns=['dV'])
    for ID in IDs:
        dz.loc[ID, 'dZsum'], dz.loc[ID, 'count'] = proc.getDZ_ID(dif, shapes[IDs.index(ID):IDs.index(ID)+1])
    
    # pixel size is 5x5 m, multiply dz by 5x5 to get dV
    dz['dV'] = dz['dZsum'] * 25
    dz['dZmean'] = dz['dZsum'] /dz['count']
    print(dz.head())
    print(dz['dZmean'].mean())

    return(dz)


# get dz for individual glaciers
def getDZ_xdem(dif, shp):
    dzList = []
    glList = []

    dz = xdem.DEM(dif)
    outlines = gpd.read_file(shp)
    for gl in outlines.nr: 
        gl_out = outlines[outlines.nr == gl]
        gl_mask = gu.Vector(gl_out).create_mask(dz)
        sum_dz = np.nansum(dz.data[gl_mask.data])
        # mean_dz = np.nanmean(dz.data[gl_mask.data])
        # # pixel size is 5x5 m, multiply dz by 5x5 to get dV
        dzList.append(sum_dz*5*5)
        glList.append(gl)

    dz_df = pd.DataFrame(index=glList, columns=['dz'])
    dz_df['dz'] = dzList
    return (dz_df)

# def get_ice_abs(dif, shapes, yrs, IDs):
#     # loop over glacier IDs to get total dV per glacier.
#     dz = pd.DataFrame(index=yrs.ID.values, columns=['dV'])
#     for ID in yrs.ID.values:
#         dz.loc[ID, 'dZsum'], dz.loc[ID, 'count'] = proc.getDZ_ID(dif, shapes[IDs.index(ID):IDs.index(ID)+1])

#     # pixel size is 5x5 m, multiply dz by 5x5 to get dV
#     dz['dV'] = dz['dZsum'] * 25
#     return (dz)


# helper function to get geometries
def getgeoms(gdf):
    shps = []
    for i in gdf.index:
        shp = gdf.loc[0, 'geometry']
        shps.append(shp)
    return (shps, gdf.nr.values)


# get volume change per size class
def getDZ_perSizeClass(dif, shp, rep, pxsize):
    mG = gpd.read_file(shp)
    mG = mG.loc[mG.nr != 2125]
    if rep == 'reproject':
        mG = mG.to_crs(31287)
    mG = mG.reset_index()
    
    All = mG
    G_vsmall = mG.loc[mG.area < 0.5e6]
    G_small = mG.loc[(mG.area >= 0.5e6) & (mG.area < 1e6)]
    G_mid = mG.loc[(mG.area >= 1e6) & (mG.area < 5e6)]
    G_big = mG.loc[(mG.area >= 5e6) & (mG.area < 10e6)]
    G_vbig = mG.loc[(mG.area >= 10e6)]

    dz_class = pd.DataFrame(columns=['dz', 'ar'], index = ['all', 'vsmall', 'small', 'mid', 'big', 'vbig'])
    classes = [All, G_vsmall, G_small, G_mid, G_big, G_vbig]

    dzList = []
    dz2List = []
    dzmeanList = []
    arList = []
    ar2List = []
    # for cl in classes: 
    #     Gs=cl.geometry.unary_union
    #     dz, ar, pixel_size_y, pixel_size_x = proc.getDZ_ID2(dif, [Gs])
    #     dzList.append(dz)
    #     arList.append(ar)

    dz = xdem.DEM(dif)
    for cl in classes: 
        gl_mask = gu.Vector(cl).create_mask(dz)
        sum_dz = np.nansum(dz.data[gl_mask.data])
        mean_dz = np.nanmean(dz.data[gl_mask.data])
        dzList.append(sum_dz)

        dzmeanList.append(mean_dz)
        # arList.append(cl.area.sum())

        sum2_dz = mean_dz * cl.area.sum()
        dz2List.append(sum2_dz)

    pixel_size_y = pxsize
    pixel_size_x = pxsize

    dz_class['dz'] = dzList
    dz_class['dz_mean*area'] = dz2List
    dz_class['mean_dz'] = dzmeanList
    # dz_class['ar'] = arList
    dz_class['dz_px'] = dz_class['dz']*pixel_size_y*pixel_size_x
    # dz_class['mean_dz*area*px*px'] = dz_class['dz_mean*area']*pixel_size_y*pixel_size_x
    # dz_class['ar_px'] = dz_class['ar']*pixel_size_y*pixel_size_x

    return (dz_class)


# count glaciers per size class, get area per size class
def countGlaciers(mGG):
    mG = gpd.read_file(mGG)
    # account for different numbering of HEF in GI5 - keep only toteis (0) and HEF ohne Toteis (2125000), remove "HEF" (2125)
    mG = mG.loc[mG.nr != 2125]
    mG = mG.reset_index()
    G_vsmall = mG.loc[mG.area < 0.5e6]
    G_small = mG.loc[(mG.area >= 0.5e6) & (mG.area < 1e6)]
    G_mid = mG.loc[(mG.area >= 1e6) & (mG.area < 5e6)]
    G_big = mG.loc[(mG.area >= 5e6) & (mG.area < 10e6)]
    G_vbig = mG.loc[(mG.area >= 10e6)]

    df = pd.DataFrame(index=['totNr', 'vsmall', 'small', 'mid', 'big', 'vbig'], columns=['Nr', 'prcNr', 'area', 'areaPrc'])
    df['Nr'] = [mG.shape[0], G_vsmall.shape[0], G_small.shape[0], G_mid.shape[0], G_big.shape[0], G_vbig.shape[0]]
    df['prcNr'] = 100 *df['Nr'] / mG.shape[0]
    df['area'] = [mG.area.sum(), G_vsmall.area.sum(), G_small.area.sum(), G_mid.area.sum(), G_big.area.sum(), G_vbig.area.sum()]
    df['areaPrc'] = 100 *df['area'] / mG.area.sum()

    #print(df)
    return(df)

# get absolute volume per glacier (clip ice thickness raster with shapefiles)
def getVol(ice, shapes, IDs):

    # loop over glacier IDs to get total V per glacier.
    Vol = pd.DataFrame(index=IDs, columns=['VolGI3', 'AreaGI3'])

    for ID in IDs:
        Vol.loc[ID, 'Vol'], Vol.loc[ID, 'count'] = proc.getDZ_ID(ice, shapes[IDs.index(ID):IDs.index(ID)+1])

    # pixel size is 10x10 m, multiply dz by 10x10 to get dV
    Vol['VolGI3'] = Vol['Vol']*100
    Vol['AreaGI3'] = Vol['count']*100

    return (Vol)


# get absolute volume per glacier (clip ice thickness raster with shapefiles)
def getAspect(aspectdem, slopedem, shp):
    
    asp = xdem.DEM(aspectdem)
    slp = xdem.DEM(slopedem)
    glacier_outlines = gpd.read_file(shp)

    df2 = pd.DataFrame(columns=['aspect', 'slope', 'nr'], index=glacier_outlines["nr"].values)
    for gl in glacier_outlines["nr"]:
        # if missing == 'no':
        if gl not in [2167, 2168, 2169, 2170, 2171]:
            print(gl)
            gl_shp = gu.Vector(glacier_outlines[glacier_outlines["nr"] == gl])
            gl_mask = gl_shp.create_mask(asp)

            nmeanasp = circmean(np.deg2rad(asp[gl_mask]))
            nmeanslope = np.ma.mean(slp[gl_mask])

            df2.loc[gl, 'aspect'] = np.rad2deg(nmeanasp)
            df2.loc[gl, 'slope'] = nmeanslope
            df2.loc[gl, 'nr'] = gl

    return (df2)


# -------------------------------------------
# set path to data (dictionary in file "meta.py")
meta = mt.meta

# ice thickness data:
# read raster file with proper crs settings
ice_crs = 'data/GI3_ice_thickness_clip_crs.tif'

# -------------------------------------------


# get area change per size class, make table
cG1 = countGlaciers(meta['GI1']['shp'])  
cG2 = countGlaciers(meta['GI2']['shp'])
cG3 = countGlaciers(meta['GI3']['shp'])
cG5 = countGlaciers(meta['GI5']['shp'])

pd.options.display.float_format = '{:.2f}'.format
dif3_5 = cG5 - cG3
#print(dif3_5.T.round(0))


tab = cG5
tab['area'] = tab['area'] / 1e6
tab['NrChangeG3G5'] = dif3_5['Nr'].astype(int)
tab['ArChangeG3G5'] = dif3_5['area'] / 1e6
tab['ArChangeG3G5_perc'] = 100*(dif3_5['area'] / 1e6) / (cG3['area'] / 1e6)

tab[['Nr', 'NrChangeG3G5']] = tab[['Nr', 'NrChangeG3G5']].round(0)

# THIS IS THE AREA PART OF THE TABLE IN THE MANUSCRIPT
tab.T.to_csv('tables/AreaChangesGI3GI5_table4paper.csv')
print(tab.T)

# get volume change per size class, make table
# function is rewritten to use xdem but code should be cleaned. minor rounding (?) differences for calculation of sum of dz...
df_dz_class  = getDZ_perSizeClass(meta['GI5']['f_dif'], meta['GI3']['shp'], 'isfine', 5)

# convert m3 to km3
df_dz_class['dz_px_km3'] = df_dz_class['dz_px'] * 1e-9
df_dz_class['dz_px_km3_(meandz*ar)'] = df_dz_class['dz_mean*area'] * 1e-9

# print(df_dz_class)

# get volume per size class, make table
Vol_class_2006  = getDZ_perSizeClass('data/GI3_ice_thickness_clip_crs.tif', meta['GI3']['shp'], 'reproject', 10)

Vol_class_2006['vol_px_km3'] = Vol_class_2006['dz_px'] * 1e-9
Vol_class_2006['percLoss'] = 100 * (df_dz_class['dz_px_km3']) / Vol_class_2006['vol_px_km3']

# print(Vol_class_2006)


# combine relevant columns of the tables and print:
# use sum calculations where sum(dz) is multiplied by pixelsize^2, not mean(dz)*area - results are very close to the same (rounding issues?), chose this option for consistency...
tabVol = df_dz_class[['dz_px_km3']]
tabVol['percLoss'] = Vol_class_2006['percLoss']
# get change in m/a per size class, add to table

changerates = getDZ_perSizeClass(meta['GI5']['f_dif_ma'], meta['GI3']['shp'], 'reproject', 5)
tabVol['changerate'] = changerates['mean_dz']
# THIS IS THE volume PART OF THE TABLE IN THE MANUSCRIPT


changerates19972006 = getDZ_perSizeClass(meta['GI3']['f_dif_ma'], meta['GI2']['shp'], 'reproject', 5)
tabVol['changerate19972006'] = changerates19972006['mean_dz']
changerates19961997 = getDZ_perSizeClass(meta['GI2']['f_dif_ma'], meta['GI1']['shp'], 'reproject', 5)
tabVol['changerate19691997'] = changerates19961997['mean_dz']


print(tabVol.T)
tabVol.T.to_csv('tables/VolumeChangesGI3GI5_table4paper.csv')
#stop

# get dA and dV for individual glaciers, produce table.
# get dA for the three time steps.
dA_p1 = get_dA(gpd.read_file(meta['GI1']['shp']), gpd.read_file(meta['GI2']['shp']))
dA_p2 = get_dA(gpd.read_file(meta['GI2']['shp']), gpd.read_file(meta['GI3']['shp']))
dA_p3 = get_dA(gpd.read_file(meta['GI3']['shp']), gpd.read_file(meta['GI5']['shp']))

# merge to one dataframe for all periods
dA_12 = dA_p1.merge(dA_p2, how='outer', on='nr', suffixes=('_1', '_2'))
dA = dA_12.merge(dA_p3, how='outer', on='nr', suffixes=(False, False))
dA.rename(columns={'dA': 'dA_3', 'nr': 'ID'}, inplace=True)
print(dA)
print('bla')
#stop
# xdem version of the function is quite a bit slower, results for vol change are the same. use the other one (my version w shapely loop)
dz_p1 = getDZ(meta['GI2']['f_dif'], meta['GI1']['shp'])
# dz_p1 = getDZ_xdem(meta['GI2']['f_dif'], meta['GI1']['shp'])
dz_p1['ID'] = dz_p1.index
dz_p2 = getDZ(meta['GI3']['f_dif'], meta['GI2']['shp'])
# dz_p2 = getDZ_xdem(meta['GI3']['f_dif'], meta['GI2']['shp'])
dz_p2['ID'] = dz_p2.index
dz_p3 = getDZ(meta['GI5']['f_dif'], meta['GI3']['shp'])
# dz_p3 = getDZ_xdem(meta['GI5']['f_dif'], meta['GI3']['shp'])
dz_p3['ID'] = dz_p3.index

dz_12 = dz_p1.merge(dz_p2, how='inner', on='ID', suffixes=('_1', '_2'))
dz = dz_12.merge(dz_p3, how='inner', on='ID', suffixes=(False, False))
dz.rename(columns={'dV': 'dV_3'}, inplace=True)
print('dz: ', dz.head())
# stop

dAll = dA.merge(dz, on='ID', how='inner')
dAll.rename(columns={'ID': 'nr'}, inplace=True)

# attach GI3 geometries to the file
A_gpd = dAll.merge(gpd.read_file(meta['GI3']['shp']), on='nr', how = 'inner')

All_gpd = gpd.GeoDataFrame(A_gpd, geometry=A_gpd.geometry)


#All_gpd.drop(columns=['Area', 'nr'], inplace=True)

All_gpd = All_gpd.set_crs('epsg:31254')
All_gpd['area'] = All_gpd['geometry'].area
# print(All_gpd)
# print(All_gpd.crs)



# absolute ice volume stuff:
# read shapefile of glacier boundary 2006
GI3 = meta['GI3']['shp']
gpdGI3 = gpd.read_file(GI3)
# read shapefile of glacier boundary 2017
GI5 = meta['GI5']['shp']
gpdGI5 = gpd.read_file(GI5)

# make reprojected file and save - project to crs of the ice thickness raster
gpdGI3_reproj = gpdGI3.to_crs(31287)
gpdGI3_reproj.to_file('data/mergedGI3_31287.shp')

# read reprojected file, export list of shapes
#shapes, IDs = proc.listShapes('/Users/leahartl/Desktop/OGGM_Patrick/Oetztaler_Alpen_GI3_31287.shp', 'ID')
shapes, IDs = proc.listShapes('data/mergedGI3_31287.shp', 'nr')


# loop through clipping function, export df of extracted volume per glacier
Vol = getVol(ice_crs, shapes, IDs)
#print(Vol)
# modify df to merge with gpd of glacier boundaries
Vol['nr'] = Vol.index.values

gpdGI3['area_gpd_GI3'] = gpdGI3.area
gpdGI5['area_gpd_GI5'] = gpdGI5.area
dAll_v2 = gpdGI5[['area_gpd_GI5', 'nr']].merge(gpdGI3[['area_gpd_GI3', 'geometry', 'nr']], on='nr', how='inner')

dAll_v2 = Vol.merge(dAll_v2, on='nr', how='inner')
dAll_v2['area_dif'] = dAll_v2['area_gpd_GI3'] - dAll_v2['area_gpd_GI5']
dAll_v2['area_difPrc'] = dAll_v2['area_gpd_GI5']*100 / dAll_v2['area_gpd_GI3']
# print(dAll_v2.head())
# get glaciers that have lost x% of area: 
# more than half lost: 
half_area = dAll_v2.loc[dAll_v2.area_difPrc <50]
print(half_area)
print('lost more than 50% of area: ', half_area.shape)
# more than 80% lost: 
over80_area = dAll_v2.loc[dAll_v2.area_difPrc <20]
print(over80_area)
print('lost more than 80% of area: ', over80_area.shape)



dz_p3.rename(columns={'ID': 'nr'}, inplace=True)
dAll_v3 = dAll_v2.merge(dz_p3, on='nr', how='inner')


# dAll_v3['dVrate'] = dAll_v3['dV'] / (2017-2006)
dAll_v3['dV/area'] = dAll_v3['dV'] / dAll_v3['area_gpd_GI3']

dAll_v3['VolGI5'] = dAll_v3['VolGI3'] + dAll_v3['dV']

dAll_v3['VolDifPrc'] = dAll_v3['VolGI5']*100 / dAll_v3['VolGI3']
#print(dAll_v3.head())

# get glaciers that have lost x% of volume: 
# more than half lost: 
half_vol = dAll_v3.loc[dAll_v3.VolDifPrc <50]
print(half_vol)
print('lost more than 50% of volume: ', half_vol.shape)
# more than 80% lost: 
over80_vol = dAll_v3.loc[dAll_v3.VolDifPrc <20]
print(over80_vol)
print('lost more than 80% of volume: ', over80_vol.shape)




dfslp = getAspect(meta['GI5']['f_aspect'], meta['GI5']['f_slope'], meta['GI5']['shp'])
print(dfslp)

dAll_v4 = dAll_v3.drop(columns=['geometry']).merge(dfslp, on='nr', how='inner')


dAll_v4.to_csv('tables/VolumeTable_v3.csv')
#stop




print(All_gpd.loc[All_gpd.dV_3 > 0])



# All_gpd.drop(All_gpd[All_gpd.ID == nr].index, inplace=True)
All_gpd[['nr', 'Gletschern', 'dA_1', 'dA_2', 'dA_3', 'dV_1', 'dV_2', 'dV_3']].to_csv('data/output_tabular_1.csv')
# All_gpd[['ID', 'Gletschern', 'dA_1', 'dA_2', 'dA_3', 'dV_1', 'dV_2', 'dV_3', 'geometry']].to_file('boundaries_2006.shp')

fig, ax = plt.subplots(1, 2)
ax=ax.flatten()
ax[0].scatter(All_gpd.dA_1, All_gpd.area)
ax[0].scatter(All_gpd.dA_2, All_gpd.area)
ax[0].scatter(All_gpd.dA_3, All_gpd.area)
ax[1].scatter(All_gpd.dV_1, All_gpd.area)
ax[1].scatter(All_gpd.dV_2, All_gpd.area)
ax[1].scatter(All_gpd.dV_3, All_gpd.area)
plt.show()


stop



