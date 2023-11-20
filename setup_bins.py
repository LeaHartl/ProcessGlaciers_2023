#! /usr/bin/env python3
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap

import geopandas as gpd

import helpers as proc
import meta as mt

## supress copy warning - careful!
pd.options.mode.chained_assignment = None  # default='warn'
##


# SET BINS

Bins = np.arange(2000, 4000, 50)
Bins20 = np.arange(2000, 4000, 20)
BinsDz_all = np.arange(-60, 40, 0.5)
BinsDz = np.arange(-10, 10, 0.05)


# files with ice volume data
ice_crs2006 = 'data/GI3_ice_thickness_clip_crs.tif'

ice_2017 = 'data/ice2017_clip_GI3.tif'
dem2017resampled = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/data/dem17_10m_resampled_to_ice.tif'

ice_2030 = 'data/ice_2030_resamp.tif'
ice_2050 = 'data/ice_2050_resamp.tif'
ice_2075 = 'data/ice_2075_resamp.tif'
ice_2100 = 'data/ice_2100_resamp.tif'

fldr = 'data/'

## initialize dictionary containing  GI info, filenames, etc.
meta = mt.meta

# glacier boundaries
GI5shp = meta['GI5']['shp']
gpdGI5 = gpd.read_file(GI5shp)
gpdGI5_reproj = gpdGI5.to_crs(31287)
# gpdGI5_reproj.to_file('mergedGI5_31287.shp')


def floHoAltAll(Bins, IDs, fname_dif, fname_w, shapes, yrs):
    flHoAlt_list = []
    flHoAltAr_list = []
    flHoAltSum_list = []
    flHoAltMed_list = []
    # for ID in yrs.ID.values:
    for ID in IDs:

        flHoAlt_out, flHoAltAr_out, flHoAltMed_out = proc.getFlHoAlt(fname_dif, fname_w, shapes[IDs.index(ID):IDs.index(ID)+1], Bins)
        flHoAlt_list.append(flHoAlt_out)
        flHoAltAr_list.append(flHoAltAr_out)
        flHoAltMed_list.append(flHoAltMed_out)

        flHoAltSum_out = proc.getFlHoAltSum(fname_dif, fname_w, shapes[IDs.index(ID):IDs.index(ID)+1], Bins)
        flHoAltSum_list.append(flHoAltSum_out)

        ## compute m/a value
        ## print (yrs.Data[yrs.nr == ID].values)
        ## flHoAlt[ID] = flHoAlt[ID]/(yrs.Data[yrs.ID == ID].values)

    flHoAlt = pd.DataFrame(data=flHoAlt_list, columns=Bins, index=IDs)
    flHoAltAr = pd.DataFrame(data=flHoAltAr_list, columns=Bins, index=IDs)
    flHoAltSum = pd.DataFrame(data=flHoAltSum_list, columns=Bins, index=IDs)
    flHoAltMed = pd.DataFrame(data=flHoAltMed_list, columns=Bins, index=IDs)
    # print(flHoAltAr)
    # print(flHoAltAr.max())
    # print(flHoAltAr.sum())
    # print(flHoAltAr.sum().sum())
    return  (flHoAlt.T, flHoAltAr.T, flHoAltSum.T, flHoAltMed.T)




# # # loop through dict and add new content
for i, GI in enumerate(meta):
    if GI != 'GI1': # skip GI1
    # if GI == 'GI5': # ONLY FOR TESTS!!!
        # print(GI)
        meta[GI]['shapes'],  meta[GI]['IDs'] = proc.listShapes(meta[GI]['shp'], 'nr')

        # use f_dif_ma here to consistently account for acquisition date
        meta[GI]['flHoDZ'] = proc.getFlHo(meta[GI]['f_dif_ma'], meta[GI]['shapes'], meta[GI]['IDs'], BinsDz, meta[GI]['yrs'])
        # meta[GI]['flHoDZ'].to_csv(fldr+'flHoDZ_'+str(meta[GI]['yrs'][1])+'.csv')

        # # # use f_dif ma... s.o.
        # # 20m bins 
        # # meta[GI]['flHoAlt'], meta[GI]['flHoAltAr'], meta[GI]['flHoAltSum']= floHoAltAll(Bins20, meta[GI]['IDs'], meta[GI]['f_dif'], meta[GI]['f_dem'][1], meta[GI]['shapes'], meta[GI]['yrs'])
        # # meta[GI]['flHoAlt'].to_csv(fldr+'/flHoAlt_20_'+str(meta[GI]['yrs'][1])+'.csv')
        # # meta[GI]['flHoAltAr'].to_csv(fldr+'/flHoAltAr_20_'+str(meta[GI]['yrs'][1])+'.csv')
        # # meta[GI]['flHoAltSum'].to_csv(fldr+'/flHoAltSum_20_'+str(meta[GI]['yrs'][1])+'.csv')
        #  50m bins
        # meta[GI]['flHoAlt'], meta[GI]['flHoAltAr'], meta[GI]['flHoAltSum'], meta[GI]['flHoAltMed']= floHoAltAll(Bins, meta[GI]['IDs'], meta[GI]['f_dif_ma'], meta[GI]['f_dem'][1], meta[GI]['shapes'], meta[GI]['yrs'])
        # meta[GI]['flHoAlt'].to_csv(fldr+'/flHoAlt_50_'+str(meta[GI]['yrs'][1])+'.csv')
        # meta[GI]['flHoAltAr'].to_csv(fldr+'/flHoAltAr_50_'+str(meta[GI]['yrs'][1])+'.csv')
        # meta[GI]['flHoAltSum'].to_csv(fldr+'/flHoAltSum_50_'+str(meta[GI]['yrs'][1])+'.csv')


# reproject shapefiles: 
def reprSHP(meta):
    for i, GI in enumerate(meta):
        sh = gpd.read_file(meta[GI]['shp'])
        sh.to_crs(epsg=31287, inplace=True)
        sh.to_file('data/merged'+GI+'_31287.shp')

reprSHP(meta)
# stop

def getIce(icefile, outfilename, shpfl):
    shpsRepr, IDSrepr = proc.listShapes(shpfl, 'nr')
    flHoAlt_ice, flHoAltAr_ice, flHoAltSum_ice, flHoAltMed_ice = floHoAltAll(Bins, IDSrepr, icefile, dem2017resampled, shpsRepr, meta['GI5']['yrs'])
    flHoAlt_ice.to_csv(fldr+'/flHoAlt_50_ice_'+outfilename+'.csv')
    flHoAltAr_ice.to_csv(fldr+'/flHoAltAr_50_ice_'+outfilename+'.csv')
    flHoAltSum_ice.to_csv(fldr+'/flHoAltSum_50_ice_'+outfilename+'.csv')
    flHoAltMed_ice.to_csv(fldr+'/flHoAltMed_50_ice_'+outfilename+'.csv')

getIce(ice_2030, 'proj2030', 'data/mergedGI5_31287.shp')
getIce(ice_2050, 'proj2050', 'data/mergedGI5_31287.shp')
getIce(ice_2075, 'proj2075', 'data/mergedGI5_31287.shp')
getIce(ice_2100, 'proj2100', 'data/mergedGI5_31287.shp')

getIce(ice_crs2006, 'data2006', 'data/mergedGI3_31287.shp')
getIce(ice_2017, 'data2017', 'data/mergedGI5_31287.shp')


def sanitycheck(fl, pxsz):
    df = pd.read_csv(fl)
    print(df.head())
    print(df.sum())
    print(df.sum().sum())
    print(df.sum().sum()*pxsz*pxsz)

sanitycheck(fldr+'/flHoAltAr_50_'+str(meta['GI5']['yrs'][1])+'.csv', 5)
sanitycheck(fldr+'/flHoAltAr_50_ice_data2017.csv', 10)
sanitycheck(fldr+'/flHoAltAr_50_ice_data2006.csv', 10)
sanitycheck(fldr+'/flHoAltAr_50_ice_proj2075.csv', 10)

# stop



