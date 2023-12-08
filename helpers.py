#! /usr/bin/env python3
import numpy as np
import pandas as pd
import xarray as xr
import os
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from matplotlib.colors import ListedColormap
import fiona
import os
# os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

import rioxarray
# from matplotlib import pyplot
import affine
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.plot import show_hist
from rasterio.crs import CRS
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT
from rasterio.plot import plotting_extent
from rasterio.merge import merge

from rasterstats import zonal_stats

from rasterio.plot import show
import json

# import gisexport_helpers as gh

# import seaborn as sns

### helper functions to process geodata

fldr = '/Volumes/Seagate Expansion Drive/IGF projekte/DEMS_OG/'


## this makes a dataframe similar to arcgis zonal stats/tabulate area output. needs fname of DEM to process, 
## list of shapes, IDs to match glaciers, Bins for tabulating. yrs is a list of year values used to convert to m/a. make list of ones if not desired or adapt script.
def getFlHo(fname, shapes, IDs, Bins, yrs):

    #flHodf= pd.DataFrame(index= Bins[:-1])
    hist = []
    ids = []
    with rasterio.open(fname) as src:

        for i in range(len(shapes)):
            shp = shapes[i:i+1]

            out_image, out_transform = mask(src, shp, crop=True, nodata = np.nan)
            out_meta = src.meta

            ## turn into 2D array
            ar = out_image.ravel()
            # print(ar)
           
            HistogramData = np.histogram(ar,Bins)
            hist.append(HistogramData[0])
            ids.append(IDs[i])
            # print(HistogramData)
            # add array of Hist values to DF. glacier ID is df col name.
            # flHodf[IDs[i]] = HistogramData[0]
        # print(len(hist))
        # print(len(ids))
        
        flHodf = pd.DataFrame(data = hist, columns=Bins[:-1], index=ids)
            
        # print(flHodf.T.head())
        # stop
    return (flHodf.T)

## this makes series of mean/ median dz per elevation bin and counts pixels
def getFlHoAlt(dif, dem, shapes, BinsAlt):

    ## open dif raster
    with rasterio.open(dif) as src:
            dif_image, dif_transform = mask(src, shapes, crop=True, nodata = np.nan)
            dif_meta = src.meta
            dif_image = dif_image.reshape(dif_image.shape[1], dif_image.shape[2])
    
    ## open dem raster
    with rasterio.open(dem) as src_2:
            dem_image, dem_transform = mask(src_2, shapes, crop=True, nodata = np.nan)
            dem_meta = src.meta
            dem_image = dem_image.reshape(dem_image.shape[1], dem_image.shape[2])
            # plt.imshow(dem_image)
            # plt.show()

    src.close()
    src_2.close()

    ## write to xarray for further processing
    dm = xr.DataArray(dem_image, dims = ['x', 'y'])
    df = xr.DataArray(dif_image, dims = ['x', 'y'])

    d= xr.concat([dm, df], 'images')

    data = xr.Dataset({'dem':dm, 'dif':df})
    rr_median=[]
    rr_mean=[]
    aa=[]
    bins = []
    for i, b in enumerate(BinsAlt[:-1]):
        dat = data.dif.where((data.dem>=BinsAlt[i]) & (data.dem<BinsAlt[i+1]))

        r_mean = dat.mean()
        r_med = dat.median()
        rr_median.append(r_med.item(0))
        rr_mean.append(r_mean.item(0))
        bins.append(b)


        a = dat.count()
        # print(a.item(0))
        aa.append(a.item(0))

    
    rm = pd.Series(rr_mean, index= bins)
    ar = pd.Series(aa, index= bins)
    # print(ar)

    rmd = pd.Series(rr_median, index= bins)

    return (rm, ar, rmd)  

## this makes 1) series of sum dz per elevation bin. needs dem, dif-raster, shapefile.
def getFlHoAltSum(dif, dem, shapes, BinsAlt):

    ## open dif raster
    with rasterio.open(dif) as src:
            dif_image, dif_transform = mask(src, shapes, crop=True, nodata = np.nan)
            dif_meta = src.meta
            dif_image = dif_image.reshape(dif_image.shape[1], dif_image.shape[2])
    
    ## open dem raster
    with rasterio.open(dem) as src_2:
            dem_image, dem_transform = mask(src_2, shapes, crop=True, nodata = np.nan)
            dem_meta = src.meta
            dem_image = dem_image.reshape(dem_image.shape[1], dem_image.shape[2])
            # plt.imshow(dem_image)
            # plt.show()

    src.close()
    src_2.close()

    ## write to xarray for further processing
    dm = xr.DataArray(dem_image, dims = ['x', 'y'])
    df = xr.DataArray(dif_image, dims = ['x', 'y'])

    d= xr.concat([dm, df], 'images')

    data = xr.Dataset({'dem':dm, 'dif':df})
    rr=[]
    aa=[]
    bins = []
    
    for i, b in enumerate(BinsAlt[:-1]):
        r= data.dif.where((data.dem>=BinsAlt[i]) & (data.dem<BinsAlt[i+1])).sum()
        rr.append(r.item(0))
        bins.append(b)

    
    rs = pd.Series(rr, index= BinsAlt[:-1])
    # ar = pd.Series(aa, index= BinsAlt)

    return (rs)

def SetCRSIceThk(fn):
    crs = rasterio.crs.CRS({"init": "epsg:31287"})
    src = rasterio.open(fn, mode='r+')
    src.crs = crs
    out_meta = src.profile
    # print(src.shape)
    # print(src.crs)
    # plt.imshow(src.read(1), cmap='pink')
    # plt.show()
 ## writing
    with rasterio.open('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/GI3_ice_thickness_clip_crs.tif','w',**out_meta) as ff:
        ff.write(src.read(1),1)


# def ClipIceThk(fn):


def getDZ_ID(dif, shapes):

    # ## open dif raster
    with rasterio.open(dif) as src:
        dif_image, dif_transform = mask(src, shapes, crop=True, nodata = np.nan)
        dif_meta = src.meta
        dif_image = dif_image.reshape(dif_image.shape[1], dif_image.shape[2])
            # print(dif_image.crs)
            # shapes.plot()
            #plt.imshow(dif_image)
            #plt.show()
            #stop
    
    src.close()
    
    # ## write to xarray for further processing
    df = xr.DataArray(dif_image, dims = ['x', 'y'])

    dz = df.sum()
    ar = df.count()

    return (dz.item(0), ar.item(0))


def getDZ_ID2(dif, shapes):

    # ## open dif raster
    with rasterio.open(dif) as src:
        pixel_size_x, pixel_size_y = src.res
        print(pixel_size_y, pixel_size_x)

        dif_image, dif_transform = mask(src, shapes, crop=True, nodata = np.nan)
        dif_meta = src.meta
        dif_image = dif_image.reshape(dif_image.shape[1], dif_image.shape[2])
            # print(dif_image.crs)
            # shapes.plot()
            # plt.imshow(dif_image)
            # plt.show()
            # stop
    
    src.close()
    
    # ## write to xarray for further processing
    df = xr.DataArray(dif_image, dims = ['x', 'y'])

    dz = df.sum()
    ar = df.count()

    return (dz.item(0), ar.item(0), pixel_size_y, pixel_size_x)


## read shape files to clip with , export as shapes and IDs in lists.
def listShapes(GGname, idcol):
    with fiona.open(GGname, "r") as shapefile:
        # print(shapefile.schema)
        # nope
        shapes = [feature["geometry"] for feature in shapefile]
        IDs = [feature["properties"][idcol] for feature in shapefile ]
        # print(IDs)

    shapes = list(shapes)
    return(shapes, IDs)




## this takes 2 DEM filenames as input and makacts fname 1 from fname 2, writing to new tiff. assumes aligned DEMs. 
def subtractRaster(dem1, dem2, out):

    with rasterio.open(dem1) as src:
        D1 = src.read(1, masked=True)
        out_meta = src.profile

    with rasterio.open(dem2) as src2:
        D2 = src2.read(1, masked=True)


    # if 2rast == True:
    #   with rasterio.open(keydem) as src3:
    #       key = src3.read(1, masked=True)

    ## subtract
    dz = D2 - D1
    ## remove ouliers with change more than ±200m (faulty values occur e.g. when dem extent is different or glaciers are missing)
    ## supress run time warning generated by nan in comparison
    #with np.errstate(invalid='ignore'):
     #   dz[dz>60] = np.nan
      #  dz[dz<-200] = np.nan


    ## writing
    with rasterio.open('data/dz_'+out+'.tif','w',**out_meta) as ff:
        ff.write(dz,1)

    return (dz)


## this takes 2 DEM filenames as input and makacts fname 1 from fname 2, writing to new tiff. assumes aligned DEMs. 
def addRaster(dem1, dem2, out):

    with rasterio.open(dem1) as src:
        D1 = src.read(1, masked=True)
        out_meta = src.profile
        show(D1)

    with rasterio.open(dem2) as src2:
        D2 = src2.read(1, masked=True)
        show(D2)


    # if 2rast == True:
    #   with rasterio.open(keydem) as src3:
    #       key = src3.read(1, masked=True)

    ## add
    dz = D2 + D1
    show(dz)
    ## remove ouliers with change more than ±200m (faulty values occur e.g. when dem extent is different or glaciers are missing)
    ## supress run time warning generated by nan in comparison
    #with np.errstate(invalid='ignore'):
     #   dz[dz>60] = np.nan
      #  dz[dz<-200] = np.nan


    ## writing
    with rasterio.open('data/'+out+'.tif','w',**out_meta) as ff:
        ff.write(dz,1)

    return (dz)


## this clips raster to (multi) polygon shapefile, input filename of rasters and list of shapes, output is array, 
## writes to new geotiff.
def clipRaster(dem, shapes, fname):

    with rasterio.open(dem) as src:
        #show(src)
        print(src.shape)

        out_image, out_transform = mask(src, shapes, crop=True, nodata = np.nan)
        out_meta = src.meta
        out_meta.update({'nodata': -999,
                 'height' : out_image.shape[1],
                 'width' : out_image.shape[2],
                 'transform' : out_transform}) 
        #show(out_image)
        print(out_image.shape)
        out_image = out_image.reshape(out_image.shape[1], out_image.shape[2])
        #plt.show()
        # out_image[out_image<0] = np.nan

            ## writing
    with rasterio.open('data/'+fname+".tif",'w',**out_meta) as ff:
        ff.write(out_image,1)

    return (out_image)


## This takes a list of GEOTIFF files and warps them to common dimenions, resolution, projection. Use carefully. 
## based on https://rasterio.readthedocs.io/en/stable/topics/virtual-warping.html#normalizing-data-to-a-consistent-grid
def warpDEMS(input_files, outfile):

    ## data1 = data set to be used as "destination". other are warped to match this.
    data1 = rasterio.open(input_files[0])
    # dst_crs = CRS.from_epsg(31254)
    dst_crs = data1.crs
    # print(data1.crs)

    ## Use data 1 bounding box
    dst_bounds = data1.bounds

    ## Output image dimensions, use data 1 dimensions (= resolution)
    dst_height = data1.height
    dst_width = data1.width

    ## Output image transform
    left, bottom, right, top = dst_bounds
    xres = (right - left) / dst_width
    yres = (top - bottom) / dst_height
    dst_transform = affine.Affine(xres, 0.0, left,
                                  0.0, -yres, top)

    ## options for rasterio vrt. Resampling method can be set here.
    vrt_options = {
        'resampling': Resampling.cubic,
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
        'dtype': data1.dtypes[0]
        
    }

    for path in input_files:

        with rasterio.open(path) as src:

            with WarpedVRT(src, **vrt_options) as vrt:

                # At this point 'vrt' is a full dataset with dimensions,
                # CRS, and spatial extent matching 'vrt_options'.

                # Read all data into memory.
                data = vrt.read()

                # Process the dataset in chunks.  Likely not very efficient.
                for _, window in vrt.block_windows():
                    data = vrt.read(window=window)

                # Dump the aligned data into a new file.  A VRT representing
                # this transformation can also be produced by switching
                # to the VRT driver.
                directory, name = os.path.split(path)
                # outfile = os.path.join(directory, fn+'{}'.format(name))
                rio_shutil.copy(vrt, outfile, driver='GTiff')


def divideVol(dem1, dem2, out):
    with rasterio.open(dem1) as src:
        D1 = src.read(1, masked=True)
        out_meta = src.profile
        print(D1.shape)

    with rasterio.open(dem2) as src2:
        D2 = src2.read(1, masked=True)
        D2 = D2 #/ (2017-2006)
        print(D2.shape)

    D3 = D1 / D2
    # plt.imshow(D3)
    # plt.show()

    ## writing
    with rasterio.open('data/'+out+'.tif','w',**out_meta) as ff:
        ff.write(D3,1)

    return (D3)







