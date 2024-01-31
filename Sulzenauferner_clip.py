import rasterio
from rasterio.plot import show
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import geopandas as gpd
import matplotlib.pyplot as plt
import geoutils as gu
import xdem

import meta as mt


meta = mt.meta
# read shapefile of glacier boundary 2006
GI3 = meta['GI3']['shp']
# GI3 = meta.GI3_y
gpdGI3 = gpd.read_file(GI3)

# read shapefile of glacier boundary 2017
GI5 = meta['GI5']['shp']
gpdGI5 = gpd.read_file(GI5)
# GI5 = meta.GI5_y
# gpdGI5 = gpd.read_file(GI5)


dh_2006 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_20062017.tif'
fn2017_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'

GI3_y =  mt.GI3_y#'/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessFiles2023/mergedOutlines/mergedGI3_withYEAR_3.shp'
GI5_y =  mt.GI5_y#'/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessFiles2023/mergedOutlines/mergedGI5_withYEAR_3.shp'
GI5_y_gpd = gpd.read_file(GI5_y)
# GI5_2_gpd = gpd.read_file('mergedGI5_2.shp')
GI5_y_gpd.YEAR=GI5_y_gpd.YEAR.astype(float)
print(GI5_y_gpd.YEAR.unique())
print(GI5_y_gpd.shape)
# print(GI5_2_gpd.shape)

# gi3 = gpd.read_file(GI3_y)
# sulz_gpd = gi3.loc[gi3.nr == 3032]

outlines = gu.Vector(GI3_y)
# outlines5 = gu.Vector(GI5_y)
dh = xdem.DEM(dh_2006)

sulz_gu = outlines.ds[outlines.ds.nr == 3032]

flight_years = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/data/als_flugjahr_v22/als_flugjahr_v22.shp')
flight_lines = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/flightlines/als1718_fldat_epsg31254.shp')
#print(flight_lines['date'].unique())

fl2018 = flight_years.loc[flight_years.YEAR == '2018']
fl2017 = flight_years.loc[flight_years.YEAR == '2017']

sulz_clip2018 = gpd.clip(sulz_gu, fl2018)
#sulz_clip2018.to_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessFiles2023/mergedOutlines/Sulzenauferner_2018_clip.shp')

sulz_clip2017 = gpd.clip(sulz_gu, fl2017)
#sulz_clip2017.to_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessFiles2023/mergedOutlines/Sulzenauferner_2017_clip.shp')

outlines_clip2018 = gpd.clip(GI5_y_gpd, fl2018)
outlines_clip2017 = gpd.clip(GI5_y_gpd, fl2017)

outlines_flightlines = gpd.clip(flight_lines[['date','geometry']],GI5_y_gpd)
#print(outlines_flightlines)
print(outlines_flightlines['date'].unique())
#outlines_flightlines.to_file('data/outlineswdates.shp')


# flightline plots
def datelines(outlines_flightlines, dh_2006):
    dh = xdem.DEM(dh_2006)

    outlines_flightlines.sort_values(by='date', ascending=True, inplace=True)
    print(outlines_flightlines.head())

    f, a = plt.subplots(1, 1, figsize=(12, 7))
    labels = []
    for i, day in enumerate(outlines_flightlines["date"]):
        # mask dh rasater with flightline shapes:
        gl_shp = gu.Vector(outlines_flightlines[outlines_flightlines["date"] == day])
        gl_mask = gl_shp.create_mask(dh)
        # Derive mean elevation change
        dh_arr = dh.data[gl_mask.data]

        #Remove large outliers
        # dh_arr[np.abs(dh_arr) > 4 * xdem.spatialstats.nmad(dh_arr)] = np.nan
        nonan = dh_arr[~np.isnan(dh_arr)]
        gl_dh = np.median(nonan)
        # gl_dh = np.nanmean(dh_arr)

        # dh_arr = masked[masked<-1000]#.filled(np.nan)
        dh_arr[dh_arr < -100] = np.nan
        print(gl_dh)
        labels.append(day)
        a.boxplot(dh_arr[~np.isnan(dh_arr)], positions=[i], showfliers=False)


    a.set_xticklabels(labels, rotation=45)
    a.set_ylabel('dh (m)')
    a.set_title('dh (2017/18-2006) over GI5 glacier area clipped by flight lines')
    a.grid('both')
    f.savefig('plots/boxplots_flightlines.png')

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    outlines_flightlines.plot(ax=ax, column='date', cmap='nipy_spectral', legend=True)
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((1.2, 0.8, 0.2, 0.2))
    fig.savefig('plots/outlines_flightlines.png')
    



def goneglaciers(outlines, dh_2006, fn2017_5al):
    gone = [2053.0, 2054., 2146.0, 2147.0, 2162.0]
    dh = xdem.DEM(dh_2006)
    dem = xdem.DEM(fn2017_5al)

    gone_gl = outlines.ds.loc[outlines.ds.nr.isin(gone)]
    print(gone_gl)
    print(gone_gl.area*1e-6)
    # aspect = xdem.terrain.aspect(dem)
    # asp=[]
    # nr=[]
    # for i, gl in enumerate(gone_gl["nr"]):
    #     # mask dh rasater with flightline shapes:
    #     gl_shp = gu.Vector(gone_gl[gone_gl["nr"] == gl])
    #     gl_mask = gl_shp.create_mask(aspect)
    #     gl_aspect = np.nanmean(gl_mask)
    #     asp.append(gl_aspect)
    #     nr.append(gl)
    # print(nr)
    # print(aspect)

    
  


# datelines(outlines_flightlines, dh_2006)

# goneglaciers(outlines, dh_2006, fn2017_5al)

# plt.show()




# stop


GI5_y_gpd['area'] = GI5_y_gpd.geometry.area

areaAll = GI5_y_gpd['area'].sum()
area2017 = GI5_y_gpd.loc[GI5_y_gpd['YEAR']==2017]['area'].sum()
area2018 = GI5_y_gpd.loc[GI5_y_gpd['YEAR']==2018]['area'].sum()
areaNan = GI5_y_gpd.loc[np.isnan(GI5_y_gpd['YEAR'])]['area'].sum()
print(GI5_y_gpd.loc[np.isnan(GI5_y_gpd['YEAR'])])

print(areaAll)
print(areaNan)
print(area2018)
print(area2017)

print(area2018+area2017+areaNan)
print('precentage 2017: ', area2017/areaAll )
print('precentage 2018: ', area2018/areaAll )
print('precentage coverage: ', (area2017+area2018)/areaAll )


print(sulz_gu.crs)
print(flight_years.crs)
print(outlines_clip2017.head())

def makeDF(sulz_clip2017, sulz_clip2018, dh):
    dfdV = pd.DataFrame(columns=['dh', 'dV', 'y1', 'y2'], index=[0, 1])
    dfdV['y1'] = 2006
    dfdV['y2'] = [2017, 2018]
    for i, sulz in enumerate([sulz_clip2017, sulz_clip2018]):
        gl_shp = gu.Vector(sulz)
        gl_mask = gl_shp.create_mask(dh)
        # Derive mean elevation change
        gl_dh = np.nanmean(dh.data[gl_mask.data])
        gl_dh_total = gl_dh * sulz.area.values

        dfdV.loc[i, 'dh'] = gl_dh
        dfdV.loc[i, 'dV'] = gl_dh_total[0]

    dfdV['dh/a'] = dfdV['dh'] / (dfdV['y2']-dfdV['y1'])
    dfdV['dV/a'] = dfdV['dV'] / (dfdV['y2']-dfdV['y1'])

    dfdV.to_csv('tables/Sulzenau.csv')
    return(dfdV)

dfdV = makeDF(sulz_clip2017, sulz_clip2018, dh)
weightedavg = dfdV['dV/a'].sum() / sulz_gu.area

print(weightedavg)

print(dfdV)

def figSulzenau(sulz_clip2017, sulz_clip2018):
    fig, ax = plt.subplots(1, 1)
    # fl2018.plot(ax=ax)
    # sulz.plot(ax = ax, alpha=0.7, color="pink")
    sulz_clip2018.plot(ax = ax, alpha=0.7, color="red", label='2018')
    sulz_clip2017.plot(ax = ax, alpha=0.7, color="green", label='2017')
    ax.set_xlim([61000, 65000])
    ax.set_ylim([203000, 206000])
    ax.legend()

    fig.savefig('tables/Sulzenau.png')


def figROI(outlines_clip2017, outlines_clip2018, GI3_y, GI5_y):


    raster = rasterio.open('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif')

    print(raster.shape)
    raster1 = np.reshape(raster.read(), (raster.read().shape[0]*raster.read().shape[1], raster.read().shape[2]))
    outlines = gu.Vector(GI5_y)
    outlines3 = gu.Vector(GI3_y)

    sulz_gu5 = outlines.ds[outlines.ds.nr == 3032]
    sulz_clip2018 = gpd.clip(sulz_gu5, fl2018)
    sulz_clip2017 = gpd.clip(sulz_gu5, fl2017)
    # sulz_clip2018.to_file('Sulzenauferner_2018_clip.shp')

    countries = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp') 
    countries.to_crs(outlines.ds.crs, inplace=True)

    formarker = outlines.ds[outlines.ds.nr == 2152].centroid
    print(formarker)

    RGI = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp')
    RGI.to_crs(outlines.ds.crs, inplace=True)

    cookRoi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/cook_area/cook_area.shp')
    cookRoi.to_crs(outlines.ds.crs, inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    base1 = plt.imshow(raster1, cmap='cividis', vmin=1500, vmax=3400)
    rasterio.plot.show(raster, ax=ax, cmap='cividis', vmin=1500, vmax=3400)
    #flight_years.plot(ax = ax, alpha=0.7, column='JAHR', label='flight years')

    outlines_clip2017.plot(ax = ax, alpha=1, edgecolor="k", facecolor='none', label='2017', zorder=10)
    outlines_clip2018.plot(ax = ax, alpha=1, edgecolor="red", facecolor='none', label='2018', zorder=10)
    outlines3.ds.plot(ax = ax, edgecolor="grey", facecolor='none', linestyle='--', zorder=8)

    RGI.plot(ax = ax, alpha=1, linewidth=0.5, edgecolor="lightblue", facecolor='none', label='RGI', zorder=5)
    cookRoi.plot(ax = ax, alpha=1, edgecolor="blue", facecolor='none', label='region of interest')
    # outlines3.ds.plot(ax = ax, edgecolor="grey", facecolor='none', linestyle='--')
    # countries.plot(ax = ax, alpha=0.7, edgecolor='k', facecolor='none', linestyle='--', linewidth=2)
    #countries.plot(ax = ax, alpha=1, edgecolor="k", facecolor='none')

    #    sulz_clip2018.plot(ax = ax, alpha=0.7, color="red", label='2018')
    # sulz_clip2017.plot(ax = ax, alpha=0.7, color="green", label='2017')
    # cb = plt.colorbar(base1, ax=ax, shrink=0.6, location='bottom')
    cb = plt.colorbar(base1, fraction=0.0346, pad=0.08, location='bottom')
    cb.set_label('Elevation (m)', fontsize=10)

    ax.set_xlim([20000, 75000])
    ax.set_ylim([179000, 229000])
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    ax.grid('both')


    axins = ax.inset_axes([0.8, 0.05, 0.35, 0.35])
    #rasterio.plot.show(raster, ax=axins, cmap='cividis')
    sulz_clip2017.plot(ax = axins, alpha=0.5, edgecolor="k", facecolor='k', label='2017', hatch='//')
    sulz_clip2018.plot(ax = axins, alpha=0.5, edgecolor="k", facecolor='k', label='2018', hatch='+')
    outlines3.ds.plot(ax = axins, edgecolor="grey", facecolor='none', linestyle='--', zorder=11)
    outlines_clip2017.plot(ax = axins, alpha=1, edgecolor="k", facecolor='none',zorder=12)#, categorical=True, legend=True)
    outlines_clip2018.plot(ax = axins, alpha=1, edgecolor="r", facecolor='none',zorder=12)#, categorical=True, legend=True)
    RGI.plot(ax = axins, alpha=1, edgecolor="lightblue", facecolor='none', label='RGI', zorder=5)



    axins.set_xlim([61400, 65200])
    axins.set_ylim([203000, 206200])
    # axins.set_xticklabels([])
    axins.set_yticklabels([])


    axins2 = ax.inset_axes([-0.1, 0.7, 0.4, 0.6])

    #rasterio.plot.show(raster, ax=axins, cmap='cividis')
    countries.plot(ax = axins2, alpha=1, edgecolor="k", facecolor='none')#, categorical=True, legend=True)
    #formarker.plot(ax = axins2, marker='*', color='blue', markersize=200)
    RGI.plot(ax = axins2, alpha=1, linewidth=0.5, edgecolor="lightblue", facecolor='lightblue', label='RGI', zorder=5)
    # ax.set_xlim([20000, 75000])
    # ax.set_ylim([179000, 229000])


    axins2.add_patch(Rectangle((20000, 179000), 75000-20000, 229000-179000, facecolor="grey"))
    axins2.set_xticklabels([])
    axins2.set_yticklabels([])
    axins2.set_xticks([])
    axins2.set_yticks([])
    axins2.set_xlim([500, 510000])
    axins2.set_ylim([120000, 450000])
    axins2.annotate("Austria",
            xy=(300000, 280000), xycoords='data',
            #xytext=(x2, y2), textcoords='data',
            ha="center", va="center")

    
    

    lnGI17 = Line2D([0], [0], linestyle = '-', label='Updated outlines (DEM 2017)', color='k')
    lnGI18 = Line2D([0], [0], linestyle = '-', label='Updated outlines (DEM 2018)', color='r')
    lnGI3 = Line2D([0], [0], linestyle = '--', label='GI3 (2006)', color='grey')
    lnRGI = Line2D([0], [0], linestyle = '-', label='RGI', color='LightBlue')
    lnROI = Line2D([0], [0], linestyle = '-', label='Region of interest', color='blue')
    
    handles = [lnGI17, lnGI18, lnGI3, lnRGI, lnROI]

    # add manual symbols to auto legend
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.9, 0.95), ncol=1)

    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.savefig('plots/ROI.png', bbox_inches='tight', dpi=300)
    
    # ax.legend()

# figSulzenau(sulz_clip2017, sulz_clip2018)
figROI(outlines_clip2017, outlines_clip2018, GI3_y, GI5_y)


plt.show()

