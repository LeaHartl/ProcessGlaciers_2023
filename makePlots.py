#! /usr/bin/env python3
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.colors as colors

import fiona

import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import numpy as np
import pandas as pd

import geopandas as gpd

import geoutils as gu
import xdem
import helpers as proc
import meta as mt



meta = mt.meta
# read shapefile of glacier boundary 2006
GI3 = meta['GI3']['shp']
gpdGI3 = gpd.read_file(GI3)

# read shapefile of glacier boundary 2017
GI5 = meta['GI5']['shp']
gpdGI5 = gpd.read_file(GI5)

# print(gpdGI3.head())
# print(gpdGI5.head())

# read volume table
Vol = pd.read_csv('tables/VolumeTable_v3.csv', index_col='Unnamed: 0')
# print(Vol.columns)
# print(Vol.shape)
#print(Vol.loc[Vol['area_dif']<0])
#Vol.loc[Vol['area_dif']<0] = np.nan
growing = Vol.loc[Vol['dV']>0]
print('growing: ', growing[['dV', 'area_dif', 'nr', 'area_gpd_GI5', 'area_gpd_GI3']])
# print(Vol.loc[Vol['dV']>0])

#print(Vol.loc[Vol['VolDifPrc']<0])

goneGlaciers = Vol.loc[Vol['VolDifPrc']<0]
print('gone glaciers:', goneGlaciers)
print('gone glaciers, nr:', goneGlaciers.shape)
# stop
## this count includes the "problem glaciers...."
print(Vol.loc[Vol['VolDifPrc']<50])
print('50%, nr:', Vol.loc[Vol['VolDifPrc']<50].shape)
print('80%, nr:', Vol.loc[Vol['VolDifPrc']<20].shape)

#Vol.loc[Vol['VolDifPrc']<0] = np.nan
# dAll2.drop(columns=['geometry']).to_csv('VolumeTable.csv')
print(Vol.loc[Vol['VolDifPrc']==np.nan])

# gpdGI3['area_gpd_GI3'] = gpdGI3.area
# gpdGI5['area_gpd_GI5'] = gpdGI5.area
# dAll = gpdGI5[['area_gpd_GI5','ID']].merge(gpdGI3[['area_gpd_GI3', 'geometry', 'ID']], on='ID', how='inner')
# # print(dAll)
# dAll['area_dif'] = dAll['area_gpd_GI3'] - dAll['area_gpd_GI5']
# dArea = dAll[['area_gpd_GI3', 'area_gpd_GI5','area_dif', 'geometry', 'ID']]
# # print(dArea)
# print(dArea.loc[dArea['area_dif']<0])
# # stop

def mapFig(gpdGI3, gpdGI5, Vol, goneGlaciers):
    GI5merge = gpdGI5[['geometry', 'nr']].merge(Vol, on='nr', how='inner')
    # GI5Gone = gpdGI5[['geometry', 'ID']].merge(goneGlaciers, on='ID', how='inner')

    GI5merge['dif2'] = 100-GI5merge['VolDifPrc']
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))#, sharex=True, sharey=True)
    #ax = ax.flatten()
    vmin = 0
    print(GI5merge.loc[GI5merge['VolDifPrc']<0])
    vmax = 100

    cmap = cm.get_cmap('magma_r', int((vmax-vmin)/10))
    cmap.set_over('green')
    #for i, s in enumerate(scen):
        # GG_merge.plot(column=s+'_2', ax=ax[i], legend=False, cmap=cmap, vmin=vmin, vmax=vmax)
        # c = s + col
    # dArea.loc[dArea['area_dif']<0].plot(ax=ax, legend=False, edgecolor='red', facecolor='red')
    #gpdGI3.plot(ax=ax, legend=False, edgecolor='k', facecolor='white')
    gpdGI5.plot(ax=ax, legend=False, edgecolor='k', facecolor='white')
    GI5merge.plot(column='dif2', ax=ax, legend=False, cmap=cmap, vmin=vmin, vmax=vmax)
    # GI5Gone.plot(ax=ax, legend=False, edgecolor='k', facecolor='green')
    
    ax.set_title('Volume loss 2006-2017/18')
        #ax[i].grid()

    fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.91, 0.3, 0.02, 0.4])
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.4])
    cbim = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
                        cax=cbar_ax, orientation='vertical', extend='max')
    cbim.set_label('Difference as % of 2006 volume')
    fig.savefig('plots/VolLoss2006_2017.png', bbox_inches='tight')
    
def mapFig2(gpdGI3, gpdGI5, Vol, goneGlaciers):
    GI5merge = gpdGI5[['geometry', 'nr']].merge(Vol, on='nr', how='inner')
    # GI5Gone = gpdGI5[['geometry', 'ID']].merge(goneGlaciers, on='ID', how='inner')

    GI5merge['dZ/a'] = GI5merge['dV/area']/(2017-2006)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))#, sharex=True, sharey=True)
    #ax = ax.flatten()
    vmin = -20
    print(GI5merge[['dV/area']])
    vmax = 0

    cmap = cm.get_cmap('magma', 10)
    # cmap.set_over('green')
    #for i, s in enumerate(scen):
        # GG_merge.plot(column=s+'_2', ax=ax[i], legend=False, cmap=cmap, vmin=vmin, vmax=vmax)
        # c = s + col
    # dArea.loc[dArea['area_dif']<0].plot(ax=ax, legend=False, edgecolor='red', facecolor='red')
    #gpdGI3.plot(ax=ax, legend=False, edgecolor='k', facecolor='white')
    GI5merge.plot(ax=ax, legend=False, edgecolor='k', facecolor='white')
    GI5merge.plot(column='dV/area', ax=ax, legend=False, cmap=cmap, vmin=vmin, vmax=vmax)
    # GI5Gone.plot(ax=ax, legend=False, edgecolor='k', facecolor='green')
    
    ax.set_title('Mean dZ per glacier, 2006-2017')
        #ax[i].grid()

    fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.91, 0.3, 0.02, 0.4])
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.4])
    cbim = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax)),
                        cax=cbar_ax, orientation='vertical', extend='max')
    cbim.set_label('m')
    fig.savefig('plots/dZperglaicer2006_2017.png', bbox_inches='tight')


def ScatterFig(gpdGI3, gpdGI5, Vol, goneGlaciers):
    GI5merge = gpdGI5[['geometry', 'nr']].merge(Vol, on='nr', how='inner')
    # GI5Gone = gpdGI5[['geometry', 'ID']].merge(goneGlaciers, on='ID', how='inner')

    # GI5merge = GI5merge1.merge(gpdGI3[['geometry', 'nr']], on='nr', how='inner')
    # fig, ax = plt.subplots(1, 1, figsize=(8, 7))#, sharex=True, sharey=True)
    #ax = ax.flatten()
    
    print(GI5merge.columns)
    subset = GI5merge[['geometry', 'area_gpd_GI5', 'area_dif', 'dV', 'nr']]
    subset['areaGI5'] = subset['geometry'].area
    subset.drop(columns='geometry', inplace=True)


    print(gpdGI3.columns)
    print(subset.columns)

    subset1 = gpdGI3[['geometry', 'nr']].merge(subset, on='nr', how='inner')
    # print(subset1.columns)
    subset1['areaGI3'] = subset1['geometry'].area

    subset1 = subset1.sort_values('areaGI5', ascending=True)

    subset1['csm_dA'] = (subset1['areaGI3'] - subset1['areaGI5']).cumsum()
    subset1['csm_dV'] = subset1['dV'].cumsum()

    dVtotal = subset1['dV'].sum()
    dAtotal = (subset1['areaGI3'] - subset1['areaGI5']).sum()

    subset1['dA_prc'] = 100 * (subset1['areaGI3'] - subset1['areaGI5']) / dAtotal
    subset1['dV_prc'] = 100 * subset1['dV'] / dVtotal

    subset1['csm_dA_prc'] = subset1['dA_prc'].cumsum()
    subset1['csm_dV_prc'] = subset1['dV_prc'].cumsum()
    return(subset1)


def ScatterFig2(gpdGI3, gpdGI5, Vol, goneGlaciers):
    GI5merge = gpdGI5[['geometry', 'nr']].merge(Vol, on='nr', how='inner')
    # GI5Gone = gpdGI5[['geometry', 'ID']].merge(goneGlaciers, on='ID', how='inner')

    # GI5merge = GI5merge1.merge(gpdGI3[['geometry', 'nr']], on='nr', how='inner')
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))#, sharex=True, sharey=True)
    #ax = ax.flatten()
    
    print(GI5merge.columns)
    subset = GI5merge[['geometry', 'area_gpd_GI5', 'area_dif', 'dV', 'nr']]
    subset['areaGI5'] = subset['geometry'].area
    subset.drop(columns='geometry', inplace=True)

    subset1 = subset.merge(gpdGI3[['geometry', 'nr']], on='nr', how='inner')

    subset1['areaGI3'] = subset1['geometry'].area
    subset1 = subset1.sort_values('areaGI5', ascending=False)

    subset1['csm_dA'] = (subset1['areaGI3'] - subset1['areaGI5']).cumsum()
    subset1['csm_dV'] = subset1['dV'].cumsum()

    dVtotal = subset1['dV'].sum()
    dAtotal = (subset1['areaGI3'] - subset1['areaGI5']).sum()

    subset1['dA_prc'] = 100 * (subset1['areaGI3'] - subset1['areaGI5']) / dAtotal
    subset1['dV_prc'] = 100 * subset1['dV'] / dVtotal

    subset1['csm_dA_prc'] = subset1['dA_prc'].cumsum()
    subset1['csm_dV_prc'] = subset1['dV_prc'].cumsum()

    cmap = cm.get_cmap('viridis', 8)

    print(subset1.shape)
    print(subset1.duplicated())

    print(subset1.loc[(subset1['dA_prc']<2) & (subset1['areaGI5']> 4*1e6)])
    print(subset1.loc[(subset1['csm_dA_prc']<40) & (subset1['csm_dV_prc']>55)]) #(subset1['areaGI5']< 4*1e6)& (subset1['areaGI5']> 2*1e6)])
    # print(subset1.shape())
    vmax = 0
    vmin = -20

    # ax.plot(subset1['areaGI5'], subset1['csm_dA_prc'])
    # plt.show()
    # stop

    sc = ax.scatter(subset1['dA_prc'], subset1['dV_prc'], s=subset1['areaGI5']/1e4, c=subset1['areaGI5']/1e6, cmap=cmap, vmin=0, vmax=16)
    ax.set_xlabel('area change (% of total)')
    ax.set_ylabel('volume change (% of total)')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.grid('both')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('GI5 area, km2')
    fig.savefig('plots/Scatter_notcumulative.png', bbox_inches='tight')

## this makes  series of mean, median, ... dz per elevation bin..
def getHyps(dif, dem, shapes, BinsAlt):
    # print(shapes)
    ## open dif raster
    with rasterio.open(dif) as src:
            dif_image, dif_transform = mask(src, shapes, crop=True, nodata = np.nan)
            dif_meta = src.meta
            dif_image = dif_image.reshape(dif_image.shape[1], dif_image.shape[2])
            # print(src.crs)
            # plt.imshow(dif_image)
            # plt.show()
    
    ## open dem raster
    with rasterio.open(dem) as src_2:
            dem_image, dem_transform = mask(src_2, shapes, crop=True, nodata = np.nan)
            dem_meta = src.meta
            dem_image = dem_image.reshape(dem_image.shape[1], dem_image.shape[2])

    src.close()
    src_2.close()

    ## write to xarray for further processing
    dm = xr.DataArray(dem_image, dims = ['x', 'y'])
    df = xr.DataArray(dif_image, dims = ['x', 'y'])

    d = xr.concat([dm, df], 'images')
    data = xr.Dataset({'dem':dm, 'dif':df})

    rr=[]
    sd=[]
    count=[]
    bins = []
    sums = []
    
    for i, b in enumerate(BinsAlt[:-1]):
        dat = data.dif.where((data.dem>=BinsAlt[i]) & (data.dem<BinsAlt[i+1]))

        r = dat.mean(skipna=True)
        rr.append(r.item(0))

        s = dat.std(skipna=True)
        sd.append(s.item(0))
        
        cn = dat.count()
        count.append(cn.item(0))

        sm = dat.sum(skipna=True)
        sums.append(sm.item(0))

        bins.append(b)

    # print(count)
    
    rs = pd.Series(rr, index= bins)
    sd = pd.Series(sd, index= bins)
    count = pd.Series(count, index=bins)
    sms = pd.Series(sums, index=bins)

    print(count)
    print(count.sum()*25)
    
    return (rs, sd, count, sms)  







#makes gridspec figure with 3 subplots, dz (m/a) map, histogram, hypsometry of mean dz.
def figDZmap(meta):

    outlines5 = gu.Vector(meta['GI5']['shp'])
    outlines3 = gu.Vector(meta['GI3']['shp'])

    fig = plt.figure(constrained_layout=False, figsize=(10, 7))
    spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec2[:, :2])

    ax2 = fig.add_subplot(spec2[0, 2])
    ax3 = fig.add_subplot(spec2[1, 2])
    # f2_ax4 = fig.add_subplot(spec2[1, 1])

    norm = colors.TwoSlopeNorm(vcenter=0, vmin=-3, vmax=0.1)

    with fiona.open(meta['GI3']['shp'], "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    raster = rasterio.open('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/data/dif_5m_20062017_ma_Clip.tif')
    raster1 = np.reshape(raster.read(), (raster.read().shape[0]*raster.read().shape[1], raster.read().shape[2]))

    base1 = ax1.imshow(raster1, cmap='RdYlBu', norm=norm)
    rasterio.plot.show(raster, ax=ax1, cmap='RdYlBu', norm=norm)

    outlines3.ds.plot(ax=ax1, alpha=1, edgecolor="grey", facecolor='none', linestyle='--', linewidth=0.5)
    outlines5.ds.plot(ax=ax1, alpha=1, edgecolor="k", facecolor='none', linewidth=0.5)
    
    ax1.set_xlim([20000, 75000])
    ax1.set_ylim([179000, 229000])
    ax1.set_xlabel('m')
    ax1.set_ylabel('m')
    ax1.grid('both')


    dz = xdem.DEM('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/data/dif_5m_20062017_ma_Clip.tif')
    # print(dz.data)
    bins=np.arange(-6.5, 1, 0.25)
    hs, be = np.histogram(dz.data, bins=bins)

    # convert counts (hs) to area for plot
    ax2.bar(bins[0:-1], hs*25*1e-6, color='slategray', edgecolor='k', width=0.2)
    ax2.grid('both')
    ax2.set_ylabel('Area (km2)')
    ax2.set_xlabel('Mean annual elevation change (m/a)')
    ax2.set_xlim([-6.5, 0.5])
    ax2.yaxis.set_label_position("right")
    #ax2.xaxis.set_label_position("top")
    ax2.yaxis.tick_right()


    dat, sd, count, sm = getHyps('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/data/dif_5m_20062017_ma_Clip.tif', '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/data/aligned_resampled_2_5m_tir1718_Clip.tif', 
        gpd.read_file(meta['GI3']['shp']).geometry, np.arange(2000, 4000, 50))

    print(count.sum()*25)
    
    # shift up by half bin size (25) to plot values in the middle of respective elevation bins
    ax3.plot(25+dat.values, dat.index, color='slategray', zorder=4)#, edgecolor='k')
    # ax3.plot(dat.values, dat.index, color='slategray')#
    ax3.fill_betweenx(25+dat.index, dat.values-sd.values, dat.values+sd.values, alpha=0.2)
    ax3.grid('both')
    ax3.set_ylabel('Elevation (m)')
    ax3.set_xlabel('Mean. annual elevation change (m/a)')
    ax3.yaxis.set_label_position("right")
    ax3.set_xlim([-6.5, 0.5])
    ax3.yaxis.tick_right()
    ax3.set_ylim([2000, 3800])

    # ax4 = ax3.twiny()
    # ax4.plot(ice2006.sum(axis=1)*10*10, ice2006.index.values.astype(int), label = '2006', color='k', linestyle='--')
    # ax4.plot(ice2017.sum(axis=1)*10*10, ice2017.index.values.astype(int), label = '2017/18', color='k')
    # ax4.set_ylim([2100, 3700])
    # ax4.legend()
    # cbar_ax = fig.add_axes([0.2, 0.07, 0.4, 0.02])
    # cb=fig.colorbar(cm.ScalarMappable(cmap='RdYlBu', norm=norm), cax=cbar_ax, orientation='horizontal', extend='both')
    # cb.ax.set_xscale('linear')
    # cbar_ax.set_xlabel('Mean annual elevation change 2006-2017/18 (m/a)', fontsize=10)

    cbar_ax = ax1.inset_axes([0.1, -0.12, 0.8, 0.02])
    cb=fig.colorbar(cm.ScalarMappable(cmap='RdYlBu', norm=norm), cax=cbar_ax, orientation='horizontal', extend='both')
    cb.ax.set_xscale('linear')
    cbar_ax.set_xlabel('Mean annual elevation change 2006-2017/18 (m/a)', fontsize=10)



    lbl1 = ax1.annotate('a)', xy=(25000, 224000),  xycoords='data',
                       fontsize=15, horizontalalignment='left', verticalalignment='top',)
    lbl2 = ax2.annotate('b)', xy=(-5, 34),  xycoords='data',
                       fontsize=15, horizontalalignment='left', verticalalignment='top',)
    lbl3 = ax3.annotate('c)', xy=(-5, 3600),  xycoords='data',
                       fontsize=15, horizontalalignment='left', verticalalignment='top',)

    fig.savefig('plots/dz_ma_map.png', bbox_inches='tight', dpi=300)

# makes gridspec figure with 4 subplots: area by aspect, vol change (%) by aspect, by slope & cumulative vol and area change
def figRelChange(meta, df):
 
    fig = plt.figure(constrained_layout=False, figsize=(10, 7))
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    ax3 = fig.add_subplot(spec2[1, 0])
    ax4 = fig.add_subplot(spec2[1, 1])
    ax1 = fig.add_subplot(spec2[0, 0], projection='polar')
    ax2 = fig.add_subplot(spec2[0, 1], projection='polar')


    # print(count.sum()*25)
    # print(count5.sum()*25)
    # # count[360] = count[0]
    # # count5[360] = count5[0]
    # print(count)

    # this clips the change raster by the 2006 and 20017/18 outlines and bins pixals by aspect.
    # output: dat -> mean dz, sd -> standard dev of dz, count-> number of pixels, sm -->sum of dz
    # multiply pix count by pix size (5mx5m) to get area.
    dat, sd, count, sm = getHyps(meta['GI5']['f_dif_ma'], meta['GI5']['f_aspect'], 
        gpd.read_file(meta['GI3']['shp']).geometry, np.arange(0, 360+22.5, 22.5))

    dat5, sd5, count5, sm5 = getHyps(meta['GI5']['f_dif_ma'], meta['GI5']['f_aspect'], 
        gpd.read_file(meta['GI5']['shp']).geometry, np.arange(0, 360+22.5, 22.5))


    # this adds the value of the series' first row as an extra row at the end of the series 
    # to achieve a connected line in the first polar plot
    def addrowforplot(count):
        count_ix = pd.Series(index=np.arange(0, 360+22.5, 22.5))
        count_ix.iloc[:-1] = count.values
        count_ix.iloc[-1] = count.iloc[0]
        print(count.iloc[0])
        return(count_ix)

    smprc = 100*sm/sm.sum()
    print(smprc)

    count_ix = addrowforplot(count)
    count_ix5 = addrowforplot(count5)
    smprc_ix = addrowforplot(smprc)
    sm_ix = addrowforplot(sm)
    print(count_ix)
    print(smprc_ix)
    # shift by half bin size so it plots in center of bin
    ax1.plot(np.radians(count_ix.index+22.5/2), count_ix*25*1e-6, color='slategray', label='area 2006 ($km^2$)')
    ax1.plot(np.radians(count_ix5.index+22.5/2), count_ix5*25*1e-6, color='k', label='area 2017/18 ($km^2$)')
    ax1.plot(np.radians(smprc_ix.index+22.5/2), smprc_ix, color='r', label='% of total change')
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(260)
    ax1.legend()
    ax1.grid(True)
    # ax3.set_title('glacier area')

    # color map for size classes:
    cmap = cm.get_cmap('viridis')
    bounds = np.array([0, 0.5, 1, 5, 10, 15, 20])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    # correct for negative values due to errors in 2006 volume, set negative vol dif to 0
    df.VolDifPrc.loc[df.VolDifPrc<0]=0

    # percentage volume change, mean aspect (polar plot)
    ax2.scatter(np.radians(df.aspect.values), 100-df.VolDifPrc.values, s=20*1e-6*df.area_gpd_GI5.values, c =df.area_gpd_GI5/1e6, cmap=cmap, norm=norm)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.set_rlabel_position(224)
    # position label of radial axis:
    label_position=ax2.get_rlabel_position()
    ax2.text(np.radians(label_position+20),ax2.get_rmax()/2.,'%',
        rotation=label_position,ha='center',va='center')
    ax2.grid(True)

    # percentage volume change against mean slope
    ax3.scatter(df.slope.values, 100-df.VolDifPrc.values, s=20*1e-6*df.area_gpd_GI5.values, c =df.area_gpd_GI5/1e6, cmap=cmap, norm=norm)
    ax3.set_xlabel('Mean slope (°)')
    ax3.set_ylabel('Volume loss, percentage of 2006 (%)')
    ax3.grid(True)

    # cumulative area change vs cumul. vol. change (cumulative data generated in "ScatterFig" subfunction.)
    subset1 = ScatterFig(gpdGI3, gpdGI5, Vol, goneGlaciers)
    sc = ax4.scatter(subset1['csm_dA_prc'], subset1['csm_dV_prc'], s=20*subset1['areaGI5']/1e6,
                    c=subset1['areaGI5']/1e6, cmap=cmap, norm=norm)
    ax4.set_xlabel('Cumulative area change (% of total)')
    ax4.set_ylabel('Cumulative volume change (% of total)')
    ax4.set_xlim([0, 105])
    ax4.set_ylim([0, 105])
    ax4.grid('both')
    cb = fig.colorbar(sc, ax=ax4)
    cb.set_label('2017/18 area ($km^2$)')


    lbl1 = ax1.annotate('a)', xy=(np.radians(310), 25),  xycoords='data',
                        fontsize=15, horizontalalignment='left', verticalalignment='top',)
    lbl2 = ax2.annotate('b)', xy=(np.radians(275), 100),  xycoords='data',
                        fontsize=15, horizontalalignment='left', verticalalignment='top',)
    lbl3 = ax3.annotate('c)', xy=(12, 100),  xycoords='data',
                        fontsize=15, horizontalalignment='left', verticalalignment='top',)
    lbl4 = ax4.annotate('d)', xy=(10, 100),  xycoords='data',
                        fontsize=15, horizontalalignment='left', verticalalignment='top',)

    fig.savefig('plots/rel_change.png', bbox_inches='tight', dpi=300)

# helper function, count glaciers where sum of ice vol over all evelation bins = 0
def getnum(ice_dat):
    gln2030 = ice_dat.sum()
    return(len(gln2030.loc[gln2030==0.0]))


def plotVolArHyps(goneGlaciers):
    # color map for size classes:
    cmap = cm.get_cmap('viridis')
    bounds = np.array([0, 0.5, 1, 5, 10, 15, 20])
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    yr = ['2006', '2017/18', '2030', '2050', '2075', '2100']
    icedata_sum = mt.icedata_sum
    icedata_mean = mt.icedata_mean
    icedata_median = mt.icedata_median
    icedata_ar = mt.icedata_ar

    remove = goneGlaciers.nr.values.astype(str)
    print(remove)





    fig, ax = plt.subplots(2, 3, figsize=(10, 7), sharey= True, sharex= True)
    ax = ax.flatten()

    for i, ice in enumerate(icedata_sum):
        # if i > 0:
            # ice.drop(columns = remove, inplace=True)
            # icedata_ar[i].drop(columns = remove, inplace=True)
            #icedata_sum[i].drop(columns = remove, inplace=True)

        elev_maxvol = ice[ice>0].idxmax()
        glsize = icedata_ar[i].sum()*10*10
        glvol = icedata_sum[i].sum()*10*10

        sc = ax[i].scatter(glvol/1e9, elev_maxvol+25, c =glsize/1e6, cmap=cmap, norm=norm, s=30*1e-6*glsize.values)
        

        # for i, ice in enumerate(icedata_sum):
        print(yr[i], ', zero sum: ', ice.shape, getnum(ice))

        sm = ice.sum()
        greater0 = sm[sm>0]
        zero = sm[sm==0]
        print(zero.index)
        # print(len(n))
        ax[i].set_title(yr[i])#+ ', n = '+ str(len(greater0)))

    for a in ax:
        a.grid('both', zorder=0)
        a.set_xscale('log')

    ax[0].set_ylabel('Elevation (m)')
    ax[3].set_ylabel('Elevation (m)')
    ax[3].set_xlabel('Ice volume ($km^3$)')
    ax[4].set_xlabel('Ice volume ($km^3$)')
    ax[5].set_xlabel('Ice volume ($km^3$)')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    cb = fig.colorbar(sc, cax=cbar_ax)
    cb.set_label('Area ($km^2$)')
    fig.savefig('plots/VolArIndividualGlaciers.png', bbox_inches='tight', dpi=300)


def plotHyps(goneGlaciers):
    yr = ['2006', '2017/18', '2030', '2050', '2075', '2100']
    cl = ['slategray', 'k', 'slategray', 'k', 'slategray', 'k']
    ls = ['-', '-', '--', '--', ':', ':']
    icedata_sum = mt.icedata_sum
    icedata_mean = mt.icedata_mean
    icedata_median = mt.icedata_median
    icedata_ar = mt.icedata_ar

    fig, ax = plt.subplots(1, 3, figsize=(10, 7), sharey= True, sharex= False)
    ax = ax.flatten()

    for i, ice in enumerate(icedata_sum):
        print(yr[i], ', zero sum: ', ice.shape, getnum(ice))

    for i, ice in enumerate(icedata_sum):
        # if (yr[i] == '2006') | (yr[i] == '2017/18'):
        #     ls = '-'
        # else:
        #     ls = '--'
        # print(ls[i])
        # ice[ice==0]=np.nan
        ax[0].plot(ice.sum(axis=1)*10*10*1e-9, (25+ice.index.values).astype(int), label=yr[i], linestyle=ls[i] , color=cl[i])
        sm1 = ice.sum(axis=1)*100*1e-9
        # print(sm1)
        # print(sm1.values-0.3*sm1.values)
        ax[0].fill_betweenx(25+sm1.index, sm1.values-0.3*sm1.values, sm1.values+0.3*sm1.values, alpha=0.2)
 
        # exclude HEF and Gepatsch - just checking to see how that affects the mean at low elevations.
        #icedata_mean_excl = icedata_mean[i].drop(['2125000.0', '7022.0'], axis=1)#, errors='ignore')
        
        ax[2].plot(icedata_mean[i].mean(axis=1), (25+icedata_mean[i].index.values).astype(int), label=yr[i], linestyle=ls[i] , color=cl[i])
        # ax[1].plot(icedata_median[i].median(axis=1), icedata_median[i].index.values.astype(int), label=yr[i], linestyle=ls[i] , color=cl[i])
        # ax[1].plot(icedata_mean_excl.mean(axis=1), icedata_mean_excl.index.values.astype(int), label=yr[i], linestyle=ls)

        # normed = icedata_mean[i] / (100*icedata_ar[i])
        # # glsm = ice.sum()
        # countgl = ice[ice > 0.0].count(axis=1)
        # # print(countgl)
        # # print(normed)
        ax[1].plot(100*icedata_ar[i].sum(axis=1)*1e-6, (25+icedata_ar[i].index.values).astype(int), label=yr[i], linestyle=ls[i] , color=cl[i])
        # ax[2].plot(countgl.values, countgl.index.values.astype(int), label=yr[i], linestyle=ls)

    # sm1 = icedata_sum[1].sum(axis=1)
    # print(sm1)
    # print(sm1.values-0.3*sm1.values)
    # ax[0].fill_betweenx(25+sm1.index, sm1.values-0.3*sm1.values, sm1.values+0.3*sm1.values, alpha=0.2)
 
    ax[0].legend()
    ax[0].set_xlabel('Ice volume ($km^3$)')
    ax[0].set_title('Ice volume per elevation bin')
    # ax[1].legend()
    ax[1].set_xlabel('Ice area ($km^2$)')
    ax[1].set_title('Glacier area per elevation bin')
    # ax[2].legend()
    ax[2].set_xlabel('mean ice thickness (m')
    ax[2].set_title('Mean ice thickness')

    for a in ax:
        a.grid('both')
        a.set_ylim([2000, 3800])
    fig.savefig('plots/HypsometryConstantChange.png', bbox_inches='tight', dpi=300)


#
# figDZmap(meta)
# figRelChange(meta, Vol)
plotVolArHyps(goneGlaciers)
plotHyps(goneGlaciers)


# def plot_dzGl(dat, fn):
#     fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharey= True, sharex= True)
#     # ax = ax.flatten()

#     # dat['GI3'].plot()
#     # refgl = ['2125', '2129', '2133', '7022', '2074']
#     # glnm = ['HEF', 'KWF', 'VF', 'GEP', 'GF']
#     clr = ['b', 'k', 'g']

#     # mn = dat[GI]
#     lbl = ['1969-97', '1997-2006', '2006-17']

#     for i, GI in enumerate(dat):
#         if i==0:
#             dat[GI]=dat[GI]/(1997-1969)

#         if i==1:
#             dat[GI]=dat[GI]/(2006-1997)
#         if i==2:
#             dat[GI]=dat[GI]/(2017-2006)
#         mn_dZ = dat[GI].mean(axis=0)


#         ax.scatter(mn_dZ, dat[GI].index.values.astype(int),
#                           color=clr[i], label='meean dZ per glacier, '+lbl[i])#,  color= 'k', alpha = 1., label = glc)
#         # ax.fill_betweenx(dat[GI].index.values.astype(int), mn, mx, color=clr[i], alpha=0.4, label='dZ range, '+lbl[i])
#             #ax[i].set_xlim([-6, 2.5])
#         # for j, g in enumerate(refgl):
#         #     ax[i].scatter((dat[GI][g].values).astype(float), dat[GI].index.values.astype(int), s=10,
#         #                   color=clr[j], label=glnm[j])
            
#         # ax[i].set_xlim([-8,8])
#         ax.set_ylabel('Elevation (m)')
#         ax.set_xlabel('elevation change (m/a)')
#     # ax[0].set_title('1969-97')
#     ax.legend()
#     ax.grid('both')

#     fig.savefig('Range.png')


mapFig(gpdGI3, gpdGI5, Vol, goneGlaciers)
mapFig2(gpdGI3, gpdGI5, Vol, goneGlaciers)

# ScatterFig(gpdGI3, gpdGI5, Vol, goneGlaciers)
# ScatterFig2(gpdGI3, gpdGI5, Vol, goneGlaciers)

plt.show()
