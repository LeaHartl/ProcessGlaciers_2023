import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import geoutils as gu # note: version 0.0.12
import xdem # note: version 0.0.10
import meta as mt

# import dictionary with paths to files:
meta = mt.meta

# ice thickness data:
# read raster file with proper crs settings
ice_crs = 'data/GI3_ice_thickness_clip_crs.tif'



df_dA = pd.read_csv('tables/areachange.csv')
df_dA.index = df_dA['nr']
df_A = pd.read_csv('tables/glacierarea.csv')
df_A.index = df_A['nr']

df_dV97 = pd.read_csv('tables/volumechange1969_1997.csv', index_col=0)
df_dV06 = pd.read_csv('tables/volumechange1997_2006.csv', index_col=0)
df_dV17 = pd.read_csv('tables/volumechange2006_2017.csv', index_col=0)




# merge area data:
merged = pd.concat([df_A[['area_1997', 'area_2006', 'area_201718']], df_dA[['201718-06']]], axis=1, join="inner")
# remove glaciers that have nan area value in 201718
# merged = merged1.dropna(subset='area_201718')


def SizeClasses(merged, colname, colname2):
    All = merged.copy()
    G_toosmall = merged.loc[merged[colname] < 0.01e6]
    G_vsmall = merged.loc[merged[colname] < 0.5e6]
    G_small = merged.loc[(merged[colname] >= 0.5e6) & (merged[colname] < 1e6)]
    G_mid = merged.loc[(merged[colname] >= 1e6) & (merged[colname] < 5e6)]
    G_big = merged.loc[(merged[colname] >= 5e6) & (merged[colname] < 10e6)]
    G_vbig = merged.loc[(merged[colname] >= 10e6)]

    df = pd.DataFrame(index=['totNr', 'toosmall', 'vsmall', 'small', 'mid', 'big', 'vbig'], columns=['Nr', 'prcNr', 'area', 'areaPrc'])
    df['Nr'] = [merged.shape[0], G_toosmall.shape[0], G_vsmall.shape[0], G_small.shape[0], G_mid.shape[0], G_big.shape[0], G_vbig.shape[0]]
    df['prcNr'] = 100 *df['Nr'] / merged.shape[0]
    if colname2 ==colname:
        df['abs_val'] = [merged[colname2].sum(), G_toosmall[colname2].sum(), G_vsmall[colname2].sum(), G_small[colname2].sum(), G_mid[colname2].sum(), G_big[colname2].sum(), G_vbig[colname2].sum()]
        df['percentage'] = 100 *df['abs_val'] / merged[colname2].sum()
        return(df)
    
    else:
        df['abs_val'] = [merged[colname2].sum(), G_toosmall[colname2].sum(), G_vsmall[colname2].sum(), G_small[colname2].sum(), G_mid[colname2].sum(), G_big[colname2].sum(), G_vbig[colname2].sum()]
    
        # df['abs_val'] = [merged[colname2].mean(), G_toosmall[colname2].mean(), G_vsmall[colname2].mean(), G_small[colname2].mean(), G_mid[colname2].mean(), G_big[colname2].mean(), G_vbig[colname2].mean()]
        return(df[['Nr', 'abs_val']])
    





# produce table with area changes by size classes
def AreaSizeClass(merged):

    merged1 = merged.dropna(subset='area_201718')
    merged2 = merged.dropna(subset='area_2006')
    df17 = SizeClasses(merged1, 'area_201718', 'area_201718')
    df06 = SizeClasses(merged2, 'area_2006', 'area_2006')

    table1 = df17.copy()
    table1['NrChange'] = df17.Nr - df06.Nr
    table1['Ar_Change_Abs'] = df17.abs_val - df06.abs_val
    table1['Ar_loss_perc'] = 100*(df17.abs_val - df06.abs_val) / df06.abs_val

    # convert m2 to km2
    table1['area'] = table1['area'] / 1e6
    table1['Ar_Change_Abs'] = table1['Ar_Change_Abs'] / 1e6

    print(table1)
    table = table1.T
    table = table.round(2)
    print(table)
    table.to_csv('tables/areachange_4paper.csv')



# produce table with volume changes by size classes
def VolumeSizeClass(merged, df_dV97, df_dV06, df_dV17):

    #nr,dh,dV,y1,y2,dh/a,dV/a

    print(df_dV17)
    print(merged)

    merged_V17 = merged.dropna(subset='area_2006')
    merged_V17 = pd.concat([merged_V17[['area_2006', 'area_201718']], df_dV17], axis=1, join="inner")
    merged_V17['dV/a_w'] = merged_V17['dV/a'] / merged_V17['area_2006']


    merged_V06 = merged.dropna(subset='area_1997')
    merged_V06 = pd.concat([merged_V06[['area_2006', 'area_201718']], df_dV06], axis=1, join="inner")
    merged_V06['dV/a_w'] = merged_V06['dV/a'] / merged_V06['area_2006']
    print(merged_V17)
    # merged2 = merged.dropna(subset='area_2006')
    df17 = SizeClasses(merged_V17, 'area_201718', 'dV/a')

    df06 = SizeClasses(merged_V06, 'area_2006', 'dV/a')

    print(df17)
    print(df06)
    stop




# get volume change per size class
def getDZ_perSizeClass(dif, shp, rep, pxsize):
    mG = gpd.read_file(shp)
    if rep == 'reproject':
        mG = mG.to_crs(31287)
    mG = mG.reset_index()
    
    All = mG
    G_toosmall = mG.loc[mG.area < 0.01e6]
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

    dz = xdem.DEM(dif)
    for cl in classes: 
        print(cl)
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



# get volume change per size class, make table
# function is rewritten to use xdem/gu. minor rounding (?) differences for calculation of sum of dz.
df_dz_class  = getDZ_perSizeClass(meta['GI5']['f_dif'], meta['GI3']['shp'], 'isfine', 5)

# convert m3 to km3
df_dz_class['dz_px_km3'] = df_dz_class['dz_px'] * 1e-9
df_dz_class['dz_px_km3_(meandz*ar)'] = df_dz_class['dz_mean*area'] * 1e-9

print('here', df_dz_class)

# get volume per size class, make table
Vol_class_2006  = getDZ_perSizeClass('data/GI3_ice_thickness_clip_crs.tif', meta['GI3']['shp'], 'reproject', 10)

Vol_class_2006['vol_px_km3'] = Vol_class_2006['dz_px'] * 1e-9
Vol_class_2006['percLoss'] = 100 * (df_dz_class['dz_px_km3']) / Vol_class_2006['vol_px_km3']

print('here2', Vol_class_2006)


# combine relevant columns of the tables and print:
# sum calculations where sum(dz) is multiplied by pixelsize^2, chose this option for consistency v binning script?
# use mean(dz)*area - results are very close to the same (rounding issues?)
#tabVol = df_dz_class[['dz_px_km3']]
tabVol = df_dz_class[['dz_px_km3_(meandz*ar)']]
tabVol['percLoss'] = Vol_class_2006['percLoss']
# get change in m/a per size class, add to table

changerates = getDZ_perSizeClass(meta['GI5']['f_dif_ma'], meta['GI3']['shp'], 'reproject', 5)
tabVol['changerate'] = changerates['mean_dz']
# THIS IS THE volume PART OF THE TABLE IN THE MANUSCRIPT

print('here3', changerates)

changerates19972006 = getDZ_perSizeClass(meta['GI3']['f_dif_ma'], meta['GI2']['shp'], 'reproject', 5)
tabVol['changerate19972006'] = changerates19972006['mean_dz']
print('here4', tabVol)

test = gpd.read_file(meta['GI1']['shp'])
print(test)

changerates19961997 = getDZ_perSizeClass(meta['GI2']['f_dif_ma'], meta['GI1']['shp'], 'reproject', 5)
tabVol['changerate19691997'] = changerates19961997['mean_dz']


print(tabVol.T)
tabVol.T.to_csv('tables/VolumeChangesGI3GI5_table4paper_2.csv')
stop


#AreaSizeClass(merged)

VolumeSizeClass(merged, df_dV97, df_dV06, df_dV17)

stop
# subroutine to get area change from area, write to dataframe
def get_dA2(dfarea):
    df = pd.DataFrame(columns=['nr', '1997-69', '2006-97', '201718-06'])
    df['nr'] = dfarea['nr']
    df['1997-69'] = dfarea['area_1997'] - dfarea['area_1969'] 
    df['2006-97'] = dfarea['area_2006'] - dfarea['area_1997']
    df['201718-06'] = dfarea['area_201718'] - dfarea['area_2006']

    return (df)


# subroutine to get area, write to dataframe
def get_A(gp1, gp2, gp3, gp5):
    # generate dataframe with dA for each glacier
    gp1['area'] = gp1['geometry'].area
    gp2['area'] = gp2['geometry'].area
    gp3['area'] = gp3['geometry'].area
    gp5['area'] = gp5['geometry'].area

    # merge the dataframes:
    both = gp1[['nr', 'area']].merge(gp2[['nr', 'area']], how='outer', on='nr', suffixes=('_1969', '_1997'))
    both2 = gp3[['nr', 'area']].merge(gp5[['nr', 'area']], how='outer', on='nr', suffixes=('_2006', '_201718'))

    both3 = both.merge(both2, how='outer', on='nr')

    # check for glaciers that disappeared in the most recent outlines and write to separate df:
    gone = both2.loc[np.isnan(both2.area_201718)]
    # account for different numbering of HEF in GI5 - HEF did not disappear. // IDs fixed in all GI, no longer needed!
    # gone = gone.loc[gone.nr != 2125]
    return (both3, gone)


# subroutine to get volume change (dh: mean vol change over glacier area. dV: dh*area --> total vol change.)
def getdV(outlines, dh_fn, y1, y2, missing):
    glacier_outlines = gu.Vector(outlines)
    dh = xdem.DEM(dh_fn)

    dfdV = pd.DataFrame(columns=['dh', 'dV', 'y1', 'y2'], index=glacier_outlines.ds['nr'])

    dfdV['y1'] = y1

    # if calculation is being done for most recent time step, use year of survey as stated in the
    # file for each glacier. year might be 2017 or 2018. most are 2017 but part of stubai was surveyed in 2018.
    if y2 == 2017:
        dfdV['y2'] = glacier_outlines.ds['YEAR'].astype(int).values

    # otherwise use year as passed to the function.
    else:
        dfdV['y2'] = y2

    # loop through each individual glacier in the outlines:
    # get mean elevation change for each glacier
    # multiply mean elev change with area to get totvol change
    # write both to a data frame where each row is a glacier

    for gl in glacier_outlines.ds['nr']:
        # check if it is a year with missing glaciers and if so, exclude them. This refers to DEMs for 2006 and 
        # 2017/18, which do not cover 5 very small glaciers at the edge of the ROI that are covered by the earlier DEMs.
        # the 'if not in' statement just skips the missing glaciers to avoid the loop breaking, this could be streamlined. 
        if missing == 'yes':
            if gl not in [2167, 2168, 2169, 2170, 2171]:
                print(gl)
                gl_shp = gu.Vector(glacier_outlines.ds[glacier_outlines.ds["nr"] == gl])
                gl_mask = gl_shp.create_mask(dh)

                # Derive mean elevation change
                gl_dh = np.nanmean(dh.data[gl_mask.data])
                gl_dh_total = gl_dh * gl_shp.area.values

                dfdV.loc[gl, 'dh'] = gl_dh
                dfdV.loc[gl, 'dV'] = gl_dh_total[0]
        else:
            print(gl)
            gl_shp = gu.Vector(glacier_outlines.ds[glacier_outlines.ds["nr"] == gl])
            gl_mask = gl_shp.create_mask(dh)

            # Derive mean elevation change
            gl_dh = np.nanmean(dh.data[gl_mask.data])
            gl_dh_total = gl_dh * gl_shp.area.values

            dfdV.loc[gl, 'dh'] = gl_dh
            dfdV.loc[gl, 'dV'] = gl_dh_total[0]

    # compute values per year by dividing elev and vol change by the number of years between DEMs:
    dfdV['dh/a'] = dfdV['dh'] / (dfdV['y2']-dfdV['y1'])
    dfdV['dV/a'] = dfdV['dV'] / (dfdV['y2']-dfdV['y1'])


    # deal with Sulzenau Ferner - this was partially surveyed in 2017 and partially in 2018. 
    # USE THE 2017 VALUES. most of the glacier was surveyed in 2017 and the 2017 area covers 
    # the full elevation range, 2018 does not.

    # --> if second year is 2017 (i.e., if calculation is being done for 2006-2017/18 time step), replace dh/a and dV/a
    if y2 == 2017:
        # read csv previously generated for Sulzenauferner
        sul=pd.read_csv('tables/Sulzenau.csv', index_col='Unnamed: 0')

        # print(sul['dh/a'].loc[sul.y2==2017])
        # print(dfdV.loc[3032, 'dh/a'])
        # replace value with the value from the file for only the area surveyed in 2017:
        dfdV.loc[3032, 'dV/a'] = sul['dV/a'].loc[sul.y2==2017].values[0]


    dfdV.to_csv('tables/volumechange'+str(y1)+'_'+str(y2)+'.csv')
    return(dfdV)


#------Area part ------
# # get A for the three time steps
# These are the paths to the shapefiles:
GI1 = meta['GI1']['shp']
GI2 = meta['GI2']['shp']
GI3 = meta['GI3']['shp']
GI5 = meta['GI5']['shp']

# load shapes as geodataframes and pass to get_A function:
dfarea, gone = get_A(gpd.read_file(GI1), gpd.read_file(GI2), gpd.read_file(GI3), gpd.read_file(GI5))
# write area of each glacier in each inventory year to csv:
dfarea.to_csv('tables/glacierarea.csv')

# get Area change for all time steps - subtract area of t2 from area of t1:
dfdA = get_dA2(dfarea)
# write area change to csv:
dfdA.to_csv('tables/areachange.csv')

#print stuff to check it looks as intended:
print(dfdA.head())
print(dfarea.head())
print(gone)


#------Volume part ------
# get volume change for all time steps and write to csv)
# get file paths:
# this is a GI3 shapefile with the GI5 survey year of each glacier - used in computation of annual change:
GI3_y = mt.GI3_y
# file paths to dif dems
dh_1969 = meta['GI2']['f_dif']
dh_1997 = meta['GI3']['f_dif']
dh_2006 = meta['GI5']['f_dif']

# pass files and years for each time step to function. Yes/No indicates whether glaciers are missing in the DEM. 
# For 2006 and 2017, 5 very small glaciers are not covered. 
dfdV19691997 = getdV(GI1, dh_1969, 1969, 1997, 'no')
dfdV19972006 = getdV(GI2, dh_1997, 1997, 2006, 'yes')
dfdV20062017 = getdV(GI3_y, dh_2006, 2006, 2017, 'yes')

# reality check:
# read csv of volume change produced in above function
# dfdV20062017 = pd.read_csv('tables/volumechange'+str(2006)+'_'+str(2017)+'.csv', index_col='nr')
# print(dfdV20062017.loc[dfdV20062017['dh']>0])


