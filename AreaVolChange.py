import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import geoutils as gu # note: version 0.0.12
import xdem # note: version 0.0.10
import meta as mt

# import dictionary with paths to files:
meta = mt.meta


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
    # multiply mean elev change with area to get total  vol change
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
# print(dfdA.head())
# print(dfarea.head())
# print(dfarea.shape)
print(dfarea.count())
print(dfarea.sum())
# print(gone)

stop
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


