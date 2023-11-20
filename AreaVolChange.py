import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import geoutils as gu
import xdem
import meta as mt

meta = mt.meta

# subroutine to get area change from area, write to dataframe
def get_dA2(dfarea):
     #nr      area_1969      area_1997      area_2006    area_201718
    df = pd.DataFrame(columns=['nr', '1997-69', '2006-97', '201718-06'])
    df['nr'] = dfarea['nr']
    df['1997-69'] = dfarea['area_1997'] - dfarea['area_1969'] 
    df['2006-97'] = dfarea['area_2006'] - dfarea['area_1997']
    df['201718-06'] = dfarea['area_201718'] - dfarea['area_2006']

    return (df)


# subroutine to get area, write to dataframe
def get_A(gp1, gp2, gp3, gp5):
    # print(gp1.head())
    # print(gp2.head())
    # generate dataframe with dA for each glacier
    gp1['area'] = gp1['geometry'].area
    gp2['area'] = gp2['geometry'].area
    gp3['area'] = gp3['geometry'].area
    gp5['area'] = gp5['geometry'].area


    print(gp3.shape, gp5.shape)

    both = gp1[['nr', 'area']].merge(gp2[['nr', 'area']], how='outer', on='nr', suffixes=('_1969', '_1997'))
    both2 = gp3[['nr', 'area']].merge(gp5[['nr', 'area']], how='outer', on='nr', suffixes=('_2006', '_201718'))

    both3 = both.merge(both2, how='outer', on='nr')

    # print(both2.head(), both2.shape)
    # print(both2.loc[np.isnan(both2.area_201718)])

    # gone glaciers: 
    gone = both2.loc[np.isnan(both2.area_201718)]
    # account for different numbering of HEF in GI5 - HEF did not disappear.
    gone = gone.loc[gone.nr != 2125]
    # print(gone)
    return (both3, gone)


# subroutine to get volume change (dh: mean vol change over glacier area. dV: dh*area --> total vol change.)
def getdV(outlines, dh_fn, y1, y2, missing):
    glacier_outlines = gu.Vector(outlines)
    dh = xdem.DEM(dh_fn)

    dfdV = pd.DataFrame(columns=['dh', 'dV', 'y1', 'y2'], index=glacier_outlines.ds['nr'])

    dfdV['y1'] = y1

    if y2 == 2017:
        dfdV['y2'] = glacier_outlines.ds['YEAR'].astype(int).values

    else:
        dfdV['y2'] = y2

    for gl in glacier_outlines.ds['nr']:
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

    dfdV['dh/a'] = dfdV['dh'] / (dfdV['y2']-dfdV['y1'])
    dfdV['dV/a'] = dfdV['dV'] / (dfdV['y2']-dfdV['y1'])


    # deal with Sulzenau Ferner:
    # if second year is 2017 (if calculation is being done for 2006-2017/18 time step), replace dh/a and dV/a
    if y2 == 2017:
        sul=pd.read_csv('tables/Sulzenau.csv', index_col='Unnamed: 0')
        print(sul)
        print(sul['dh/a'].loc[sul.y2==2017])
        print(dfdV.loc[3032, 'dh/a'])

        #dfdV.loc[3032, 'dh/a'] = sul['dh/a'].loc[sul.y2==2017].values[0]
        dfdV.loc[3032, 'dV/a'] = sul['dV/a'].loc[sul.y2==2017].values[0]

    print(dfdV)
    print(dfdV.loc[dfdV.index==3032])
    dfdV.to_csv('tables/volumechange'+str(y1)+'_'+str(y2)+'.csv')
    return(dfdV)



# # get A for the three time steps:
GI1 = meta['GI1']['shp']
GI2 = meta['GI2']['shp']
GI3 = meta['GI3']['shp']
GI5 = meta['GI5']['shp']

dfarea, gone = get_A(gpd.read_file(GI1), gpd.read_file(GI2), gpd.read_file(GI3), gpd.read_file(GI5))
dfarea.to_csv('tables/glacierarea.csv')

# # get Area change for all time steps:
dfdA = get_dA2(dfarea)
dfdA.to_csv('tables/areachange.csv')
print(dfdA.head())
print(dfarea.head())
print(gone)

# get volume change for all time steps (writes to csv)
GI3_y = mt.GI3_y
dh_1969 = meta['GI2']['f_dif']
dh_1997 = meta['GI3']['f_dif']
dh_2006 = meta['GI5']['f_dif']

dfdV19691997 = getdV(GI1, dh_1969, 1969, 1997, 'no')
dfdV19972006 = getdV(GI2, dh_1997, 1997, 2006, 'yes')
dfdV20062017 = getdV(GI3_y, dh_2006, 2006, 2017, 'yes')

# read csv of volume change produced in above function
dfdV20062017 = pd.read_csv('tables/volumechange'+str(2006)+'_'+str(2017)+'.csv', index_col='nr')
print(dfdV20062017.loc[dfdV20062017['dh']>0])



# no longer needed, previous versions of the functions not using GU/XDEM
# # get dA for the three time steps. 
# dA_p1 = get_dA(gpd.read_file(GI1), gpd.read_file(GI2))
# dA_p2 = get_dA(gpd.read_file(GI2), gpd.read_file(GI3))
# dA_p3 = get_dA(gpd.read_file(GI3), gpd.read_file(GI5))

# # merge to one dataframe for all periods
# dA_12 = dA_p1.merge(dA_p2, how='outer', on='nr', suffixes=('_1', '_2'))
# dA = dA_12.merge(dA_p3, how='outer', on='nr', suffixes=(False))
# dA.rename(columns={'dA': 'dA_3'}, inplace=True)



