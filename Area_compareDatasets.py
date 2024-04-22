import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import geoutils as gu # note: version 0.0.12
import xdem # note: version 0.0.10
import meta as mt

# import dictionary with paths to files:
meta = mt.meta


# subroutine to get area, write to dataframe
def get_A(gp1, gp2, gp3, gp5, gpLIA, rgi, gi4):
    # generate dataframe with dA for each glacier
    gp1['area'] = gp1['geometry'].area
    gp2['area'] = gp2['geometry'].area
    gp3['area'] = gp3['geometry'].area
    gp5['area'] = gp5['geometry'].area 

    gpLIA['area'] = gpLIA['geometry'].area

    rgi['area'] = rgi['geometry'].area

    gi4['area'] = gi4['geometry'].area


    dfNew = pd.DataFrame(columns=['area', 'howmany'], index=['LIA', 'GI1', 'GI2', 'GI3', '201718', 'rgi06', 'gi4'])
    dfNew.loc['LIA', 'area'] = gpLIA['area'].sum()
    dfNew.loc['LIA', 'howmany'] = gpLIA['area'].count()

    dfNew.loc['GI1', 'area'] = gp1['area'].sum()
    dfNew.loc['GI1', 'howmany'] = gp1['area'].count()

    dfNew.loc['GI2', 'area'] = gp2['area'].sum()
    dfNew.loc['GI2', 'howmany'] = gp2['area'].count()

    dfNew.loc['GI3', 'area'] = gp3['area'].sum()
    dfNew.loc['GI3', 'howmany'] = gp3['area'].count()

    dfNew.loc['201718', 'area'] = gp5['area'].sum()
    dfNew.loc['201718', 'howmany'] = gp5['area'].count()
    
    dfNew.loc['rgi06', 'area'] = rgi['area'].sum()
    dfNew.loc['rgi06', 'howmany'] = rgi['area'].count()

    dfNew.loc['gi4', 'area'] = gi4['area'].sum()
    dfNew.loc['gi4', 'howmany'] = gi4['area'].count()

    return (dfNew)


#------Area part ------
# # get A for the three time steps
# These are the paths to the shapefiles:
GI1 = meta['GI1']['shp']
GI2 = meta['GI2']['shp']
GI3 = meta['GI3']['shp']
GI5 = meta['GI5']['shp']


# load RGI 6:
rgi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/11_rgi60_CentralEurope/11_rgi60_CentralEurope.shp')
rgi.to_crs(gpd.read_file(GI1).crs, inplace=True)
# load Patrick's list of RGI IDs:
ids = pd.read_csv('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/RGI_PIZ_OTZ_STB_2.csv')
print(ids)
print(rgi)
rgi_in_roi_list = rgi.loc[rgi['RGIId'].isin(ids['rgi_id'].values)]
print(rgi_in_roi_list)


# load GI4:
gi4 = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/GI_4_2015/GI_4_2015.shp')
gi4.to_crs(gpd.read_file(GI1).crs, inplace=True)


# load cook ROI
cookRoi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/cook_area/cook_area.shp')
cookRoi.to_crs(gpd.read_file(GI1).crs, inplace=True)

# clip rgi and gi4 with cook outline
# rgi_in_roi = gpd.clip(rgi, cookRoi)
rgi_in_roi = rgi_in_roi_list

gi4_in_roi = gpd.clip(gi4, cookRoi)

# print(rgi_in_roi, rgi_in_roi_list)

missing_rgi = rgi_in_roi.loc[~rgi_in_roi['RGIId'].isin(rgi_in_roi_list['RGIId'].values)]
print(missing_rgi)

# missing_rgi.to_file('data/missing_rgi.shp')
# rgi_in_roi.to_file('data/rgi_in_roi_clippedwcook.shp')
# rgi_in_roi_list.to_file('data/rgi_in_roi_list.shp')
print(rgi_in_roi.shape, rgi_in_roi_list.shape, missing_rgi.shape)

# load GI LIA:
lia = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedLIA_3.shp')

# load shapes as geodataframes and pass to the "get_A" function, output is a summary data frame:
# dfNew = get_A(gpd.read_file(GI1), gpd.read_file(GI2), gpd.read_file(GI3), gpd.read_file(GI5), lia, rgi_in_roi, gi4_in_roi)
dfNew = get_A(gpd.read_file(GI1), gpd.read_file(GI2), gpd.read_file(GI3), gpd.read_file(GI5), lia, rgi_in_roi_list, gi4_in_roi)
dfNew['ar_km2'] =dfNew['area'] / 1e6

dfNew.to_csv('tables/compare_area.csv')
print(dfNew)

#--- Area uncertainties by size class for the regional inventories:
def SizeClassesUNC(df):
    df['area'] = df.geometry.area
    All = df.copy()
    G_toosmall = df.loc[df['area'] < 0.01e6]
    G_vsmall = df.loc[df['area'] < 0.5e6]
    G_small = df.loc[(df['area'] >= 0.5e6) & (df['area'] < 1e6)]
    G_mid = df.loc[(df['area'] >= 1e6) & (df['area'] < 5e6)]
    G_big = df.loc[(df['area'] >= 5e6) & (df['area'] < 10e6)]
    G_vbig = df.loc[(df['area'] >= 10e6)]

    df = pd.DataFrame(index=['all', 'vsmall', 'small', 'mid', 'big', 'vbig'], columns=['area', 'Uncertainty'])

    df['area'] = [All['area'].sum(), G_vsmall['area'].sum(), G_small['area'].sum(), G_mid['area'].sum(), G_big['area'].sum(), G_vbig['area'].sum()]
    df['Uncertainty'] = df['area']*0.015
    #ädf.loc['toosmall', 'Uncertainty'] = df.loc['toosmall', 'area']*0.05
    df.loc['vsmall', 'Uncertainty'] = df.loc['vsmall', 'area']*0.05
    df.loc['small', 'Uncertainty'] = df.loc['small', 'area']*0.05
    df.loc['all', 'Uncertainty'] = np.nan
    df.loc['all', 'Uncertainty'] = df['Uncertainty'].sum()


    print(df)
    return(df)
    # df['Nr'] = [merged.shape[0], G_toosmall.shape[0], G_vsmall.shape[0], G_small.shape[0], G_mid.shape[0], G_big.shape[0], G_vbig.shape[0]]
    # df['prcNr'] = 100 *df['Nr'] / merged.shape[0]
    # if colname2 ==colname:
    #     df['abs_val'] = [merged[colname2].sum(), G_toosmall[colname2].sum(), G_vsmall[colname2].sum(), G_small[colname2].sum(), G_mid[colname2].sum(), G_big[colname2].sum(), G_vbig[colname2].sum()]
    #     df['percentage'] = 100 *df['abs_val'] / merged[colname2].sum()
    #     return(df)
    
    # else:
    #     df['abs_val'] = [merged[colname2].sum(), G_toosmall[colname2].sum(), G_vsmall[colname2].sum(), G_small[colname2].sum(), G_mid[colname2].sum(), G_big[colname2].sum(), G_vbig[colname2].sum()]
    
    #     # df['abs_val'] = [merged[colname2].mean(), G_toosmall[colname2].mean(), G_vsmall[colname2].mean(), G_small[colname2].mean(), G_mid[colname2].mean(), G_big[colname2].mean(), G_vbig[colname2].mean()]
    #     return(df[['Nr', 'abs_val']])

# SizeClassesUNC(lia)
# SizeClassesUNC(gpd.read_file(GI1))
df1997 = SizeClassesUNC(gpd.read_file(GI2))
df2006 = SizeClassesUNC(gpd.read_file(GI3))
df2017 = SizeClassesUNC(gpd.read_file(GI5))

df1997.rename(columns={"area": "area1997", "Uncertainty": "Unc1997"}, inplace=True)
df2006.rename(columns={"area": "area2006", "Uncertainty": "Unc2006"}, inplace=True)
df2017.rename(columns={"area": "area2017", "Uncertainty": "Unc2017"}, inplace=True)

changeDF = pd.concat([df1997, df2006, df2017], axis=1)
changeDF['change20062017']=changeDF['area2017']-changeDF['area2006']
changeDF['changeUnc20062017']= np.sqrt(changeDF['Unc2006']**2 + changeDF['Unc2017']**2)
changeDF['Unc2006km2'] = changeDF['Unc2006'] /1e6
changeDF['Unc2017km2'] = changeDF['Unc2017'] /1e6
changeDF['change_KM2'] = changeDF['change20062017'] /1e6
changeDF['changeUnc_KM2'] = changeDF['changeUnc20062017'] /1e6

changeDF[['Unc2006km2', 'Unc2017km2', 'change_KM2', 'changeUnc_KM2']] = changeDF[['Unc2006km2', 'Unc2017km2', 'change_KM2', 'changeUnc_KM2']].round(decimals=2)

print(changeDF)
changeDF.to_csv('tables/Uncertainties_Area.csv')

