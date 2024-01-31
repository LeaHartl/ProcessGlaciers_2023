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

# load GI4:
gi4 = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/GI_4_2015/GI_4_2015.shp')
gi4.to_crs(gpd.read_file(GI1).crs, inplace=True)


# load cook ROI
cookRoi = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/data/cook_area/cook_area.shp')
cookRoi.to_crs(gpd.read_file(GI1).crs, inplace=True)

# clip rgi and gi4 with cook outline
rgi_in_roi = gpd.clip(rgi, cookRoi)
gi4_in_roi = gpd.clip(gi4, cookRoi)


# load GI LIA:
lia = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedLIA_3.shp')

# load shapes as geodataframes and pass to get_A function:
dfNew = get_A(gpd.read_file(GI1), gpd.read_file(GI2), gpd.read_file(GI3), gpd.read_file(GI5), lia, rgi_in_roi, gi4_in_roi)

dfNew['ar_km2'] =dfNew['area'] / 1e6

dfNew.to_csv('tables/compare_area.csv')
print(dfNew)

