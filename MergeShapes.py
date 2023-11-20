
# import rasterio
# from rasterio.plot import show
import numpy as np
import pandas as pd
import geopandas as gpd

# supress setting with copy warning, use carefully!
pd.options.mode.chained_assignment = None  # default='warn'


# set filepaths to shapefiles to be merged:
fldr = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/shapefiles'

Otz = {
    'GI1': {
            'shp': fldr + '/Oetztal_GI1.shp',
    },
    'GI2': {
            'shp': fldr + '/Oetztal_GI2.shp',
    },
    'GI3': {
            'shp': fldr + '/Oetztal_GI3.shp',
    },
    'GI5': {# Kay's latest version: this has HEF total (=including Toteis) as 2125 and HEF excluding Toteis as 2125000. No extra Toteis.
            'shp': '/Users/leahartl/Desktop/ELA_EAZ/v2/2023/GI5_Pangaea/Oetztaler_Alpen_GI5_pangaea.shp',
    }
    }

Stb = {
    'GI1': {
            'shp': fldr + '/Stubaier_Alpen_GI1.shp',
    },
    'GI2': {
            'shp': fldr + '/Stubaier_Alpen_GI2.shp',
    },
    'GI3': {
            'shp': fldr + '/Stubaier_Alpen_GI3.shp',
    },
    'GI5': {# Kay's latest version: small adjustments to Sulzenau and Alpeiner Ferner
            'shp': '/Users/leahartl/Desktop/ELA_EAZ/v2/2023/GI5_Pangaea/Stubai_GI5_pangaea.shp',

    }
    }


# combine Stubai and Otztal shapefiles & account for HEF ID number issue:
def combine(stbGPD, otzGPD, GI):
    stbGPD = gpd.read_file(Stb[GI]['shp'])
    otzGPD = gpd.read_file(Otz[GI]['shp'])

    if otzGPD.crs != 'epsg:31254':
        otzGPD = otzGPD.to_crs('epsg:31254')
    if stbGPD.crs != 'epsg:31254':
        stbGPD = stbGPD.to_crs('epsg:31254')

    # reality check: print crs
    print(stbGPD.crs)
    print(otzGPD.crs)

    # if GI == 'GI1': # should all be consistent now!
    otzGPD = otzGPD[['nr', 'Gletschern', 'geometry']]
    # else:
    #     otzGPD = otzGPD[['ID', 'Gletschern', 'geometry']]
    #     otzGPD.rename({'ID': 'nr'}, axis=1, inplace=True)

    stbGPD = stbGPD[['nr', 'Gletschern', 'geometry']]

    merged = pd.concat([stbGPD, otzGPD])
    merged['area'] = merged.area

    # check 2027/2072 problem: Langtaler F. should be 2072, Wannen Ferner 2027 - Set to 2072 for Langtaler F.
    merged['nr'].loc[merged.Gletschern == 'Langtaler Ferner'] = 2072

    # deal with Hintereisferner. Set number for HEF ohne Toteis to 2125000 in all GI for constistency.
    # nr HEF: 2125, nr Toteis: 0. Keep only HEF total (incl. Toteis) for further processing.
    merged['nr'].loc[merged.Gletschern == 'Hintereis Ferner'] = 2125
    merged['nr'].loc[merged.Gletschern == 'Hintereisferner'] = 2125
    merged['nr'].loc[merged.Gletschern == 'Toteis'] = 0
    # remove Toteis if it is present:
    merged = merged.loc[merged.nr != 0]
    # remove HEF without Toteis if present:
    merged = merged.loc[merged.nr != 2125000]

    # print to check:
    print(GI)
    print(merged.loc[merged.nr == 2125000])
    print(merged.loc[merged.nr == 0])
    print(merged.loc[merged.nr == 2125])

    merged.to_file('mergedOutlines/merged'+GI+'_3.shp')
    return(merged)


mG1 = combine(Stb, Otz, 'GI1')
# print(mG1.shape)
mG2 = combine(Stb, Otz, 'GI2')
mG3 = combine(Stb, Otz, 'GI3')
mG5 = combine(Stb, Otz, 'GI5')




# add column with survey years to geodataframe for GI3 and GI5
flight_years = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/data/als_flugjahr_v22/als_flugjahr_v22.shp')
flight_lines = gpd.read_file('/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/flightlines/als1718_fldat_epsg31254.shp')

join1 = mG5.sjoin(flight_years[['YEAR', 'geometry']], how="left")
#Sulzenau Ferner was partly surveyed in 2017 and partly in 2018. Remove second (2018) entry for Sulzenau Ferner, fix later:
join1.drop_duplicates(subset=['nr'], inplace=True)
# print for reality check:
print(join1.loc[join1.Gletschern == 'Sulzenau Ferner'])
join1.to_file('mergedOutlines/mergedGI5_withYEAR_3.shp')

# also merge for GI3 to have the second survey year in the GI3 shapefile for later computations:
join2 = mG3.sjoin(flight_years[['YEAR', 'geometry']], how="left")
#Sulzenau Ferner was partly surveyed in 2017 and partly in 2018. Remove second (2018) entry for Sulzenau Ferner, fix later:
join2.drop_duplicates(subset=['nr'], inplace=True)
# print for reality check:
print(join2.loc[join2.Gletschern == 'Sulzenau Ferner'])
join2.to_file('mergedOutlines/mergedGI3_withYEAR_3.shp')


# # remove HEF and save shapefiles w/out HEF:
# folder = 'mergedOutlines/'
# GI1 = 'mergedGI1_3'
# GI2 = 'mergedGI2_3'
# GI3 = 'mergedGI3_3'
# GI5 = 'mergedGI5_3'
# GI3_y = 'mergedGI3_withYEAR_3'
# GI5_y = 'mergedGI5_withYEAR_3'


# for f in [GI1, GI2, GI3, GI5, GI3_y, GI5_y]:
#     mG = gpd.read_file(folder+f+'.shp')
#     # account for different numbering of HEF in GI5 - keep only HEF total (2125), remove toteis (0) and HEF ohne Toteis (2125000)
#     mG = mG.loc[mG.nr != 2125000]
#     mG = mG.loc[mG.nr != 0]
#     mG.to_file(folder+'HEFtotal_'+f+'.shp')



# not needed, used this to check it matches the function that makes the tables...
# def countGlaciers(mG):
#     # print(mG)
#     mG = mG.reset_index()
#     # print(mG)
#     G_vsmall = mG.loc[mG.area < 0.5e6]
#     G_small = mG.loc[(mG.area >= 0.5e6) & (mG.area < 1e6)]
#     G_mid = mG.loc[(mG.area >= 1e6) & (mG.area < 5e6)]
#     G_big = mG.loc[(mG.area >= 5e6) & (mG.area < 10e6)]
#     G_vbig = mG.loc[(mG.area >= 10e6)]

#     # ls = [mG.shape[0], G_vsmall.shape[0], G_small.shape[0], G_mid.shape[0], G_big.shape[0], G_vbig.shape[0]]
#     # lsPr = [mG.shape[0]/mG.shape[0], G_vsmall.shape[0]/mG.shape[0], G_small.shape[0]/mG.shape[0], G_mid.shape[0]/mG.shape[0], G_big.shape[0]/mG.shape[0], G_vbig.shape[0]/mG.shape[0]]
#     # print(lsPr)
#     df = pd.DataFrame(index=['totNr', 'vsmall', 'small', 'mid', 'big', 'vbig'], columns=['Nr', 'prcNr', 'area', 'areaPrc'])
#     df['Nr'] = [mG.shape[0], G_vsmall.shape[0], G_small.shape[0], G_mid.shape[0], G_big.shape[0], G_vbig.shape[0]]
#     df['prcNr'] = 100 *df['Nr'] / mG.shape[0]
#     df['area'] = [mG.area.sum(), G_vsmall.area.sum(), G_small.area.sum(), G_mid.area.sum(), G_big.area.sum(), G_vbig.area.sum()]
#     df['areaPrc'] = 100 *df['area'] / mG.area.sum()

#     print(df)
#     return(df)
#     #print(df.sum())

# countGlaciers(mG1)  
# countGlaciers(mG2)
# cG3 = countGlaciers(mG3)
# cG5 = countGlaciers(mG5)

# dif3_5 = cG5 - cG3
# print('dif=', dif3_5)

# pd.options.display.float_format = '{:.2f}'.format
# tab = cG5
# tab['area'] = tab['area'] / 1e6
# tab['NrChangeG3G5'] = dif3_5['Nr'].astype(int)
# tab['ArChangeG3G5'] = dif3_5['area'] / 1e6
# tab['ArChangeG3G5_perc'] = 100*(dif3_5['area'] / 1e6) / (cG3['area'] / 1e6)

# tab[['Nr', 'NrChangeG3G5']] = tab[['Nr', 'NrChangeG3G5']].round(0)

# print(tab.T)


