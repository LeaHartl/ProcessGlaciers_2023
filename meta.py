import numpy as np
import pandas as pd


# these are the warped, coregesiterd  versions:
fn1969_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1969_coreg.tif'
fn1997_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1997_coreg.tif'
fn2006_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/2006_coreg.tif'
fn2017_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'

dh_1969 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19691997.tif'
dh_1969_ma = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19691997_ma.tif'

dh_1997 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19972006.tif'
dh_1997_ma = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19972006_ma.tif'

dh_2006 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_20062017.tif'
dh_2006_ma = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_20062017_ma.tif'

# GI1 = 'mergedGI1_2.shp'
# GI2 = 'mergedGI2_2.shp'
# GI3 = 'mergedGI3_2.shp'
# GI5 = 'mergedGI5_2.shp'
# GI3_y = 'mergedGI3_withYEAR.shp'
# GI5_y = 'mergedGI5_withYEAR.shp'

GI1 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI1_3.shp'
GI2 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI2_3.shp'
GI3 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI3_3.shp'
GI5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI5_3.shp'
GI3_y = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI3_withYEAR_3.shp'
GI5_y = '/Users/leahartl/Desktop/ELA_EAZ/v2/ProcessGlaciers_2023/mergedOutlines/mergedGI5_withYEAR_3.shp'

slope201718 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/slope201718.tif'
aspect201718 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/saspect201718.tif'

meta = {
    'GI1': {
    'shp' : GI1,
    },
    'GI2': {
    'shp' : GI2,
    'f_dem': [fn1969_5al, fn1997_5al],
    'f_dif': dh_1969,
    'f_dif_ma': dh_1969_ma,
    'yrs': [1969, 1997]
    },
    'GI3': {
    'shp' : GI3,
    'f_dem': [fn1997_5al, fn2006_5al],
    'f_dif': dh_1997,
    'f_dif_ma': dh_1997_ma,
    'yrs': [1997, 2006],
    },
    'GI5': {
    'shp' : GI5,
    'f_dem': [fn2006_5al, fn2017_5al],
    'f_dif': dh_2006,
    'f_dif_ma': dh_2006_ma,
    'yrs': [2006, 2017],
    'f_slope': slope201718,
    'f_aspect': aspect201718,
    }
    }



# ice data: 
fldr = 'data/'
ice2006 = pd.read_csv(fldr+'flHoAltSum_50_ice_data2006.csv', index_col=0)
ice2017 = pd.read_csv(fldr+'flHoAltSum_50_ice_data2017.csv', index_col=0)
ice2030 = pd.read_csv(fldr+'flHoAltSum_50_ice_proj2030.csv', index_col=0)
ice2050 = pd.read_csv(fldr+'flHoAltSum_50_ice_proj2050.csv', index_col=0)
ice2075 = pd.read_csv(fldr+'flHoAltSum_50_ice_proj2075.csv', index_col=0)
ice2100 = pd.read_csv(fldr+'flHoAltSum_50_ice_proj2100.csv', index_col=0)

ice2006_mn = pd.read_csv(fldr+'flHoAlt_50_ice_data2006.csv', index_col=0)
ice2017_mn = pd.read_csv(fldr+'flHoAlt_50_ice_data2017.csv', index_col=0)
ice2030_mn = pd.read_csv(fldr+'flHoAlt_50_ice_proj2030.csv', index_col=0)
ice2050_mn = pd.read_csv(fldr+'flHoAlt_50_ice_proj2050.csv', index_col=0)
ice2075_mn = pd.read_csv(fldr+'flHoAlt_50_ice_proj2075.csv', index_col=0)
ice2100_mn = pd.read_csv(fldr+'flHoAlt_50_ice_proj2100.csv', index_col=0)

ice2006_md = pd.read_csv(fldr+'flHoAltMed_50_ice_data2006.csv', index_col=0)
ice2017_md = pd.read_csv(fldr+'flHoAltMed_50_ice_data2017.csv', index_col=0)
ice2030_md = pd.read_csv(fldr+'flHoAltMed_50_ice_proj2030.csv', index_col=0)
ice2050_md = pd.read_csv(fldr+'flHoAltMed_50_ice_proj2050.csv', index_col=0)
ice2075_md = pd.read_csv(fldr+'flHoAltMed_50_ice_proj2075.csv', index_col=0)
ice2100_md = pd.read_csv(fldr+'flHoAltMed_50_ice_proj2100.csv', index_col=0)

ice2006_ar = pd.read_csv(fldr+'flHoAltAr_50_ice_data2006.csv', index_col=0)
ice2017_ar = pd.read_csv(fldr+'flHoAltAr_50_ice_data2017.csv', index_col=0)
ice2030_ar = pd.read_csv(fldr+'flHoAltAr_50_ice_proj2030.csv', index_col=0)
ice2050_ar = pd.read_csv(fldr+'flHoAltAr_50_ice_proj2050.csv', index_col=0)
ice2075_ar = pd.read_csv(fldr+'flHoAltAr_50_ice_proj2075.csv', index_col=0)
ice2100_ar = pd.read_csv(fldr+'flHoAltAr_50_ice_proj2100.csv', index_col=0)

# icedata_sum = [ice2006, ice2017, ice2030, ice2050, ice2075, ice2100]
# icedata_mean = [ice2006_mn, ice2017_mn, ice2030_mn, ice2050_mn, ice2075_mn, ice2100_mn]
# icedata_median = [ice2006_md, ice2017_md, ice2030_md, ice2050_md, ice2075_md, ice2100_md]
# icedata_ar = [ice2006_ar, ice2017_ar, ice2030_ar, ice2050_ar, ice2075_ar, ice2100_ar]
icedata_sum = [ice2006, ice2017, ice2030, ice2050, ice2100]
icedata_mean = [ice2006_mn, ice2017_mn, ice2030_mn, ice2050_mn, ice2100_mn]
icedata_median = [ice2006_md, ice2017_md, ice2030_md, ice2050_md, ice2100_md]
icedata_ar = [ice2006_ar, ice2017_ar, ice2030_ar, ice2050_ar, ice2100_ar]


