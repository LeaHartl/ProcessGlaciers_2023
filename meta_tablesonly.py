import numpy as np
import pandas as pd


# these are the warped, coregesiterd  versions:
# fn1969_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1969_coreg.tif'
# fn1997_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1997_coreg.tif'
# fn2006_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/2006_coreg.tif'
# fn2017_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'

dh_1969 = 'difdems/dif_5m_19691997.tif'
# dh_1969_ma = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19691997_ma.tif'

dh_1997 = 'difdems/dif_5m_19972006.tif'
# dh_1997_ma = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_19972006_ma.tif'

dh_2006 = 'difdems/dif_5m_20062017.tif'
# dh_2006_ma = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/dif_5m_20062017_ma.tif'

GI1 = 'shapes/mergedGI1_3.shp'
GI2 = 'shapes/mergedGI2_3.shp'
GI3 = 'shapes/mergedGI3_3.shp'
GI5 = 'shapes/mergedGI5_3.shp'
GI3_y = 'shapes/mergedGI3_withYEAR_3.shp'
GI5_y = 'shapes/mergedGI5_withYEAR_3.shp'

#slope201718 = 'misc/xdem1/slope201718.tif'
#aspect201718 = 'misc/xdem1/saspect201718.tif'

meta = {
    'GI1': {
    'shp' : GI1,
    },
    'GI2': {
    'shp' : GI2,
    # 'f_dem': [fn1969_5al, fn1997_5al],
    'f_dif': dh_1969,
    # 'f_dif_ma': dh_1969_ma,
    'yrs': [1969, 1997]
    },
    'GI3': {
    'shp' : GI3,
    # 'f_dem': [fn1997_5al, fn2006_5al],
    'f_dif': dh_1997,
    # 'f_dif_ma': dh_1997_ma,
    'yrs': [1997, 2006],
    },
    'GI5': {
    'shp' : GI5,
    # 'f_dem': [fn2006_5al, fn2017_5al],
    'f_dif': dh_2006,
    # 'f_dif_ma': dh_2006_ma,
    'yrs': [2006, 2017],
    # 'f_slope': slope201718,
    # 'f_aspect': aspect201718,
    }
    }


