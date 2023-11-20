# import procDEMS as proc
# import EAZ_setup as st
import rasterio
from rasterio.plot import show
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import procDEMS_new as proc

import geoutils as gu
import xdem

# old data - these have the same extent and crs for Ötztal and Stubai
demOtz_1969fn = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/OG/Oetztal_1969_31254.tif'
demOtz_1997fn = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/OG/Oetztal_1997_31254.tif'
demStb_1969fn = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/OG/Stubai_1969_31254.tif'
demStb_1997fn = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/OG/Stubai_1997_31254.tif'

# resample DEMs to exactly 5x5 m resolution
# proc.warpDEMS_resample([demOtz_1969fn], 5.0, 'Oetztal_1969_31254')
# proc.warpDEMS_resample([demOtz_1997fn], 5.0, 'Oetztal_1997_31254')

# proc.warpDEMS_resample([demStb_1969fn], 5.0, 'Stubai_1969_31254')
# proc.warpDEMS_resample([demStb_1997fn], 5.0, 'Stubai_1997_31254')

#tirol new
proc.warpDEMS_resample(['/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/tirol_v2/tir1718_clip_extratiles.tif'], 5.0, 'merged1718_extratiles.tif')
stop


# resmpled and aligned files (output of above functions):
demOtz_1969fn_5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/resampled_2_5m_Oetztal_1969_31254.tif'
demOtz_1997fn_5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/resampled_2_5m_Oetztal_1997_31254.tif'
demStb_1969fn_5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/resampled_2_5m_Stubai_1969_31254.tif'
demStb_1997fn_5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/resampled_2_5m_Stubai_1997_31254.tif'

# merge stubai and otztal DEMs: 
# proc.makeMerged(demOtz_1969fn_5, demStb_1969fn_5, 'dem_1969_merged_5')
# proc.makeMerged(demOtz_1997fn_5, demStb_1997fn_5, 'dem_1997_merged_5')

# merged DEMs, filenames (output of above functions): 
fn1969_5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/dem_1969_merged_5.tif'
fn1997_5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/dem_1997_merged_5.tif'


## NEW FILES:

# large files, 1m resolution as provided by Land Tirol
fn2006 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/OG/ges2006_clip.tif'
fn201718 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/OG/tir1718_clip.tif'

# resample to 5 m
# proc.warpDEMS_resample([fn2006], 5.0, 'ges2006')
# proc.warpDEMS_resample([fn201718], 5.0, 'tir1718')

# set proper crs code (no need to reproject but code is not set in the same way in the OG file)
# fn2006_5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/resampled_2_5m_ges2006.tif'
# proc.setCRS([fn2006_5], '_ges2006')


# resampled files, with proper CRS for 2006 (output of above functions):
fn2006_5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/resampled_2_5m_CRS_ges2006.tif'
fn2017_5 = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/resampled_2_5m_tir1718.tif'

# align all dems to same extent and shape
# proc.warpDEMS_v2([fn1969_5, fn1997_5, fn2006_5, fn2017_5])

# alternative function to align, doesn't do all at once and would have to be adjusted for consistent output for all years
def align_xdem(dem_in_fn, dem_ref, fout):
    dem_in = xdem.DEM(dem_in_fn)
    dem_repr = dem_in.reproject(xdem.DEM(dem_ref))
    dem_repr.save(fout+"_xdem.tif")

# align_xdem(fn1969_5, fn1997_5, 'xdem1/aligned_dem_1969_merged_5')
# align_xdem(fn2006_5, fn1997_5, 'xdem1/aligned_dem_2006_merged_5')
# align_xdem(fn2017_5, fn1997_5, 'xdem1/aligned_dem_2017_merged_5')

# these are the warped versions (output of warpDEM_v2):
# fn2006_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_CRS_ges2006.tif'
# fn2017_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'
# fn1969_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_dem_1969_merged_5.tif'
# fn1997_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_dem_1997_merged_5.tif'

# these are xdem reprojected versions (output of align_xdem):
# fn2006_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/aligned_dem_2006_merged_5_xdem.tif'
# fn2017_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/aligned_dem_2017_merged_5_xdem.tif'
# fn1969_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/aligned_dem_1969_merged_5_xdem.tif'
# fn1997_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_dem_1997_merged_5.tif'

# these are the warped, coregesiterd  versions (output of coregisration_plot_nuth_kaab.py - warped files as input):

fn1969_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1969_coreg.tif'
fn1997_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/1997_coreg.tif'
fn2006_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/xdem1/2006_coreg.tif'
fn2017_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'

def xdem_dif(infiles, outfile):

    y1 = xdem.DEM(infiles[0])
    y2 = xdem.DEM(infiles[1])


    ddem = xdem.DEM(y2 - y1)
    # ddem = xdem.DEM(y4 - y3)
    ddem.show(cmap="coolwarm_r", vmin=-20, vmax=20)#, cb_title="Elevation change (m)")

    ddem.save(outfile+".tif")
    plt.show()


xdem_dif([fn1969_5al, fn1997_5al], 'xdem1/dif_5m_19691997')  
# xdem_dif([fn1997_5al, fn2006_5al], 'xdem1/dif_5m_19972006')  
# xdem_dif([fn2006_5al, fn2017_5al], 'xdem1/dif_5m_20062017')  

