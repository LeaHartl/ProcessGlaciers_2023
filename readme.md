# Unfinished WIP
## DEM processing:   
# Step 1, basic processing:  
AlignDEMS.py: Steps to resample DEMs to 5m resolution & merge stubai and otztal (1969 & 1997). Generates and saves new resampled/merged .tif files. uncomment stuff as needed, this is not set up to run in one go and needs manual adjustment, i.e. uncommenting, if something needs to be regenerated.
* all DEMs resampled to 5mx5m resolution  
* Stubai & Ötztal merged for 1969 and 1997  
* CRS code reset for 2006 (correct projection but no proper espg number)  
* Alignment: all dems aligned to the same extent and boundary box before dif raster generation
* all of the above in: **AlignDEMS.py**    

# Step 2: coregister all the resampled, aligned DEMs to the 2017 DEM:  
* NuthKaab coregistration to 2017 DEM (--> 1969, 1997, 2006 coregistered to 2017 - all after alignment and resmpling)
	in: coregistration_plot_nuth_kaab.py / output: coregistered DEMs for 1969, 1997, 2006 in folder xdem1
	this is very similar to examples in the xdem tutorials. ran these last with xdem==0.0.9, geoutils== 0.0.12 --> not adjusted for updated xdem and GU!
 
# Step 3: generate dif rasters with the aligned, coregistered DEMS:   
* dif raster generation on aligned, coregistered DEMs
	in AlignDEMS.py 

# Step 4: Error estimation:
* run "plot_infer_heterosc_tirol.py" to get estimate of elevation error for different slop/curvature combinations

    output: plot of elevation error and table with values for different slopes and curvatures
	
    this is very similar to examples in the xdem tutorials. Input files are the coregsitered DEMs and the glacier outlines (GI1, GI2, GI3). Slow! ran these last with xdem==0.0.9, geoutils== 0.0.12 --> not adjusted for updated xdem and GU!

* run "plot_standardization_tirol.py" to get dz uncertainty estimates for mean dz of all glaciers 
	
    output is a table where each glaciers is a row
    this is very similar to examples in the xdem tutorials. Input files are the coregsitered DEMs and the GI1 glacier outlines. Slow!! Not checked for xdem version dependency, ran these last with xdem==0.0.9, geoutils== 0.0.12 --> not adjusted for updated xdem and GU! 
    
* ErrorPlots.py : makes plot of uncertainties vs slope / boxplot (reads csv files generated in plot_standardization_tirol.py) 

* SlopeAspect.py compute and save slope and aspect rasters for 2017/18. 

------
note: updated xdem to 0.0.10 and GU to 0.0.12. GU install breaks for more recent versions (0.0.15), have not figured out why

## Shapefile processing:     
start with pangea files.
merge Ötztal and Stubai:
* MergeShapes.py : make merged shapefile of stubai and ötztal outlines. files with suffix '_ 3' have consistent HEF numbers. files with suffix _ withYEAR have extra column with survey year - for Sulzenauferner 2017 is used in this case. **ID of HEF total is 2125**. All other HEF IDs are removed in this step (0 for Toteis and 2125000 for Toteis inclusive or exclusive variations).
---   
Sulzenauferner_clip.py:   
* looks at 2017/18 difference where the dateline crosses Sulzenauferner
* makes ROI overview plot. 
* prints area ratios for different years.   
* Prints glaciers that are gone in GI5 with area, nr, name

---
---   
Hochjochferner_clip.py:   
* deals with the issue of missing DEM in Italian section of HJF (boundary was mapped with orthophoto but no DEM available in Italy)
* HJF boundary clipped in Qgis with line along a ridge in the terrain (imo this is the most logical way to do this, there is no ideal solution)  
* using new, clipped boundaries, volume and area change is computed and exported to table (this is needed for the OGGM runs)
* also outputs a plot showing the different boundaries.  
---   

Misc_GeoProcessing.py [stuff that didn't fit anywhere else]
* Make raster of flights years (2017 and 2018 per pixel), uses that to make dif dem in m/a for 2016-2017/18 period.
* clip dif_ma with gi3 boundary (didn't know where to put this, should move to some better place.)
---

---
---
---
## for further processing:   

**IMPORTANT: meta.py : dictionary with file paths and names !!!**  

set file paths in meta.py and import this as needed (--> avoid setting paths in each script...)   

---
AreaVolChange.py:   
* Generates area and volume change for individual glaciers and all time steps, writes to csv. USED IN OGGM RUNS!  
* gets list of id numbers of "gone glaciers" 

---




VolData_absolute.py : 
* deal with ice thickness raster and prduce various derivatives (ice thickness raster for 2017 based on dif dem 2006 -2017 and absolute ice thickness 2006, clipped rasters, ...)
* make plot "years till melt"
* generate ice thickness data per year per pixel until 2100, write that to .nc file for further processing
* generate geotif rasters of ice thickness data for individual years, write to disk

---

dV_perGlacier.py :   
* area and vol change per size class and for indiv. glaciers  
* absolute volume for 2006 per glacier (clip ice thickness raster with GI3 shapefiles)  
* make volume change table and write to csv (VolumeTable_v3.csv). This also makes a dV and dA table with all time steps, check which format is needed? (see AreaVolChange.py for alternative version...)

---

setup_bins.py :  
* deal with generating dz per elevation bins. this makes csv files. can be adjusted to do it for ice volume rasters or dif dems. 

---

makePlots.py :  
* read shapefiles and csv table of A and V changes per glacier, print number of glaciers with X amount of volume loss, etc.  
* makes map figure coloring glaciers by percentage loss.  
* makes compound figure change rates (map, histogram, elevation change)  
* makes compound figure relative change (aspect, slope, cumulative vol/area)  
* makes hypsometry plot and fig with scatter subplots for constant change rate scenario.  



Area_compareDatasets.py:   
* load, clip and further process glacier outlines in the ROI to compare the different available data sets
* compute area and area change uncertainty, output to table (DISCUSS! sum of uncertainties or root sum of squares...)

Vol_compareDatasets.py:   
* load, clip and further process volume rasters to compare the different available data sets

Vol_changebysizeclass.py:  
* get volume change by size class, for total vol change and change rate (---> m/a dif raster). Also computes vol errors by size class

Vol_dherrorbysizeclass.py:    
* computes variogram and terrain dependent elevation change errors by size classes, outputs tables


---
Folders: 

OG/ DEMs as provided by land tirol (2006 and 17/18)  

proc_dems/ merged (1969, 1997) and resampled files  
* 'dem_1969_merged_5.tif': Stubai & Ötztal 1969 files merged, 5x5m resolution  
* 'dem_1997_merged_5.tif': Stubai & Ötztal 1997 files merged, 5x5m resolution  
* 1969 and 1997 are aligned with each other (same extent, same grid)  
* CRS : crs code was not set properly in 2006 OG file - correct projection but code has to be reset  

xdem1/ output rasters of xdem calculations










## not needed:
Variability Plots: 
Make plots of hypsometry, variability of dz, etc. plots mostly based on csv files generated in setup_bins.py. 
