"""Plot the comparison between a dDEM before and after Nuth and Kääb (2011) coregistration."""
import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np

import xdem


# these are the warped versions:
fn2006_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_CRS_ges2006.tif'
fn2017_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_resampled_2_5m_tir1718.tif'
fn1969_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_dem_1969_merged_5.tif'
fn1997_5al = '/Users/leahartl/Desktop/ELA_EAZ/v2/newDEMs/proc_dems/aligned_dem_1997_merged_5.tif'


def coregNK(fn_older, fn_newer, outlines, fnout, fntif):
    dem_older = xdem.DEM(fn_older)
    dem_newer = xdem.DEM(fn_newer)

# outlines_1990 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
    glacier_outlines = gu.Vector(outlines)
    inlier_mask = ~glacier_outlines.create_mask(dem_newer)


    nuth_kaab = xdem.coreg.NuthKaab()
    nuth_kaab.fit(dem_newer, dem_older, inlier_mask=inlier_mask)#, transform=dem_newer.transform)
    dem_coreg = nuth_kaab.apply(dem_older)
    # stop
    ddem_pre = dem_newer - dem_older
    ddem_post = dem_newer - dem_coreg

    nmad_pre = xdem.spatialstats.nmad(ddem_pre[inlier_mask])
    nmad_post = xdem.spatialstats.nmad(ddem_post[inlier_mask])

    vlim = 20
    plt.figure(figsize=(8, 5))
    plt.subplot2grid((1, 15), (0, 0), colspan=7)
    plt.title(f"Before coregistration. NMAD={nmad_pre:.3f} m")
    plt.imshow(ddem_pre.data.squeeze(), cmap="coolwarm_r", vmin=-vlim, vmax=vlim)
    plt.axis("off")
    plt.subplot2grid((1, 15), (0, 7), colspan=7)
    plt.title(f"After coregistration. NMAD={nmad_post:.3f} m")
    img = plt.imshow(ddem_post.data.squeeze(), cmap="coolwarm_r", vmin=-vlim, vmax=vlim)
    plt.axis("off")
    plt.subplot2grid((1, 15), (0, 14), colspan=1)
    cbar = plt.colorbar(img, fraction=0.4, ax=plt.gca())
    cbar.set_label("Elevation change (m)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig('xdem1/'+fnout+'.png')

    dem_coreg.save('xdem1/'+fntif+'.tif')

    plt.show()

coregNK(fn1969_5al, fn2017_5al, 'mergedGI1.shp', '1969_2017_coreg', '1969_coreg')
coregNK(fn1997_5al, fn2017_5al, 'mergedGI2.shp', '1997_2017_coreg', '1997_coreg')
coregNK(fn2006_5al, fn2017_5al, 'mergedGI3.shp', '2006_2017_coreg', '2006_coreg')



