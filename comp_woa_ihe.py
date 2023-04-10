#################################################
# Lucas Tartaro Pereira   			#			  
# Comparison of WOA18 and IHESP databases	#			  
#################################################

"""
Challenges:
	- Comparision of datas with two different GRIDS and coordenates systems.
	- Large set of data from iHesp.
	- Polar Map, with unusual format and boundary.
	- Otimizated calculation of estatistical measures.
"""

#################################################
#    	   Fundamental Datasets Info  		#
#             	    IHESP			#
# data_ihesp_tem.__xarray_dataarray_variable__	#
#     Surface = 500 cm = 5 m			#
#     Time range = 01/15/1950 to 12/15/2014	#
#     Bound Limits = lon [-89.95    89.95]	#
#                    lat [-35.70   -90.00]	#
#             	     				#
#             	    WOA18			#
# xr_woa18_tem.t_an.values[:]			#
#     surface: idx 1 = 5m			#
#     Time range = 01/01/1955 to 31/01/2017     # 
#     Bound Limits = lon [-180.00 180.00]	#
#                    lat [90.00   -90.00]	#
#################################################

#################################################
# 		All Libraries Used	  	#
#################################################

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import netCDF4 as nc
import cartopy as cp
import cartopy.crs as ccrs
import cmocean as cm
import cartopy.feature as cfeature
import datetime 
import matplotlib.dates as mdates
import matplotlib.ticker as plticker
import matplotlib.ticker as mticker
import matplotlib.path as mpath
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import glob
import gsw as gs
import scipy as sp
from scipy import stats
import scipy.interpolate as inter
import imageio
import matplotlib.pyplot as plt

###################################
# 	Defined Functions 	  #
###################################

###     mapa_r_panofsky_153	 ##
"""
This function provides the calculation of
Correlation of Time Series (r) from the page 153 of
Panofsky's book: "Some Applications of Statistics to
Meteorology". The equation of r for two time series
(X(t) and Y(t) is:

r = [E(X(t)Y(t)) - E(X(t))E(Y(t))] / (Sx Sy)

Where E(x) is the Expected Value Function;
Sx and Sy are the Standard Deviation of X(t) and Y(t).

mapa_r_panofsky_153 works for two matrix 3d, with the
follow dimensions: (t,x,y)
"""

def mapa_r_panofsky_153(mat_3d_a,mat_3d_b):
    med_x_y = np.mean(mat_3d_a*mat_3d_b,axis=0)
    med_x_med_y = np.mean(mat_3d_a,axis=0)*np.mean(mat_3d_b,axis=0)
    std_x = np.std(mat_3d_a,axis=0)
    std_y = np.std(mat_3d_b,axis=0)
    return((med_x_y - med_x_med_y)/(std_x*std_y))
    
####################################


###     find_idx_lat_lon	 ###
"""
This function finds the closest index
for a given lat or lon value wanted
within a vector of lat_val and lon_val.
"""

def find_idx_lat_lon(lat=-70,lon=0,lat_val=None,lon_val=None,err=0.2):
    ind_lat = np.where((lat_val>=(lat-err))&(lat_val<=(lat+err)))
    ind_lon = np.where((lon_val>=(lon-err))&(lon_val<=(lon+err)))
    print(ind_lat,ind_lon)
    lat_idx = ind_lat[0][0]
    lon_idx = ind_lon[0][0]
    return lat_idx,lon_idx

####################################


###   interpola & interpola_2d   ###
"""
This two functions are exactly the
same, except for the dimension of
lat,lon,lati and loni.

interpola function receives a vector
whilst interpola_2d receives a matrix

The both uses the inter.griddata
function from scipy.interpolate lib
to reshape and regrid datasets. This
tool is usefull to compare two datasets
with different grids, transforming them
to the same size and divisions.
"""

def interpola(lat,lon,data,lati,loni):
    long,latg=np.meshgrid(lon,lat)
    values=(long.flatten(),latg.flatten())
    longi,latgi=np.meshgrid(loni,lati)
    xi=(longi.flatten(),latgi.flatten())
    datai = inter.griddata(values,data.flatten(),xi).reshape(latgi.shape)
    return datai

def interpola_2d(lat,lon,data,lati,loni):
    values=(lon.flatten(),lat.flatten())
    xi=(loni.flatten(),lati.flatten())
    datai = inter.griddata(values,data.flatten(),xi).reshape(lati.shape)
    return datai

####################################


###     media_na_latitude &      ###
###     media_na_longitude       ###
"""
These two functions work with the same idea:
Fixing a layer and an time instant results
in a longitude-latitude surface.
Then, For each latitude (media_na_latitude 
function) we calculate the mean longitude values
and vice-vers (media_na_longitude function).
"""

def media_na_latitude(matriz,size = len(rec_woa18_lat)):
    media_latitude = np.zeros(size)
    for i in range(size):
        aux = matriz[i]
        media_latitude[i] = np.mean(aux[np.isfinite(aux)])
    return(media_latitude)

def media_na_longitude(matriz,size = len(rec_woa18_lon)):
    media_longitude = np.zeros(size)
    for i in range(size):
        aux = matriz[:,i]
        media_longitude[i] = np.mean(aux[np.isfinite(aux)])
    return(media_longitude)
    
####################################


###     make_boundary_path       ###
"""
This boundary path code was taken from 
https://stackoverflow.com/search?q=ccrs.Stereographic
to add boundary to the maps with Stereographic proj.
"""

def make_boundary_path(lons, lats):
    boundary_path = np.array([lons[-1, :], lats[-1, :]])
    boundary_path = np.append(boundary_path, np.array([lons[::-1, -1], lats[::-1, -1]]), axis=1)
    boundary_path = np.append(boundary_path, np.array([lons[1, ::-1], lats[1, ::-1]]), axis=1)
    boundary_path = np.append(boundary_path, np.array([lons[:, 1], lats[:, 1]]), axis=1)
    print(np.shape(boundary_path))
    boundary_path = mpath.Path(np.swapaxes(boundary_path, 0, 1))
    return boundary_path

####################################

###        make_south_map        ###
"""
At the final section there is a lot of
maps constructed. Almost all of them are
simillar but with a little changes that
doesn't allow to use generic functions
or loopings.

The very repated features on the maps
are condensed in this make_south_map
function.
"""

def make_south_map (lon_bound = [-90,90],lon_shape = 1800,lat_bound = [-90,-40], lat_shape = 800,name = "CESM1-CAM5-SE-HR",dpi=100):
    fig = plt.figure(figsize=(12.95,7.28),dpi=dpi)
    ax = plt.axes(projection=ccrs.Stereographic(central_longitude=np.mean(lon_bound), central_latitude=np.mean(lat_bound)))
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='papayawhip',linewidth=0.5)
    ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax.coastlines(resolution='50m')
    lat_virtual = np.linspace(lat_bound[0],lat_bound[1],lat_shape)
    lon_virtual = np.linspace(lon_bound[0],lon_bound[1],lon_shape)
    LON_VIRT,LAT_VIRT = np.meshgrid(lon_virtual,lat_virtual)
    boundary_path = make_boundary_path(LON_VIRT, LAT_VIRT)
    grad=[lon_bound[0],lon_bound[1],lat_bound[0],lat_bound[1]]
    ax.set_extent(grad, crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    xticks = [-110, -50, -40, -30, -20, -11, 0, 10, 20, 30, 40, 50]
    yticks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    ax.set_boundary(boundary_path, transform=ccrs.PlateCarree())
    ax.set_title(name, fontsize=16, y=1.02)
    return fig,ax
####################################

####################################
#   Open Select Reshape Datasets   #
####################################

#First Define the paths where data are storaged

path_ihesp_tem_rec = r'/home/lucastartaro/programas_ihesp/ihesp_media_mensal_tem.nc'

path_ihesp_sal_rec = r'/home/lucastartaro/programas_ihesp/ihesp_media_mensal_sal.nc'

path_woa18_tem = r'/home/lucastartaro/woa18_data/*nc'

path_woa18_sal= r'/home/lucastartaro/woa18_sal/*nc'

# Then open the multiple files at once with the function xr.open_mfdataset
# Because of spacial resolution, file's size is huge, so each file is a year

xr_woa18_tem = xr.open_mfdataset(glob.glob(path_woa18_tem),decode_times=False)

xr_woa18_sal = xr.open_mfdataset(glob.glob(path_woa18_sal),decode_times=False)

xr_ihesp_tem = xr.open_dataset(path_ihesp_tem_rec)

xr_ihesp_sal = xr.open_dataset(path_ihesp_sal_rec)

# IHESP: Select the variable of interest from ihesp data

val_tem_ihesp = xr_ihesp_tem.__xarray_dataarray_variable__.values

val_sal_ihesp = xr_ihesp_sal.__xarray_dataarray_variable__.values

# WOA18: Need to select the area before

# Extract lon and lat val from WOA18 (it's the same for both tem and sal)

lat_val = xr_woa18_tem.lat.values
lon_val = xr_woa18_tem.lon.values

# Extract lon and lat val from IHESP

lat_min=np.min(np.array(xr_ihesp_tem['lat']))
lon_min=np.min(np.array(xr_ihesp_tem['lon']))
lat_max=np.max(np.array(xr_ihesp_tem['lat']))
lon_max=np.max(np.array(xr_ihesp_tem['lon']))

# From IHESP lat-lon values we take the idx
# in WOA18 data using find_idx_lat_lon function

err=0.2

idx_min_lat,idx_min_lon = find_idx_lat_lon(lat=lat_min,lon=lon_min,lat_val=lat_val,lon_val=lon_val,err=err)

idx_max_lat,idx_max_lon = find_idx_lat_lon(lat=lat_max,lon=lon_max,lat_val=lat_val,lon_val=lon_val,err=err)

# Selecting WOA18 area from lat-lon idx
# Both work to sal and tem sets

rec_woa18_tem = xr_woa18_tem.t_an.values[:,1,idx_min_lat:idx_max_lat,idx_min_lon:idx_max_lon]

rec_woa18_sal = xr_woa18_sal.s_an.values[:,1,idx_min_lat:idx_max_lat,idx_min_lon:idx_max_lon]

rec_woa18_lon = xr_woa18_tem.lon.values[idx_min_lon:idx_max_lon]

rec_woa18_lat = xr_woa18_tem.lat.values[idx_min_lat:idx_max_lat]

# Making WOA18 grid from WOA18 lat-lon vecs

rec_woa18_lon_2d,rec_woa18_lat_2d = np.meshgrid(rec_woa18_lon,rec_woa18_lat)

# Now interpolate the IHESP grid to
# shape as the WOA18 grid

lon_ihesp = np.array(xr_ihesp_tem['lon'])
lat_ihesp = np.array(xr_ihesp_tem['lat'])

ihesp_tem_interpol = np.zeros_like(rec_woa18_tem)

ihesp_sal_interpol = np.zeros_like(rec_woa18_tem)

for i in range(12):
    data_aux_tem = np.copy(val_tem_ihesp[i])
    data_aux_sal = np.copy(val_sal_ihesp[i])
    ihesp_tem_interpol[i] = interpola_2d(lat_ihesp,lon_ihesp,data_aux_tem
                                 ,rec_woa18_lat_2d,rec_woa18_lon_2d)
    ihesp_sal_interpol[i] = interpola_2d(lat_ihesp,lon_ihesp,data_aux_sal
                                 ,rec_woa18_lat_2d,rec_woa18_lon_2d)

####################################
#    Statistical Data Analysis 	   #
####################################
"""
Each calculation with 3D means that
the product has dimensions (t,x,y)
"""

# Difference Maps 3D

dif_tem = ihesp_tem_interpol - rec_woa18_tem
dif_sal = ihesp_sal_interpol - rec_woa18_sal

# Mean Squared Error Maps 3D

RMSE_TEM = np.sqrt((1/12)*np.sum(dif_tem**2,axis=0))
RMSE_SAL = np.sqrt((1/12)*np.sum(dif_sal**2,axis=0))

# Difference Map Time Mean (All) (2D)

DIF_TEM = np.mean(ihesp_tem_interpol,axis=0) - np.mean(rec_woa18_tem,axis=0)

DIF_SAL = np.mean(ihesp_sal_interpol,axis=0) - np.mean(rec_woa18_sal,axis=0)

# Standard Deviation Map 3D

std_ihe_tem = np.std(ihesp_tem_interpol,axis=0)
std_ihe_sal = np.std(ihesp_sal_interpol,axis=0)

# Correlation of Time Series (r) by Panofsky 3D

r_tsm = mapa_r_panofsky_153(ihesp_tem_interpol,rec_woa18_tem)
r_ssm = mapa_r_panofsky_153(ihesp_sal_interpol,rec_woa18_sal)

### Making quarterly mean
# DJF = December, January and February
# MAM = March, April, May
# JJA = June, July, August
# SON = September, October, November

"""
To make this easier I will exchange the
January and December data in the array.
"""

mean_label = ['DJF','MAM','JJA','SON']

val_tem_ihesp_t = np.zeros_like(rec_woa18_tem)
val_tem_ihesp_t[0] = ihesp_tem_interpol[11]
val_tem_ihesp_t[1:] = ihesp_tem_interpol[:11]

val_sal_ihesp_t = np.zeros_like(rec_woa18_tem)
val_sal_ihesp_t[0] = ihesp_sal_interpol[11]
val_sal_ihesp_t[1:] = ihesp_sal_interpol[:11]

woa18_tem_t = np.zeros_like(rec_woa18_tem)
woa18_tem_t[0] = rec_woa18_tem[11]
woa18_tem_t[1:] = rec_woa18_tem[:11]

woa18_sal_t = np.zeros_like(rec_woa18_tem)
woa18_sal_t[0] = rec_woa18_sal[11]
woa18_sal_t[1:] = rec_woa18_sal[:11]

"""
Then, it's easy to make the quaterly mean
with simple slice.
"""

woa18_tem_smean = np.zeros_like(rec_woa18_tem[:4])
woa18_sal_smean = np.zeros_like(rec_woa18_tem[:4])
ihesp_tem_smean = np.zeros_like(rec_woa18_tem[:4])
ihesp_sal_smean = np.zeros_like(rec_woa18_tem[:4])
rmse_tem_smean = np.zeros_like(rec_woa18_tem[:4])
rmse_sal_smean = np.zeros_like(rec_woa18_tem[:4])

"""
Finally, we repeat the statistical equations for
these new set.
"""

dif_tem = val_tem_ihesp_t - woa18_tem_t
dif_sal = val_sal_ihesp_t - woa18_sal_t

for i in range(4):
    woa18_tem_smean[i] = np.nanmean(woa18_tem_t[3*i:3*(i+1)],axis=0)
    woa18_sal_smean[i] = np.nanmean(woa18_sal_t[3*i:3*(i+1)],axis=0)
    ihesp_tem_smean[i] = np.nanmean(val_tem_ihesp_t[3*i:3*(i+1)],axis=0)
    ihesp_sal_smean[i] = np.nanmean(val_sal_ihesp_t[3*i:3*(i+1)],axis=0)
    rmse_tem_smean[i] = np.sqrt((1/3)*np.sum(dif_tem[3*i:3*(i+1)]**2,axis=0))
    rmse_sal_smean[i] = np.sqrt((1/3)*np.sum(dif_sal[3*i:3*(i+1)]**2,axis=0))
    
dif_tem_smean = ihesp_tem_smean - woa18_tem_smean
dif_sal_smean = ihesp_sal_smean - woa18_sal_smean

####################################
####################################
####################################
"""
Here is the end of the logic code.
The following lines are simple plots
resulted from the code above.
"""
####################################
####################################
####################################







####################################
#               PLOTS 	           #
####################################

fig,ax = make_south_map(dpi=180,name = "IHESP TSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,np.mean(ihesp_tem_interpol,axis=0),levels=np.linspace(-2.5,15,40),transform=ccrs.PlateCarree(),cmap='cmo.balance',extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='�C', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-1,13,8,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'M�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(ihesp_tem_interpol[ihesp_tem_interpol!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/IHESP_MED_TOTAL_TSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "IHESP SSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,np.mean(ihesp_sal_interpol,axis=0),levels=np.linspace(33.5,35.5,40),transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='Salinidade', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(33.5,35.5,8,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'M�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(ihesp_sal_interpol[ihesp_sal_interpol!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/IHESP_MED_TOTAL_SSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "WOA18 TSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,np.mean(rec_woa18_tem,axis=0),levels=np.linspace(-2.5,15,40),transform=ccrs.PlateCarree(),cmap='cmo.balance',extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='�C', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-1,13,8,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'M�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(rec_woa18_tem))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/WOA18_MED_TOTAL_TSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "WOA18 SSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,np.mean(rec_woa18_sal,axis=0),levels=np.linspace(33.5,35.5,40),transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='Salinidade', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(33.5,35.5,8,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'M�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(rec_woa18_sal))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/WOA18_MED_TOTAL_SSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "RMSE TSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,RMSE_TEM,levels=np.linspace(0,2,40),transform=ccrs.PlateCarree(),cmap='plasma',extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='RMSE', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(0,2,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'RMSE m�dio', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(RMSE_TEM[RMSE_TEM!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/RMSE_TOTAL_TSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "RMSE SSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,RMSE_SAL,levels=np.linspace(0,0.7,40),transform=ccrs.PlateCarree(),cmap='plasma',extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='RMSE', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(0,0.7,8,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'RMSE m�dio', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(RMSE_SAL[RMSE_SAL!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/RMSE_TOTAL_SSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "(IHESP - WOA18) TSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,DIF_TEM,levels=np.linspace(-2,2,40),transform=ccrs.PlateCarree(),cmap='cmo.balance',extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='�C', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-2,2,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'Dif. m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(DIF_TEM[DIF_TEM!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/DIF_TOTAL_TSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "(IHESP - WOA18) SSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,DIF_SAL,levels=np.linspace(-0.5,0.5,40),transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='Salinidade', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-0.5,0.5,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'Dif. m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(DIF_SAL[DIF_SAL!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/DIF_TOTAL_SSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "STD IHESP SSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,std_ihe_sal,levels=np.linspace(0,0.25,40),transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='Salinidade', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(0.,0.25,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'STD m�dio', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(std_ihe_sal[std_ihe_sal!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/IHESP_STD_SSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "STD IHESP SSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,std_ihe_sal,levels=np.linspace(0,0.25,40),transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='Salinidade', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(0.,0.25,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'STD m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(std_ihe_sal[std_ihe_sal!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/IHESP_STD_SSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "CORRELA�AO TSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,r_tsm,levels=np.linspace(-1,1,40),transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='Salinidade', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-1,1,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'Corr. m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(r_tsm[r_tsm!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/CORR_TSM.png',bbox_inches='tight')

fig,ax = make_south_map(dpi=180,name = "CORRELA�AO SSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,r_ssm,levels=np.linspace(-1,1,40),transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='Salinidade', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-1,1,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'Corr. m�dio', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(r_ssm[np.isfinite(r_ssm)]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/CORR_SSM.png',bbox_inches='tight')

# PLotando bonito todas as figuras de uma vez 

for i in range(4):
    fig,ax = make_south_map(dpi=180,name = "IHESP SSM "+mean_label[i])
    im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,ihesp_sal_smean[i],levels=np.linspace(33.5,35.5,40),transform=ccrs.PlateCarree(),extend='both')
    # #Inserindo uma colorbar
    cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
    cbar.set_label(label='Salinidade', size=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(33.5,35.5,9,endpoint='True'))
    props = dict(boxstyle='round', alpha=0)
    ax.text(0.46, 0.24, 'M�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    ax.text(0.46, 0.12, str(np.float16(np.nanmean(ihesp_sal_smean[i][ihesp_sal_smean[i]!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/IHESP_SSM_'+mean_label[i]+'.png',bbox_inches='tight')
    
    
    fig,ax = make_south_map(dpi=180,name = "WOA18 SSM "+mean_label[i])
    im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,woa18_sal_smean[i],levels=np.linspace(33.5,35.5,40),transform=ccrs.PlateCarree(),extend='both')
    # #Inserindo uma colorbar
    cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
    cbar.set_label(label='Salinidade', size=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(33.5,35.5,9,endpoint='True'))
    props = dict(boxstyle='round', alpha=0)
    ax.text(0.46, 0.24, 'M�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    ax.text(0.46, 0.12, str(np.float16(np.nanmean(woa18_sal_smean[i][woa18_sal_smean[i]!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/WOA18_SSM_'+mean_label[i]+'.png',bbox_inches='tight')
    
    
    
    fig,ax = make_south_map(dpi=180,name = "IHESP TSM "+mean_label[i])
    im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,ihesp_tem_smean[i],levels=np.linspace(-2.5,15,40),transform=ccrs.PlateCarree(),extend='both',cmap='cmo.balance')
    # #Inserindo uma colorbar
    cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
    cbar.set_label(label='�C', size=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(-1,13,8,endpoint='True'))
    props = dict(boxstyle='round', alpha=0)
    ax.text(0.46, 0.24, 'M�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    ax.text(0.46, 0.12, str(np.float16(np.nanmean(ihesp_tem_smean[i][ihesp_tem_smean[i]!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/IHESP_TSM_'+mean_label[i]+'.png',bbox_inches='tight')
    
    
    
    fig,ax = make_south_map(dpi=180,name = "WOA18 TSM "+mean_label[i])
    im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,woa18_tem_smean[i],levels=np.linspace(-2.5,15,40),transform=ccrs.PlateCarree(),extend='both',cmap='cmo.balance')
    # #Inserindo uma colorbar
    cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
    cbar.set_label(label='�C', size=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(-1,13,8,endpoint='True'))
    props = dict(boxstyle='round', alpha=0)
    ax.text(0.46, 0.24, 'M�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    ax.text(0.46, 0.12, str(np.float16(np.nanmean(woa18_tem_smean[i][woa18_tem_smean[i]!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/WOA18_TSM_'+mean_label[i]+'.png',bbox_inches='tight')
    

    
    fig,ax = make_south_map(dpi=180,name = "RMSE TSM "+mean_label[i])
    im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,rmse_tem_smean[i],levels=np.linspace(0,2,40),transform=ccrs.PlateCarree(),cmap='plasma',extend='both')
    # #Inserindo uma colorbar
    cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
    cbar.set_label(label='RMSE', size=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(0,2,9,endpoint='True'))
    props = dict(boxstyle='round', alpha=0)
    ax.text(0.46, 0.24, 'RMSE m�dio', transform=ax.transAxes,
            fontsize=25,verticalalignment='top', bbox=props)
    ax.text(0.46, 0.12, str(np.float16(np.nanmean(rmse_tem_smean[i][rmse_tem_smean[i]!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/RMSE_TSM'+mean_label[i]+'.png',bbox_inches='tight')

    
    
    
    
    fig,ax = make_south_map(dpi=180,name = "RMSE SSM "+mean_label[i])
    im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,rmse_sal_smean[i],levels=np.linspace(0,0.7,40),transform=ccrs.PlateCarree(),cmap='plasma',extend='both')
    # #Inserindo uma colorbar
    cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
    cbar.set_label(label='RMSE', size=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(0,0.7,8,endpoint='True'))
    props = dict(boxstyle='round', alpha=0)
    ax.text(0.46, 0.24, 'RMSE m�dio', transform=ax.transAxes,
            fontsize=25,verticalalignment='top', bbox=props)
    ax.text(0.46, 0.12, str(np.float16(np.nanmean(rmse_sal_smean[i][rmse_sal_smean[i]!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/RMSE_SSM'+mean_label[i]+'.png',bbox_inches='tight')
    
    
    
    
    
    fig,ax = make_south_map(dpi=180,name = "(IHESP - WOA18) TSM "+mean_label[i])
    im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,dif_tem_smean[i],levels=np.linspace(-2.5,2.5,40),transform=ccrs.PlateCarree(),cmap='cmo.balance',extend='both')
    # #Inserindo uma colorbar
    cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
    cbar.set_label(label='�C', size=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(-2.5,2.5,9,endpoint='True'))
    props = dict(boxstyle='round', alpha=0)
    ax.text(0.46, 0.24, 'Dif. m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    ax.text(0.46, 0.12, str(np.float16(np.nanmean(dif_tem_smean[i][dif_tem_smean[i]!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/DIF_TSM'+mean_label[i]+'.png',bbox_inches='tight')
    
    fig,ax = make_south_map(dpi=180,name = "(IHESP - WOA18) SSM "+mean_label[i])
    im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,dif_sal_smean[i],levels=np.linspace(-0.5,0.5,40),transform=ccrs.PlateCarree(),extend='both')
    # #Inserindo uma colorbar
    cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
    cbar.set_label(label='Salinidade', size=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks(np.linspace(-0.5,0.5,9,endpoint='True'))
    props = dict(boxstyle='round', alpha=0)
    ax.text(0.46, 0.24, 'Dif. m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    ax.text(0.46, 0.12, str(np.float16(np.nanmean(dif_sal_smean[i][dif_sal_smean[i]!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
    fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/DIF_SSM'+mean_label[i]+'.png',bbox_inches='tight')

####### FIGURA DIF DJF-JJA

dif_ihesp_sal_smean = ihesp_sal_smean[0] - ihesp_sal_smean[2]

fig,ax = make_south_map(dpi=180,name = "IHESP (DJF - JJA) SSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,dif_ihesp_sal_smean,
                 levels=np.linspace(-0.5,0.5,40),transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='Salinidade', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-0.5,0.5,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'Dif. m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(dif_ihesp_sal_smean[dif_ihesp_sal_smean!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/dif_ihesp_sal_smean.png',bbox_inches='tight')

dif_ihesp_tem_smean = ihesp_tem_smean[0] - ihesp_tem_smean[2]

fig,ax = make_south_map(dpi=180,name = "IHESP (DJF - JJA) TSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,dif_ihesp_tem_smean,
                 levels=np.linspace(-2.5,2.5,40),cmap='cmo.balance',transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='�C', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-2.5,2.5,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'Dif. m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(dif_ihesp_tem_smean[dif_ihesp_tem_smean!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/dif_ihesp_tem_smean.png',bbox_inches='tight')

dif_woa18_sal_smean = woa18_sal_smean[0] - woa18_sal_smean[2]

fig,ax = make_south_map(dpi=180,name = "WOA18 (DJF - JJA) SSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,dif_woa18_sal_smean,
                 levels=np.linspace(-0.5,0.5,40),transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='Salinidade', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-0.5,0.5,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'Dif. m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(dif_woa18_sal_smean[dif_woa18_sal_smean!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/dif_woa18_sal_smean.png',bbox_inches='tight')

dif_woa18_tem_smean = woa18_tem_smean[0] - woa18_tem_smean[2]

fig,ax = make_south_map(dpi=180,name = "WOA18 (DJF - JJA) TSM")
im = ax.contourf(rec_woa18_lon_2d,rec_woa18_lat_2d,dif_woa18_tem_smean,
                 levels=np.linspace(-2.5,2.5,40),cmap='cmo.balance',transform=ccrs.PlateCarree(),extend='both')
# #Inserindo uma colorbar
cbar = plt.colorbar(im, ax=ax, aspect=20,extend='both',fraction=0.015)
cbar.set_label(label='�C', size=10)
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.linspace(-2.5,2.5,9,endpoint='True'))
props = dict(boxstyle='round', alpha=0)
ax.text(0.46, 0.24, 'Dif. m�dia', transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
ax.text(0.46, 0.12, str(np.float16(np.nanmean(dif_woa18_tem_smean[dif_woa18_tem_smean!=np.inf]))), transform=ax.transAxes, fontsize=25,verticalalignment='top', bbox=props)
fig.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/dif_woa18_tem_smean.png',bbox_inches='tight')

# FIGURA DA MEDIA DAS LONGITUDES POR LATITUDE
#Vou precisar dos:

#rec_woa18_lon_2d,rec_woa18_lat_2d = np.meshgrid(rec_woa18_lon,rec_woa18_lat)

#np.mean(ihesp_tem_interpol,axis=0) - np.mean(rec_woa18_tem,axis=0)
#np.mean(ihesp_sal_interpol,axis=0) - np.mean(rec_woa18_sal,axis=0)

TEM_TOTAL_IHESP = np.mean(ihesp_tem_interpol,axis=0)

SAL_TOTAL_IHESP = np.mean(ihesp_sal_interpol,axis=0)

TEM_TOTAL_WOA18 = np.mean(rec_woa18_tem,axis=0)

SAL_TOTAL_WOA18 = np.mean(rec_woa18_sal,axis=0)

#Conjunto de figuras media na latitude TEMP

med_tot_lat_ihesp_tem = media_na_latitude(TEM_TOTAL_IHESP)

med_tot_lat_ihesp_sal = media_na_latitude(SAL_TOTAL_IHESP)

med_tot_lon_ihesp_tem = media_na_longitude(TEM_TOTAL_IHESP)

med_tot_lon_ihesp_sal = media_na_longitude(SAL_TOTAL_IHESP)

med_tot_lat_woa18_tem = np.nanmean(TEM_TOTAL_WOA18, axis=0)

med_tot_lat_woa18_sal = np.nanmean(SAL_TOTAL_WOA18, axis=0)

med_tot_lon_woa18_tem = np.nanmean(TEM_TOTAL_WOA18, axis=1)

med_tot_lon_woa18_sal = np.nanmean(SAL_TOTAL_WOA18, axis=1)

fig_lat,ax = plt.subplots(1,1)

ax.plot(rec_woa18_lon,med_tot_lat_woa18_tem,label = 'WOA18')

ax.plot(rec_woa18_lon,med_tot_lon_ihesp_tem ,label = 'IHESP')

ax.plot(rec_woa18_lon,med_tot_lon_ihesp_tem  - med_tot_lat_woa18_tem,label = 'IHESP - WOA18')

ax.legend()

ax.set(xlabel='Longitude',ylabel='�C')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/TOTAL_media_lat_tem.png',bbox_inches='tight')

#Conjunto de figuras media na latitude SAL

fig_lat,ax = plt.subplots(1,1)

ax.plot(rec_woa18_lon,med_tot_lat_woa18_sal,label = 'WOA18')

ax.plot(rec_woa18_lon,med_tot_lon_ihesp_sal ,label = 'IHESP')

dif_sal = med_tot_lon_ihesp_sal  - med_tot_lat_woa18_sal

ax.plot(rec_woa18_lon,dif_sal+35,label = '(IHESP - WOA18) + 35')

ax.legend()

ax.set(xlabel='Longitude',ylabel='Salinidade')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/TOTAL_media_lat_sal.png',bbox_inches='tight')

#Conjunto de figuras media na longitude TEMP

fig_lat,ax = plt.subplots(1,1)

ax.plot(med_tot_lon_woa18_tem, rec_woa18_lat,label = 'WOA18')

ax.plot(med_tot_lat_ihesp_tem, rec_woa18_lat ,label = 'IHESP')

ax.plot(med_tot_lat_ihesp_tem  - med_tot_lon_woa18_tem, rec_woa18_lat,label = 'IHESP - WOA18')

ax.legend()

ax.set(xlabel='�C',ylabel='Latitude')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/TOTAL_media_lon_tem.png',bbox_inches='tight')

#Conjunto de figuras media na longitude TSAL

fig_lat,ax = plt.subplots(1,1)

ax.plot(med_tot_lon_woa18_sal, rec_woa18_lat,label = 'WOA18')

ax.plot(med_tot_lat_ihesp_sal, rec_woa18_lat ,label = 'IHESP')

dif_sal = med_tot_lat_ihesp_sal  - med_tot_lon_woa18_sal

ax.plot(dif_sal + 35, rec_woa18_lat,label = '(IHESP - WOA18) + 35')

ax.legend()

ax.set(xlabel='Salinidade',ylabel='Latitude')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/TOTAL_media_lon_sal.png',bbox_inches='tight')

##################

#ihesp_tem_smean
#woa18_tem_smean

#ihesp_sal_smean
#woa18_sal_smean

#dif_tem_smean
#dif_sal_smean

#DFJ

#Conjunto de figuras media na latitude TEMP

med_tot_lat_ihesp_tem = media_na_latitude(ihesp_tem_smean[0])

med_tot_lat_ihesp_sal = media_na_latitude(ihesp_sal_smean[0])

med_tot_lon_ihesp_tem = media_na_longitude(ihesp_tem_smean[0])

med_tot_lon_ihesp_sal = media_na_longitude(ihesp_sal_smean[0])

med_tot_lat_woa18_tem = np.nanmean(woa18_tem_smean[0], axis=0)

med_tot_lat_woa18_sal = np.nanmean(woa18_sal_smean[0], axis=0)

med_tot_lon_woa18_tem = np.nanmean(woa18_tem_smean[0], axis=1)

med_tot_lon_woa18_sal = np.nanmean(woa18_sal_smean[0], axis=1)

fig_lat,ax = plt.subplots(1,1)

ax.plot(rec_woa18_lon,med_tot_lat_woa18_tem,label = 'WOA18')

ax.plot(rec_woa18_lon,med_tot_lon_ihesp_tem ,label = 'IHESP')

ax.plot(rec_woa18_lon,med_tot_lon_ihesp_tem  - med_tot_lat_woa18_tem,label = 'IHESP - WOA18')

ax.legend()

ax.set(xlabel='Longitude',ylabel='�C')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/DJF_media_lat_tem.png',bbox_inches='tight')

#Conjunto de figuras media na latitude SAL

fig_lat,ax = plt.subplots(1,1)

ax.plot(rec_woa18_lon,med_tot_lat_woa18_sal,label = 'WOA18')

ax.plot(rec_woa18_lon,med_tot_lon_ihesp_sal ,label = 'IHESP')

dif_sal = med_tot_lon_ihesp_sal  - med_tot_lat_woa18_sal

ax.plot(rec_woa18_lon,dif_sal+35,label = '(IHESP - WOA18) + 35')

ax.legend()

ax.set(xlabel='Longitude',ylabel='Salinidade')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/DJF_media_lat_sal.png',bbox_inches='tight')

#Conjunto de figuras media na longitude TEMP

fig_lat,ax = plt.subplots(1,1)

ax.plot(med_tot_lon_woa18_tem, rec_woa18_lat,label = 'WOA18')

ax.plot(med_tot_lat_ihesp_tem, rec_woa18_lat ,label = 'IHESP')

ax.plot(med_tot_lat_ihesp_tem  - med_tot_lon_woa18_tem, rec_woa18_lat,label = 'IHESP - WOA18')

ax.legend()

ax.set(xlabel='�C',ylabel='Latitude')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/DJF_media_lon_tem.png',bbox_inches='tight')

#Conjunto de figuras media na longitude TSAL

fig_lat,ax = plt.subplots(1,1)

ax.plot(med_tot_lon_woa18_sal, rec_woa18_lat,label = 'WOA18')

ax.plot(med_tot_lat_ihesp_sal, rec_woa18_lat ,label = 'IHESP')

dif_sal = med_tot_lat_ihesp_sal  - med_tot_lon_woa18_sal

ax.plot(dif_sal + 35, rec_woa18_lat,label = '(IHESP - WOA18) + 35')

ax.legend()

ax.set(xlabel='Salinidade',ylabel='Latitude')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/DJF_media_lon_sal.png',bbox_inches='tight')

#JJA

#Conjunto de figuras media na latitude TEMP

med_tot_lat_ihesp_tem = media_na_latitude(ihesp_tem_smean[1])

med_tot_lat_ihesp_sal = media_na_latitude(ihesp_sal_smean[1])

med_tot_lon_ihesp_tem = media_na_longitude(ihesp_tem_smean[1])

med_tot_lon_ihesp_sal = media_na_longitude(ihesp_sal_smean[1])

med_tot_lat_woa18_tem = np.nanmean(woa18_tem_smean[1], axis=0)

med_tot_lat_woa18_sal = np.nanmean(woa18_sal_smean[1], axis=0)

med_tot_lon_woa18_tem = np.nanmean(woa18_tem_smean[1], axis=1)

med_tot_lon_woa18_sal = np.nanmean(woa18_sal_smean[1], axis=1)

fig_lat,ax = plt.subplots(1,1)

ax.plot(rec_woa18_lon,med_tot_lat_woa18_tem,label = 'WOA18')

ax.plot(rec_woa18_lon,med_tot_lon_ihesp_tem ,label = 'IHESP')

ax.plot(rec_woa18_lon,med_tot_lon_ihesp_tem  - med_tot_lat_woa18_tem,label = 'IHESP - WOA18')

ax.legend()

ax.set(xlabel='Longitude',ylabel='�C')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/JJA_media_lat_tem.png',bbox_inches='tight')

#Conjunto de figuras media na latitude SAL

fig_lat,ax = plt.subplots(1,1)

ax.plot(rec_woa18_lon,med_tot_lat_woa18_sal,label = 'WOA18')

ax.plot(rec_woa18_lon,med_tot_lon_ihesp_sal ,label = 'IHESP')

dif_sal = med_tot_lon_ihesp_sal  - med_tot_lat_woa18_sal

ax.plot(rec_woa18_lon,dif_sal+35,label = '(IHESP - WOA18) + 35')

ax.legend()

ax.set(xlabel='Longitude',ylabel='Salinidade')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/JJA_media_lat_sal.png',bbox_inches='tight')

#Conjunto de figuras media na longitude TEMP

fig_lat,ax = plt.subplots(1,1)

ax.plot(med_tot_lon_woa18_tem, rec_woa18_lat,label = 'WOA18')

ax.plot(med_tot_lat_ihesp_tem, rec_woa18_lat ,label = 'IHESP')

ax.plot(med_tot_lat_ihesp_tem  - med_tot_lon_woa18_tem, rec_woa18_lat,label = 'IHESP - WOA18')

ax.legend()

ax.set(xlabel='�C',ylabel='Latitude')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/JJA_media_lon_tem.png',bbox_inches='tight')

#Conjunto de figuras media na longitude TSAL

fig_lat,ax = plt.subplots(1,1)

ax.plot(med_tot_lon_woa18_sal, rec_woa18_lat,label = 'WOA18')

ax.plot(med_tot_lat_ihesp_sal, rec_woa18_lat ,label = 'IHESP')

dif_sal = med_tot_lat_ihesp_sal  - med_tot_lon_woa18_sal

ax.plot(dif_sal + 35, rec_woa18_lat,label = '(IHESP - WOA18) + 35')

ax.legend()

ax.set(xlabel='Salinidade',ylabel='Latitude')  

fig_lat.savefig(r'/home/lucastartaro/programas_ihesp/figures/figures_reu_28_06_2022/JJA_media_lon_sal.png',bbox_inches='tight')
