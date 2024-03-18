#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:21:46 2023

@author: nicholasforcone
"""
import numpy as np
import xarray as xa
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.dates import date2num, num2date, DateFormatter, MonthLocator, DayLocator, HourLocator
from coare36vn_zrf_et_wa import coare36vn_zrf_et_wa
from tropycal import tracks, recon


class Saildrone:
    """ Holds Saildrone Data 
    
    Methods available for computing fluxes
    """
    def __init__(self, name, filepath):
        self.name = name
        self.filePath = filepath
        self.data = xa.open_dataset(filepath)
        
    def show_metadata(self):
        print(self.data)
        print("\nData variables: ", self.data.data_vars)


class SnappingCursor:
    """
    A cross-hair cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """
    def __init__(self, ax, line):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        self.x, self.y = line.get_data()
        self._last_index = None
        # text location in axes coords
        self.text = ax.text(0.4, 2.4, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            index = min(np.searchsorted(self.x, x), len(self.x) - 1)
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x = self.x[index]
            y = self.y[index]
            # update the line positions
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            self.text.set_text(f'x={num2date(x)}, y={y:1.2f}')
            self.ax.figure.canvas.draw()


# instantiate Saildrone object
SD = Saildrone("1069", "/Users/nicholasforcone/SeniorResearchAll/sd1069_2019_c297_1b1e_6c16.nc")

# binned hourly means
start_date = SD.data.time.values[0]
end_date = SD.data.time.values[-1]
center_time = xa.date_range(start_date, end_date, freq="1 H")
bin_edge = (center_time - pd.Timedelta(30, "minutes"))
bin_edge = bin_edge.insert(len(bin_edge), bin_edge[-1] + pd.Timedelta(60, "minutes"))
mean60min = SD.data.groupby_bins(SD.data.time, bin_edge).mean()

#%%
from coare36vn_zrf_et_nf import coare36vn_zrf_et_nf
# run coare w/ no gustiness parameter
coare_out = coare36vn_zrf_et_nf(
    u=mean60min.wind_speed.values, # scalar mean wind
    zu=5.2,
    t=mean60min.TEMP_AIR_MEAN.values,
    zt=2.3,
    rh=mean60min.RH_MEAN.values,
    zq=2.3,
    P=np.nan_to_num(mean60min.BARO_PRES_MEAN.values, nan=1015),
    ts=mean60min.TEMP_SBE37_MEAN.values,
    sw_dn=mean60min.SW_IRRAD_TOTAL_MEAN.values,
    lw_dn=mean60min.LW_IRRAD_MEAN.values,
    lat=mean60min.latitude.values,
    lon=mean60min.longitude.values,
    jd=center_time.to_julian_date().to_numpy(),
    zi=600.0,
    rain=np.full(len(center_time), np.nan),
    Ss=mean60min.SAL_SBE37_MEAN.values)

# hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
# hlb = latent heat flux (W/m^2) ... positive for qair < qs
sensible = coare_out[:, 2]
latent = coare_out[:, 3]
thflx = sensible + latent # positive cools the ocean (heats the atmosphere)

# run coare w/ parameterized gustiness ug
coare_out_p = coare36vn_zrf_et_wa(
    u=np.sqrt(mean60min.UWND_MEAN.values**2 + mean60min.VWND_MEAN.values**2), # vector mean wind
    zu=5.2,
    t=mean60min.TEMP_AIR_MEAN.values,
    zt=2.3,
    rh=mean60min.RH_MEAN.values,
    zq=2.3,
    P=np.nan_to_num(mean60min.BARO_PRES_MEAN.values, nan=1015),
    ts=mean60min.TEMP_SBE37_MEAN.values,
    sw_dn=mean60min.SW_IRRAD_TOTAL_MEAN.values,
    lw_dn=mean60min.LW_IRRAD_MEAN.values,
    lat=mean60min.latitude.values,
    lon=mean60min.longitude.values,
    jd=center_time.to_julian_date().to_numpy(),
    zi=600.0,
    rain=np.full(len(center_time), np.nan),
    Ss=mean60min.SAL_SBE37_MEAN.values)

ug = coare_out_p[:, 47]
SD_missing_wind_var = mean60min.wind_speed.values**2 - (mean60min.UWND_MEAN.values**2 + mean60min.VWND_MEAN.values**2)

# hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
# hlb = latent heat flux (W/m^2) ... positive for qair < qs
sensible_p = coare_out_p[:, 2]
latent_p = coare_out_p[:, 3]
thflx_p = sensible_p + latent_p # positive cools the ocean (heats the atmosphere)

#%% without THFLX
fig02, ax02 = plt.subplots()
sc = ax02.scatter(date2num(center_time), SD_missing_wind_var, c=mean60min.TEMP_SBE37_MEAN.values,linewidth=0.7, label=r"$\langle U^{2} + V^{2} \rangle - (\langle U \rangle^2 + \langle V \rangle^2) \: \langle\rangle_{60 \, min}$")
ax02.plot(date2num(center_time), ug**2, label="$Ug^2$ from COARE", color="black", linewidth=0.7)
ax02.xaxis.set_major_locator(MonthLocator())
ax02.xaxis.set_major_formatter(DateFormatter('%b'))
fig02.colorbar(sc, label='SST (degC)')
ax02.set_ylabel("$m^2 / s^2$")
ax02.legend(loc='upper left', fontsize='small')
ax02.set_title('Measured Missing Wind Variance & $Ug^2$')
# fig02.savefig('/Users/nicholasforcone/Library/CloudStorage/OneDrive-UniversityofMiami/Nick Research Shared Folder/poster/fig3_v2.png', dpi=400)

#%%
fig02, ax02 = plt.subplots(3, sharex=True, constrained_layout=True)
sc = ax02[0].scatter(date2num(center_time), SD_missing_wind_var, s=22, color='blueviolet', label=r"$\langle U^{2} + V^{2} \rangle - (\langle U \rangle^2 + \langle V \rangle^2) \: \langle\rangle_{60 \, min}$")
ax02[0].plot(date2num(center_time), ug**2, label="$Ug^2$ from COARE", color="black", linewidth=0.7)
ax02[1].plot(date2num(center_time), thflx, label="THFLX w/ Scalar Mean Wind", color='red')
ax02[1].plot(date2num(center_time), thflx_p, label="THFLX w/ Vector Mean Wind & Ug",color='black')
ax02[2].scatter(date2num(center_time), (thflx_p * 100) / thflx, s=22, label="THFLX_Ug / THFLX",color='black')

# trim range
begin_plot = np.datetime64('2019-12-26 00:00:00')
end_plot = np.datetime64('2019-12-27 00:00:00')
ax02[0].set_xlim(begin_plot, end_plot)

ax02[0].xaxis.set_major_locator(DayLocator())
ax02[0].xaxis.set_minor_locator(HourLocator())
ax02[0].xaxis.set_major_formatter(DateFormatter('%b %d'))
ax02[0].set_ylabel("$m^2 / s^2$")
ax02[1].set_ylabel("$W / m^2$")
ax02[2].set_ylabel("%")
ax02[0].legend(fontsize='x-small')
ax02[1].legend(fontsize='x-small')
# ax02[2].legend(fontsize='x-small', loc='lower left')
ax02[0].set_title('Measured Missing Wind Variance & $Ug^2$')
ax02[1].set_title('Turbulent Heat Flux Comparison')
ax02[2].set_title('THFLX_Ug / THFLX')

ax02[0].set_ylim(top=10)
ax02[1].set_ylim(top=200)
ax02[2].set_ylim(bottom=60, top=110)

# fig02.savefig('/Users/nicholasforcone/Library/CloudStorage/OneDrive-UniversityofMiami/Nick Research Shared Folder/poster/fig6_v1.png', dpi=600)

#%%
import cartopy.crs as ccrs
fig03 = plt.figure()
ax03 = plt.axes(projection=ccrs.PlateCarree())
ax03.coastlines()
ax03.stock_img()
gl = ax03.gridlines(draw_labels=True, linewidth=1.5, color='gray', alpha=0.5, linestyle='--',zorder=0)
gl.right_labels = False
sc2=ax03.scatter(SD.data.longitude, SD.data.latitude, c=SD.data.TEMP_SBE37_MEAN, s=10, cmap='inferno', label="Saildrone track")
ax03.set_extent([-162, -135, -4.5, 25.5])
fig03.colorbar(sc2, label='SST (degC)')
ax03.legend()
# fig03.savefig('/Users/nicholasforcone/Library/CloudStorage/OneDrive-UniversityofMiami/Nick Research Shared Folder/poster/fig4_v1.png', dpi=400)

#%%
class SnappingCursorScat:
    """
    A cross-hair cursor that snaps to the data point of a PathCollection 
    object
    """
    def __init__(self, ax, collection):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        self.offsets = collection.get_offsets()
        self._last_index = None
        # text location in data coordinates
        self.text = ax.text(0, 0, '', transform=ax.transData)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                distances = np.sqrt(np.sum((self.offsets - np.array([x, y]))**2, axis=1))
                index = np.argmin(distances)
                x, y = self.offsets[index]
                # update the line positions
                self.horizontal_line.set_ydata([y])
                self.vertical_line.set_xdata([x])
                self.text.set_position((x, y))
                self.text.set_text(f'x={x:1.4f}, y={y:1.4f}')
                self.ax.figure.canvas.draw()

#%%
def onpick(event):
    ind = event.ind
    print('onpick scatter:', ind, x[ind], SD_missing_wind_var[ind])


fig04, ax04 = plt.subplots()
x = ug / 1.25
ax04.scatter(x, SD_missing_wind_var, picker=True)
fig04.canvas.mpl_connect('pick_event', onpick)

#%%
fig05, ax05 = plt.subplots()
ax05.set_xlabel('ug / 1.25')
ax05.set_ylabel('SD_missing_wind_var')

# preprocessing
X = x[~np.isnan(x)] #this should be improved (nans not necessarily in same place)
Y = SD_missing_wind_var[~np.isnan(SD_missing_wind_var)]
    # remove default gustiness
not_default = X != 0.16
X = X[not_default].reshape(-1, 1)
Y = Y[not_default].reshape(-1, 1)

point_collection = ax05.scatter(X, Y, s=1, c='black')
snap_cursor_scat = SnappingCursorScat(ax05, point_collection)
fig05.canvas.mpl_connect('motion_notify_event', snap_cursor_scat.on_mouse_move)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)
                            
print('coefficitent: ', reg.coef_)
print('intercept: ', reg.intercept_)

