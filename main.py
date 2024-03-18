#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:21:46 2023

@author: nforcone@umich.edu
"""
import numpy as np
import xarray as xa
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.dates import date2num, num2date, DateFormatter, MonthLocator
from matplotlib.dates import DayLocator, HourLocator
from sklearn.linear_model import LinearRegression
from COARE.coare36vn_zrf_et_wa import coare36vn_zrf_et_wa
from COARE.coare36vn_zrf_et_nf import coare36vn_zrf_et_nf


# %%
class Saildrone:
    """ Stores Saildrone Data
    """

    def __init__(self, name, filepath):
        self.name = name
        self.filePath = filepath
        self.data = xa.open_dataset(filepath)

    def show_metadata(self):
        print(self.data)
        print("\nData variables: ", self.data.data_vars)


# %%
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


# %%
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
            distances = np.sqrt(np.sum((self.offsets - np.array([x, y]))**2, axis=1))
            index = np.argmin(distances)
            x, y = self.offsets[index]
            # update the line positions
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            self.text.set_position((x, y))
            self.text.set_text(f'x={x:1.4f}, y={y:1.4f}')
            self.ax.figure.canvas.draw()


# %%
# instantiate Saildrone object
ROOT = "/Users/nicholasforcone/GustinessData/"
FILE = "sd1069_2019_c297_1b1e_6c16.nc"
SD = Saildrone("1069", ROOT + FILE)

# binned hourly means
start_date = SD.data.time.values[0]
end_date = SD.data.time.values[-1]
center_time = xa.date_range(start_date, end_date, freq="1 H")
bin_edge = (center_time - pd.Timedelta(30, "minutes"))
bin_edge = bin_edge.insert(len(bin_edge), bin_edge[-1]
                           + pd.Timedelta(60, "minutes"))
mean60min = SD.data.groupby_bins(SD.data.time, bin_edge).mean()

# %%
# run COARE with no gustiness parameter
coare_out = coare36vn_zrf_et_nf(
    u=mean60min.wind_speed.values,  # scalar mean wind
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
    rain=np.full(len(center_time), np.nan),  # nan array for rain
    Ss=mean60min.SAL_SBE37_MEAN.values)

# hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
# hlb = latent heat flux (W/m^2) ... positive for qair < qs
sensible = coare_out[:, 2]
latent = coare_out[:, 3]
thflx = sensible + latent  # positive cools the ocean (heats the atmosphere)

# run coare with parameterized gustiness ug
coare_out_p = coare36vn_zrf_et_wa(
    u=np.sqrt(mean60min.UWND_MEAN.values**2 + mean60min.VWND_MEAN.values**2),  # vector mean wind
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
thflx_p = sensible_p + latent_p  # positive cools the ocean (heats the atmosphere)

# %%
# measured missing wind variance and ug^2
fig01, ax01 = plt.subplots()

# missing wind variance colored by SST
sc01 = ax01.scatter(date2num(center_time),
                    SD_missing_wind_var,
                    c=mean60min.TEMP_SBE37_MEAN.values,
                    linewidth=0.7,
                    label=r"$\langle U^{2} + V^{2} \rangle - (\langle U \rangle^2 + \langle V \rangle^2) \: \langle\rangle_{60 \, min}$")

# ug^2
ax01.plot(date2num(center_time),
          ug**2,
          color="black",
          linewidth=0.7,
          label="$Ug^2$ from COARE")

# properties
ax01.xaxis.set_major_locator(MonthLocator())
ax01.xaxis.set_major_formatter(DateFormatter('%b'))
fig01.colorbar(sc01, label='SST (degC)')
ax01.set_ylabel("$m^2 / s^2$")
ax01.legend(loc='upper left', fontsize='small')
ax01.set_title('Measured Missing Wind Variance & $Ug^2$')

# %%
# measured missing wind variance and ug^2
# turbulent heat flux comparison
fig02, ax02 = plt.subplots(3, sharex=True, constrained_layout=True)

# [TOP] missing wind variance
sc02 = ax02[0].scatter(date2num(center_time),
                       SD_missing_wind_var,
                       s=22,
                       color='blueviolet',
                       label=r"$\langle U^{2} + V^{2} \rangle - (\langle U \rangle^2 + \langle V \rangle^2) \: \langle\rangle_{60 \, min}$")
# [TOP] ug^2
ax02[0].plot(date2num(center_time),
             ug**2,
             label="$Ug^2$ from COARE",
             color="black",
             linewidth=0.7)
# [MIDDLE] thflx with scalar mean wind
ax02[1].plot(date2num(center_time),
             thflx,
             label="THFLX w/ Scalar Mean Wind",
             color='red')
# [MIDDLE] thflx with vector mean wind and parameterized gustiness
ax02[1].plot(date2num(center_time),
             thflx_p,
             label="THFLX w/ Vector Mean Wind & Ug",
             color='black')
# [BOTTOM]
ax02[2].scatter(date2num(center_time),
                (thflx_p * 100) / thflx,
                s=22,
                label="THFLX_Ug / THFLX",
                color='black')

# trim range
begin_plot = np.datetime64('2019-12-26 00:00:00')
end_plot = np.datetime64('2019-12-27 00:00:00')

# properties
ax02[0].set_xlim(begin_plot, end_plot)
ax02[0].xaxis.set_major_locator(DayLocator())
ax02[0].xaxis.set_minor_locator(HourLocator())
ax02[0].xaxis.set_major_formatter(DateFormatter('%b %d'))
ax02[0].set_ylabel("$m^2 / s^2$")
ax02[1].set_ylabel("$W / m^2$")
ax02[2].set_ylabel("%")
ax02[0].legend(fontsize='x-small')
ax02[1].legend(fontsize='x-small')
ax02[0].set_title('Measured Missing Wind Variance & $Ug^2$')
ax02[1].set_title('Turbulent Heat Flux Comparison')
ax02[2].set_title('THFLX_Ug / THFLX')
ax02[0].set_ylim(top=10)
ax02[1].set_ylim(top=200)
ax02[2].set_ylim(bottom=60, top=110)

# %%
# Saildrone track colored by SST
fig03 = plt.figure()
ax03 = plt.axes(projection=ccrs.PlateCarree())

sc03 = ax03.scatter(SD.data.longitude,
                    SD.data.latitude,
                    c=SD.data.TEMP_SBE37_MEAN,
                    s=10,
                    cmap='inferno',
                    label="Saildrone track")

# properties
ax03.coastlines()
ax03.stock_img()
gl = ax03.gridlines(draw_labels=True,
                    linewidth=1.5,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                    zorder=0)
gl.right_labels = False
ax03.set_extent([-162, -135, -4.5, 25.5])
fig03.colorbar(sc03, label='SST (degC)')
ax03.legend()


# %%
def onpick(event):
    ind = event.ind
    print('onpick scatter:', ind, x[ind], SD_missing_wind_var[ind])


fig04, ax04 = plt.subplots()
x = ug / 1.25
ax04.scatter(x, SD_missing_wind_var, picker=True)
fig04.canvas.mpl_connect('pick_event', onpick)

# %%
# empirical beta
fig05, ax05 = plt.subplots()

# preprocessing
# this should be improved (nans not necessarily in same place)
x = ug / 1.25
X = x[~np.isnan(x)]
Y = SD_missing_wind_var[~np.isnan(SD_missing_wind_var)]
# remove default gustiness
not_default = X != 0.16
X = X[not_default].reshape(-1, 1)
Y = Y[not_default].reshape(-1, 1)

# properties
ax05.set_xlabel('ug / 1.25')
ax05.set_ylabel('SD_missing_wind_var')

# interactive
point_collection = ax05.scatter(X, Y, s=1, c='black')
snap_cursor_scat = SnappingCursorScat(ax05, point_collection)
fig05.canvas.mpl_connect('motion_notify_event', snap_cursor_scat.on_mouse_move)

reg = LinearRegression().fit(X, Y)  # fit a linear model
Y_pred = reg.predict(X)
ax05.plot(X, Y_pred, color='red')
ax05.text(0, 10, f'$R^2$ = {reg.score(X, Y):.3f}')
print('coefficitent: ', reg.coef_)
print('intercept: ', reg.intercept_)
