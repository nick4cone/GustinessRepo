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
from COARE.coare36vn_zrf_et import coare36vn_zrf_et  # from email thread
from COARE.coare_no_ug_param import coare_no_ug_param
from COARE.coare_new_ug_param import coare_new_ug_param
from scipy.fft import fft, fftfreq
from windrose import WindroseAxes
from datetime import datetime
import matplotlib.ticker as mticker


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

    def hourly_binned_mean(self):
        start_date = self.data.time.values[0]  # first timestamp
        end_date = self.data.time.values[-1]  # last timestamp

        # Return a fixed frequency datetime index
        self.center_time = xa.date_range(start_date, end_date, freq="1 H")

        # edges of averaging bins
        # bins straddle center_time
        bin_edge = (self.center_time - pd.Timedelta(30, "minutes"))

        # manually add in rightmost bin edge
        bin_edge = bin_edge.insert(len(bin_edge), bin_edge[-1]
                                   + pd.Timedelta(60, "minutes"))

        # splits data into bins
        # computes the mean of each bin
        # groups are combined back into a single data object
        self.mean60min = self.data.groupby_bins(self.data.time,
                                                bin_edge).mean()

        self.scalar_60min_mean_wind_speed = self.mean60min.wind_speed.values
        self.vector_60min_mean_wind_speed = np.sqrt(
            self.mean60min.UWND_MEAN.values ** 2
            + self.mean60min.VWND_MEAN.values ** 2)


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

    def __init__(self, ax, collection, zvar):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        self.offsets = collection.get_offsets()
        self.zvar = zvar
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
            x, y = event.xdata, event.ydata  # x and y coordinates of cursor
            distances = np.sqrt(np.sum((self.offsets - np.array([x, y]))**2,
                                       axis=1))
            index = np.argmin(distances)
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x, y = self.offsets[index]
            z = self.zvar[index][0]
            # update the line positions
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            self.text.set_position((x, y))
            self.text.set_text(f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}')
            self.ax.figure.canvas.draw()


# %%
# instantiate Saildrone object
ROOT = "/Users/nicholasforcone/Library/Mobile Documents/com~apple~CloudDocs/GustinessData/"
FILE = "sd1069_2019_c297_1b1e_6c16.nc"
SD = Saildrone("1069", ROOT + FILE)

# binned hourly means
SD.hourly_binned_mean()


# %%
# visualize and verify binning algorithm
# time bin left edges
left = []
bins = SD.mean60min.time_bins.values
for t in range(len(bins)):
    left.append(bins[t].left)
np.asarray(left)

fig00, ax00 = plt.subplots()
n = 10  # number of samples to plot
ymin = 0
ymax = 60
ax00.vlines(date2num(left[0:n]),
            ymin=ymin,
            ymax=ymax)
ax00.vlines(date2num(SD.center_time[0:n]),
            ymin=-ymin,
            ymax=ymax,
            linestyle='--',
            color='black')
ax00.plot(date2num(SD.data.time)[0:n*60],  # raw data
          SD.data.TEMP_AIR_MEAN[0:n*60],
          marker='^',
          markerfacecolor='red',
          markeredgecolor='red',
          markersize=2.5,
          color='red')
ax00.plot(date2num(SD.center_time)[0:n],  # 60 min mean
          SD.mean60min.TEMP_AIR_MEAN[0:n],
          marker='^',
          markerfacecolor='purple',
          markeredgecolor='purple',
          markersize=5,
          color='purple')
ax00.xaxis.set_major_locator(HourLocator())
ax00.xaxis.set_major_formatter(DateFormatter('%H:%M'))

# %%
# run COARE with no gustiness parameter
coare_out = coare_no_ug_param(
    u=SD.scalar_60min_mean_wind_speed,  # scalar mean wind speed
    zu=5.2,
    t=SD.mean60min.TEMP_AIR_MEAN.values,
    zt=2.3,
    rh=SD.mean60min.RH_MEAN.values,
    zq=2.3,
    P=np.nan_to_num(SD.mean60min.BARO_PRES_MEAN.values, nan=1015),  # fill nans
    ts=SD.mean60min.TEMP_SBE37_MEAN.values,
    sw_dn=SD.mean60min.SW_IRRAD_TOTAL_MEAN.values,
    lw_dn=SD.mean60min.LW_IRRAD_MEAN.values,
    lat=SD.mean60min.latitude.values,
    lon=SD.mean60min.longitude.values,
    jd=SD.center_time.to_julian_date().to_numpy(),
    zi=600.0,
    rain=np.full(len(SD.center_time), np.nan),  # nan array for rain
    Ss=SD.mean60min.SAL_SBE37_MEAN.values,
    sigH=SD.mean60min.WAVE_SIGNIFICANT_HEIGHT.values)

# hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
# hlb = latent heat flux (W/m^2) ... positive for qair < qs
sensible = coare_out[:, 2]
latent = coare_out[:, 3]
thflx = sensible + latent  # positive cools the ocean (heats the atmosphere)
tau = coare_out[:, 1]  # wind stress

# run coare with parameterized gustiness ug
coare_out_p = coare36vn_zrf_et(
    u=SD.vector_60min_mean_wind_speed,  # vector mean wind speed
    zu=5.2,
    t=SD.mean60min.TEMP_AIR_MEAN.values,
    zt=2.3,
    rh=SD.mean60min.RH_MEAN.values,
    zq=2.3,
    P=np.nan_to_num(SD.mean60min.BARO_PRES_MEAN.values, nan=1015),  # fill nans
    ts=SD.mean60min.TEMP_SBE37_MEAN.values,
    sw_dn=SD.mean60min.SW_IRRAD_TOTAL_MEAN.values,
    lw_dn=SD.mean60min.LW_IRRAD_MEAN.values,
    lat=SD.mean60min.latitude.values,
    lon=SD.mean60min.longitude.values,
    jd=SD.center_time.to_julian_date().to_numpy(),
    zi=600.0,
    rain=np.full(len(SD.center_time), np.nan),  # nan array for rain
    Ss=SD.mean60min.SAL_SBE37_MEAN.values,
    sigH=SD.mean60min.WAVE_SIGNIFICANT_HEIGHT.values)

# run coare with NEW parameterized gustiness
coare_out_p_new = coare_new_ug_param(
    u=SD.vector_60min_mean_wind_speed,  # vector mean wind speed
    zu=5.2,
    t=SD.mean60min.TEMP_AIR_MEAN.values,
    zt=2.3,
    rh=SD.mean60min.RH_MEAN.values,
    zq=2.3,
    P=np.nan_to_num(SD.mean60min.BARO_PRES_MEAN.values, nan=1015),  # fill nans
    ts=SD.mean60min.TEMP_SBE37_MEAN.values,
    sw_dn=SD.mean60min.SW_IRRAD_TOTAL_MEAN.values,
    lw_dn=SD.mean60min.LW_IRRAD_MEAN.values,
    lat=SD.mean60min.latitude.values,
    lon=SD.mean60min.longitude.values,
    jd=SD.center_time.to_julian_date().to_numpy(),
    zi=600.0,
    rain=np.full(len(SD.center_time), np.nan),  # nan array for rain
    Ss=SD.mean60min.SAL_SBE37_MEAN.values,
    sigH=SD.mean60min.WAVE_SIGNIFICANT_HEIGHT.values)

# gustiness parameter from COARE
ug = coare_out_p[:, 47]
ug_NEW = coare_out_p_new[:, 47]
SD_missing_wind_var = (SD.scalar_60min_mean_wind_speed ** 2 -
                       SD.vector_60min_mean_wind_speed ** 2)

# hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
# hlb = latent heat flux (W/m^2) ... positive for qair < qs
sensible_p = coare_out_p[:, 2]
latent_p = coare_out_p[:, 3]
thflx_p = sensible_p + latent_p  # positive cools the ocean
tau_p = coare_out_p[:, 1]

# %%
# csfont = {'fontname': 'Times New Roman'}
# plt.rcParams["font.family"] = "Times New Roman"
# measured missing wind variance and ug^2
fig01, ax01 = plt.subplots()

wdir = (np.arctan2(SD.mean60min.UWND_MEAN.values,
        SD.mean60min.VWND_MEAN.values) * 180 / np.pi) + 180

# missing wind variance colored by SST
sc01 = ax01.scatter(date2num(SD.center_time),
                    SD_missing_wind_var,
                    s=1,
                    c='black',
                    cmap='twilight',
                    linewidth=0.7,
                    label=r"$\langle U^{2} + V^{2} \rangle"
                    r"- (\langle U \rangle^2 + \langle V \rangle^2)"
                    r"\: \langle\rangle_{60 \, min}$")  # raw string

# ug^2
ax01.plot(date2num(SD.center_time),
          ug**2,
          color="orange",
          linewidth=0.7,
          label="$Ug^2$ from COARE")

# NEW ug^2
ax01.plot(date2num(SD.center_time),
          ug_NEW**2,
          color="forestgreen",
          linewidth=0.7,
          label="New $Ug^2$ from COARE")

# properties
ax01.xaxis.set_major_locator(MonthLocator())
ax01.xaxis.set_major_formatter(DateFormatter('%b'))
# fig01.colorbar(sc01, label='Wind Direction')
ax01.set_ylabel("$m^2 / s^2$")
ax01.legend(loc='upper left', fontsize='small')
ax01.set_title('Measured Missing Wind Variance & $Ug^2$')

# plt.savefig('/Users/nicholasforcone/GustinessRepo/FIGURES/new_v_old_' + 
#              datetime.now().strftime("%Y_%m_%d"), dpi=350)

old_diff = np.abs(SD_missing_wind_var - ug)
old_diff_mean = np.mean(old_diff[~np.isnan(old_diff)])

new_diff = np.abs(SD_missing_wind_var - ug_NEW)
new_diff_mean = np.mean(new_diff[~np.isnan(new_diff)])
print("old diff: ", old_diff_mean)
print("new diff: ", new_diff_mean)

fig09, ax09 = plt.subplots()
waluigi = np.arange(2, step=0.5)
ax09.plot(waluigi, waluigi)
ax09.scatter(ug, ug_NEW, color='black', s=5)
ax09.set_ylabel("New Param")
ax09.set_xlabel("Old Param")
ax09.grid()

# plt.savefig('/Users/nicholasforcone/GustinessRepo/FIGURES/new_v_old_scatter' + 
#               datetime.now().strftime("%Y_%m_%d"), dpi=350)

# %%
Warr = np.stack((wdir, SD.vector_60min_mean_wind_speed), axis=1)
Warr_no_nan = Warr[~np.isnan(Warr).any(axis=1)]  # remove any rows with nans

ax = WindroseAxes.from_ax()
ax.bar(Warr_no_nan[:, 0], Warr_no_nan[:, 1],
       opening=0.8, bins=4, edgecolor="white")
ax.set_legend(units='m/s', title='Mean Wind Vector Magnitude', loc='upper right',
              bbox_to_anchor=(1.25, 0.8))
ax.set_title("TPOS Saildrone 1069 Wind Rose", size=17)
rose_fig = ax.get_figure()
rose_fig.set_size_inches(9, 7)

# plt.savefig('/Users/nicholasforcone/GustinessRepo/FIGURES/rose_' + 
#             datetime.now().strftime("%Y_%m_%d"), dpi=350)

# Warr = np.stack((SD.data.wind_dir, SD.data.wind_speed), axis=1)
# Warr_no_nan = Warr[~np.isnan(Warr).any(axis=1)]  # remove any rows with nans

# ax = WindroseAxes.from_ax()
# ax.bar(Warr_no_nan[:, 0], Warr_no_nan[:, 1],
#        normed=True, opening=0.8, edgecolor="white")
# ax.set_legend()

# %%
# measured missing wind variance and ug^2
# turbulent heat flux comparison
fig02, ax02 = plt.subplots(3, sharex=True, constrained_layout=True)

# [TOP] missing wind variance
sc02 = ax02[0].scatter(date2num(SD.center_time),
                       SD_missing_wind_var,
                       s=22,
                       color='blueviolet',
                       # c=SD.mean60min.TEMP_SBE37_MEAN,
                       label=r"$\langle U^{2} + V^{2} \rangle"
                       r"- (\langle U \rangle^2 + \langle V \rangle^2)"
                       r"\: \langle\rangle_{60 \, min}$")  # raw string
# [TOP] ug^2
ax02[0].plot(date2num(SD.center_time),
             ug**2,
             label="$Ug^2$ from COARE",
             color="black",
             linewidth=0.7)
# [MIDDLE] thflx with scalar mean wind
ax02[1].plot(date2num(SD.center_time),
             thflx,
             label="THFLX w/ Scalar Mean Wind",
             color='red')
# [MIDDLE] thflx with vector mean wind and parameterized gustiness
ax02[1].plot(date2num(SD.center_time),
             thflx_p,
             label="THFLX w/ Vector Mean Wind & Ug",
             color='black')
# [BOTTOM]
ax02[2].scatter(date2num(SD.center_time),
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

# plt.savefig('/Users/nicholasforcone/Library/CloudStorage/'
#             'GoogleDrive-nforcone@umich.edu/My Drive/'
#             'Gustiness Paper/Figures/fig05.png', dpi=350)

# %%
# Saildrone track colored by SST
fig03 = plt.figure()
ax03 = plt.axes(projection=ccrs.PlateCarree())

sc03 = ax03.scatter(SD.data.longitude,
                    SD.data.latitude,
                    # c=SD.data.TEMP_SBE37_MEAN,
                    c=date2num(SD.data.time),
                    s=10,
                    cmap='RdBu_r',
                    label="Saildrone track")

# properties
ax03.coastlines()
ax03.stock_img()
ax03.set_title('TPOS Saildrone 1069')
gl = ax03.gridlines(draw_labels=True,
                    linewidth=1.5,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                    zorder=0)
gl.right_labels = False
ax03.set_extent([-162, -135, -4.5, 25.5])
ax03.text(-159, 17, 'Hawai\'i')
cbar = fig03.colorbar(sc03,
                      ticks=[date2num(np.datetime64('2019-07-01 00:00:00')),
                             date2num(np.datetime64('2019-08-01 00:00:00')),
                             date2num(np.datetime64('2019-09-01 00:00:00')),
                             date2num(np.datetime64('2019-10-01 00:00:00')),
                             date2num(np.datetime64('2019-11-01 00:00:00')),
                             date2num(np.datetime64('2019-12-01 00:00:00'))],
                      format=mticker.FixedFormatter(['July',
                                                     'August',
                                                     'September',
                                                     'October',
                                                     'November',
                                                     'December']),
                      # label='SST (degC)')
                      )
# ax03.legend(facecolor='none')

# plt.savefig('/Users/nicholasforcone/Library/CloudStorage/'
#             'GoogleDrive-nforcone@umich.edu/My Drive/'
#             'Gustiness Paper/Figures/saildrone_track_dates.png', dpi=350)

# plt.savefig('/Users/nicholasforcone/Library/CloudStorage/'
#             'GoogleDrive-nforcone@umich.edu/My Drive/'
#             'Gustiness Paper/Figures/saildrone_1069_track.png', dpi=350)

# %%
# find empirical beta
# wstar is the convective velocity
# gustiness is beta * wstar
fig05, ax05 = plt.subplots()

# preprocessing
default_ug = 0.2
beta = 1.2
wstar = ug / beta
arr = np.stack((wstar,
                np.sqrt(SD_missing_wind_var),
                SD.vector_60min_mean_wind_speed),
               axis=1)
arr_no_nan = arr[~np.isnan(arr).any(axis=1)]  # remove any rows with nans
not_default = arr_no_nan[:, 0] != (default_ug / beta)
arr_no_default = arr_no_nan[not_default, :]
# arr_no_default = arr_no_default[arr_no_default[:, 2] < 10]  # filtering
X = arr_no_default[:, 0].reshape(-1, 1)  # wstar
Y = arr_no_default[:, 1].reshape(-1, 1)  # (measure gustiness)^(1/2)
Z = arr_no_default[:, 2].reshape(-1, 1)  # hourly mean wind speed
X_cubed = X ** 3  # wstar^3
log_Y = np.log(Y, where=Y > 0)  # log(measured gustiness)
log_X_cubed = np.log(X_cubed)  # log(wstar^3)

# properties
ax05.set_xlabel(r'$\log(wstar^3)= \log(\frac{g}{T} \times \overline{w^\prime \theta_v^\prime} \times z_i)$',
                fontsize=15)
ax05.set_ylabel(r'$\log(measured\:gustiness)$', fontsize=15)
ax05.set_title(r'Log Transformation and Linear Regression', fontsize=15)

# interactive
point_collection = ax05.scatter(log_X_cubed, log_Y, s=1, c='black')
# snap_cursor_scat = SnappingCursorScat(ax05, point_collection, Z)
# fig05.canvas.mpl_connect('motion_notify_event',
#                          snap_cursor_scat.on_mouse_move)
# fig05.colorbar(point_collection, label='Hourly Mean Wind Speed')


reg = LinearRegression().fit(log_X_cubed, log_Y)  # fit a linear model
Y_pred = reg.predict(log_X_cubed)
ax05.plot(log_X_cubed, Y_pred, color='red')
ax05.text(-6.75, 0.75, f'$R^2$ = {reg.score(log_X_cubed, log_Y):.3f}', fontsize=13)
ax05.text(-6.75, 0.5, f'coef = {reg.coef_[0][0]:.3f}', fontsize=13)
ax05.text(-6.75, 0.25, r'$e^{intercept}$' f'= {np.e ** reg.intercept_[0]:.3f}', fontsize=13)
print('coefficitent: ', reg.coef_[0][0])
print('intercept: ', reg.intercept_[0])
print('Beta = e^intercept: ', np.e ** reg.intercept_[0])
print('$R^2$: ', reg.score(log_X_cubed, log_Y))
fig05.tight_layout()

# plt.savefig('/Users/nicholasforcone/GustinessRepo/FIGURES/log_transform_' + 
#             datetime.now().strftime("%Y_%m_%d"), dpi=350)

fig10, ax10 = plt.subplots()
ax10.scatter(SD_missing_wind_var, ug**2, s=1,
             label='old param $Ug^2$')
ax10.scatter(SD_missing_wind_var, ug_NEW**2, s=1,
             label='new param $Ug^2$')
ax10.set_xlabel('Observed $gustiness^2$')
ax10.set_ylabel('Parameterized $gustiness^2$')
one_to_one = np.arange(3)
ax10.plot(one_to_one, one_to_one, color='black')
ax10.legend()

# plt.savefig('/Users/nicholasforcone/GustinessRepo/FIGURES/obs_v_param_both' + 
#             datetime.now().strftime("%Y_%m_%d"), dpi=350)

fig11, ax11 = plt.subplots()
beta_fix = LinearRegression().fit(X, Y)
beta_fix_y_pred = beta_fix.predict(X)
ax11.scatter(X, Y, s=1, c='black')
ax11.plot(X, beta_fix_y_pred, color='red')

# %%
# investigate correlation between measured gustiness and gustiness parameter
discrepancy = SD_missing_wind_var - ug ** 2

X_vars = np.array([[SD.mean60min.TEMP_AIR_MEAN, 'Air T'],
                   [SD.mean60min.TEMP_AIR_MEAN - SD.mean60min.TEMP_SBE37_MEAN,
                    r'$Air\:T - SST$'],
                   [SD.mean60min.TEMP_SBE37_MEAN, 'SST'],
                   [SD.vector_60min_mean_wind_speed, 'Wind Speed'],
                   [SD.scalar_60min_mean_wind_speed, 'Scalar Wind Speed'],
                   [SD.mean60min.BARO_PRES_MEAN, 'P']], dtype='object')
fig06, ax06 = plt.subplots(len(X_vars), sharey=True)
ax06[2].set_ylabel(r"$SD\:missing\:wind\:variance - ug^2$")
fig06.tight_layout(h_pad=0.2)

for i, Xvar in enumerate(X_vars):
    arr00 = np.stack((Xvar[0], discrepancy), axis=1)

    # remove any rows with nans
    arr00_no_nan = arr00[~np.isnan(arr00).any(axis=1)]
    X00 = arr00_no_nan[:, 0].reshape(-1, 1)
    Y00 = arr00_no_nan[:, 1].reshape(-1, 1)  # discrepancy
    sc06_0 = ax06[i].scatter(X00,
                             Y00,
                             s=1,
                             color='black',
                             label=Xvar[1])
    reg00 = LinearRegression().fit(X00, Y00)
    Y_pred_00 = reg00.predict(X00)
    ax06[i].plot(X00, Y_pred_00, color='red')
    ax06[i].text(min(X00),
                 0.7 * max(Y00), f'$R^2$ = {reg00.score(X00, Y00):.3f}')
    ax06[i].legend(loc='upper right', fontsize='small')

# %%
# edges of averaging bins
# bins straddle center_time
bin_edge = (SD.center_time - pd.Timedelta(30, "minutes"))

# manually add in rightmost bin edge
bin_edge = bin_edge.insert(len(bin_edge), bin_edge[-1]
                           + pd.Timedelta(60, "minutes"))


def Subtract_The_Mean(group):
    return group - group.mean()


df = SD.data.to_dataframe()
df.index = df.time
resamp = df.resample('60min')['wind_speed'].apply(Subtract_The_Mean)
mean_again = (resamp ** 2).resample('60min').mean()

# %%
fig07, ax07 = plt.subplots()
csfont = {'fontname': 'Times New Roman'}
plt.rcParams["font.family"] = "Times New Roman"

ax07.scatter(date2num(SD.data.time),  # 1 minute
             SD.data.wind_speed,
             s=0.5,
             color='red',
             label='1-Minute Wind Speed')
ax07.plot(date2num(SD.center_time),  # 60 min mean
          SD.vector_60min_mean_wind_speed,
          linewidth=1,
          color='blue',
          label='Vector Mean Wind (Hourly)')
ax07.plot(date2num(SD.center_time),  # 60 min mean
          SD.scalar_60min_mean_wind_speed,
          linewidth=1,
          color='black',
          label='Scalar Mean Wind (Hourly)')

ax07.xaxis.set_major_locator(MonthLocator())
ax07.xaxis.set_minor_locator(HourLocator())
ax07.xaxis.set_major_formatter(DateFormatter('%b %d'))
ax07.xaxis.set_minor_formatter(DateFormatter('%H'))
ax07.set_xlim(np.datetime64('2019-11-18 21:00:00'),
              np.datetime64('2019-11-19 09:00:00'))
ax07.set_ylim(top=11)
ax07.legend()
ax07.set_title("Mean Wind Comparison SD1069", **csfont)

# plt.savefig('/Users/nicholasforcone/Library/CloudStorage/'
#             'GoogleDrive-nforcone@umich.edu/My Drive/'
#             'Gustiness Paper/Figures/fig04.png', dpi=350)

# %%
timeSlice = resamp['2019-10-20 00:00:00': '2019-12-28 00:00:00']
interp = timeSlice.interpolate().values

fig08, ax08 = plt.subplots()
y = fft(mean_again.interpolate().values)
xf = fftfreq(n=len(mean_again), d=1/60)

ax08.plot(xf, y)
