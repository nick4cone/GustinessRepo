#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:25:39 2024

@author: nforcone@umich.edu
"""
import xarray as xa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter, HourLocator, MonthLocator
from COARE.coare36vn_zrf_et import coare36vn_zrf_et  # from email thread
from COARE.coare_no_ug_param import coare_no_ug_param
from sklearn.linear_model import LinearRegression


# %%
class SaildroneHurr:
    """ Stores Saildrone Data
    """

    def __init__(self, name, filepath):
        self.name = name
        self.filePath = filepath
        self.data = xa.open_dataset(filepath)

    def show_metadata(self):
        print(self.data)
        print(self.data.data_vars)

    def hourly_binned_mean(self):

        # calculate u and v
        md = 270 - self.data.WIND_FROM_MEAN
        md[md < 0] += 360
        u = self.data.WIND_SPEED_MEAN * np.cos((np.pi / 180) * md)
        v = self.data.WIND_SPEED_MEAN * np.sin((np.pi / 180) * md)
        self.data = self.data.assign(UWND_MEAN=u)
        self.data = self.data.assign(VWND_MEAN=v)

        # ocean current u and v
        md2 = 270 - self.data.WATER_CURRENT_DIRECTION_MEAN
        md2[md2 < 0] += 360
        ucurr = self.data.WATER_CURRENT_SPEED_MEAN * np.cos((np.pi / 180)
                                                            * md2)
        vcurr = self.data.WATER_CURRENT_SPEED_MEAN * np.sin((np.pi / 180)
                                                            * md2)
        relu = u - ucurr
        relv = v - vcurr
        self.data = self.data.assign(RELU=relu)
        self.data = self.data.assign(RELV=relv)

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

        self.scalar_60min_mean_wind_spd = self.mean60min.WIND_SPEED_MEAN.values
        self.vector_60min_mean_wind_spd = np.sqrt(
            self.mean60min.UWND_MEAN.values ** 2
            + self.mean60min.VWND_MEAN.values ** 2)


# %%
ROOT = "/Users/nicholasforcone/GustinessData/"
FILE = "sd1045_hurricane_2021_4a74_3977_50f0.nc"
SD1045 = SaildroneHurr("1045", ROOT + FILE)

# binned hourly means
SD1045.hourly_binned_mean()

# %%
# visualize and verify binning algorithm
# time bin left edges
left = []
bins = SD1045.mean60min.time_bins.values
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
ax00.vlines(date2num(SD1045.center_time[0:n]),
            ymin=-ymin,
            ymax=ymax,
            linestyle='--',
            color='black')
ax00.plot(date2num(SD1045.data.time)[0:n*60],  # raw data
          SD1045.data.TEMP_AIR_MEAN[0:n*60],
          marker='^',
          markerfacecolor='red',
          markeredgecolor='red',
          markersize=2.5,
          color='red')
ax00.plot(date2num(SD1045.center_time)[0:n],  # 60 min mean
          SD1045.mean60min.TEMP_AIR_MEAN[0:n],
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
    u=SD1045.scalar_60min_mean_wind_spd,  # scalar mean wind speed
    zu=5.2,
    t=SD1045.mean60min.TEMP_AIR_MEAN.values,
    zt=2.3,
    rh=SD1045.mean60min.RH_MEAN.values,
    zq=2.3,
    P=np.nan_to_num(SD1045.mean60min.BARO_PRES_MEAN.values, nan=1015),
    ts=SD1045.mean60min.TEMP_SBE37_MEAN.values,
    sw_dn=np.full(len(SD1045.center_time), 0),
    lw_dn=np.full(len(SD1045.center_time), 0),
    lat=SD1045.mean60min.latitude.values,
    lon=SD1045.mean60min.longitude.values,
    jd=SD1045.center_time.to_julian_date().to_numpy(),
    zi=600.0,
    rain=np.full(len(SD1045.center_time), np.nan),  # nan array for rain
    Ss=SD1045.mean60min.SAL_SBE37_MEAN.values)

# hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
# hlb = latent heat flux (W/m^2) ... positive for qair < qs
sensible = coare_out[:, 2]
latent = coare_out[:, 3]
thflx = sensible + latent  # positive cools the ocean (heats the atmosphere)

# run coare with parameterized gustiness ug
coare_out_p = coare36vn_zrf_et(
    u=SD1045.vector_60min_mean_wind_spd,  # vector mean wind speed
    zu=5.2,
    t=SD1045.mean60min.TEMP_AIR_MEAN.values,
    zt=2.3,
    rh=SD1045.mean60min.RH_MEAN.values,
    zq=2.3,
    P=np.nan_to_num(SD1045.mean60min.BARO_PRES_MEAN.values, nan=1015),
    ts=SD1045.mean60min.TEMP_SBE37_MEAN.values,
    sw_dn=np.full(len(SD1045.center_time), 0),
    lw_dn=np.full(len(SD1045.center_time), 0),
    lat=SD1045.mean60min.latitude.values,
    lon=SD1045.mean60min.longitude.values,
    jd=SD1045.center_time.to_julian_date().to_numpy(),
    zi=600.0,
    rain=np.full(len(SD1045.center_time), np.nan),  # nan array for rain
    Ss=SD1045.mean60min.SAL_SBE37_MEAN.values)

# gustiness parameter from COARE
ug = coare_out_p[:, 47]
SD1045_missing_wind_var = (SD1045.scalar_60min_mean_wind_spd ** 2 -
                           SD1045.vector_60min_mean_wind_spd ** 2)

# hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
# hlb = latent heat flux (W/m^2) ... positive for qair < qs
sensible_p = coare_out_p[:, 2]
latent_p = coare_out_p[:, 3]
thflx_p = sensible_p + latent_p  # positive cools the ocean

# %%
# measured missing wind variance and ug^2
fig01, ax01 = plt.subplots()

# missing wind variance colored by SST
sc01 = ax01.scatter(date2num(SD1045.center_time),
                    SD1045_missing_wind_var,
                    c=SD1045.mean60min.TEMP_SBE37_MEAN.values,
                    s=1,
                    cmap='bwr',
                    linewidth=0.7,
                    label=r"$\langle U^{2} + V^{2} \rangle"
                    r"- (\langle U \rangle^2 + \langle V \rangle^2)"
                    r"\: \langle\rangle_{60 \, min}$")  # raw string

# ug^2
ax01.plot(date2num(SD1045.center_time),
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
# investigate correlation between measured gustiness and gustiness parameter
discrepancy = SD1045_missing_wind_var - ug ** 2

X_vars = np.array([[SD1045.mean60min.TEMP_AIR_MEAN, 'Air T'],
                   [SD1045.mean60min.TEMP_AIR_MEAN -
                    SD1045.mean60min.TEMP_SBE37_MEAN, r'$Air\:T - SST$'],
                   [SD1045.mean60min.TEMP_SBE37_MEAN, 'SST'],
                   [SD1045.vector_60min_mean_wind_spd, 'Wind Speed'],
                   [SD1045.mean60min.BARO_PRES_MEAN, 'P']], dtype='object')
fig06, ax06 = plt.subplots(len(X_vars), sharey=True)
ax06[2].set_ylabel(r"$SD1045\:missing\:wind\:variance - ug^2$")
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
