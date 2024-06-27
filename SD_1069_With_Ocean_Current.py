#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:21:13 2024

@author: nforcone@umich.edu
"""
import numpy as np
import xarray as xa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter
from matplotlib.dates import HourLocator, MonthLocator
from COARE.coare36vn_zrf_et import coare36vn_zrf_et  # from email thread
from COARE.coare_no_ug_param import coare_no_ug_param


# %%
class SaildroneADCP:
    """ Stores Saildrone Data
    """

    def __init__(self, name, filepath, adcp_path):
        self.name = name
        self.filePath = filepath
        self.data = xa.open_dataset(filepath)
        self.adcp = xa.open_dataset(adcp_path)
        self.u_adcp = self.adcp.UADCP.sel(DEPTH_ADCP=4.24)
        self.v_adcp = self.adcp.VADCP.sel(DEPTH_ADCP=4.24)
        self.df1 = self.data.to_dataframe()
        self.df2 = self.u_adcp.to_dataframe().reset_index().drop(
            columns='TRAJECTORY').rename(columns={'MYNEWT': 'time'})
        self.df2['UADCP'] = self.df2['UADCP'].divide(100)
        self.df3 = self.v_adcp.to_dataframe().reset_index().drop(
            columns=['TRAJECTORY', 'DEPTH_ADCP']).rename(
                columns={'MYNEWT': 'time'})
        self.df3['VADCP'] = self.df3['VADCP'].divide(100)
        self.df4 = pd.merge(self.df1, self.df2, on='time', how='left')
        self.ds = pd.merge(self.df4,
                           self.df3, on='time', how='left').to_xarray()

    def show_metadata(self):
        print(self.data)
        print("\nData variables: ", self.data.data_vars)

    def hourly_binned_mean(self):
        start_date = self.ds.time.values[0]  # first timestamp
        end_date = self.ds.time.values[-1]  # last timestamp

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
        self.mean60min = self.ds.groupby_bins(
            self.ds.time, bin_edge).mean()

        self.scalar_60min_mean_wind_speed = self.mean60min.wind_speed.values
        self.vector_60min_mean_wind_speed = np.sqrt(
            self.mean60min.UWND_MEAN.values ** 2
            + self.mean60min.VWND_MEAN.values ** 2)

    def ten_min_binned_mean(self):
        start_date = self.ds.time.values[0]  # first timestamp
        end_date = self.ds.time.values[-1]  # last timestamp

        # Return a fixed frequency datetime index
        self.center_time_10min = xa.date_range(start_date,
                                               end_date, freq="10 min")

        # edges of averaging bins
        # bins straddle center_time
        bin_edge = (self.center_time_10min - pd.Timedelta(5, "minutes"))

        # manually add in rightmost bin edge
        bin_edge = bin_edge.insert(len(bin_edge), bin_edge[-1]
                                   + pd.Timedelta(10, "minutes"))

        # splits data into bins
        # computes the mean of each bin
        # groups are combined back into a single data object
        self.mean10min = self.ds.groupby_bins(
            self.ds.time, bin_edge).mean()

        self.scalar_10min_mean_wind_speed = self.mean10min.wind_speed.values
        self.vector_10min_mean_wind_speed = np.sqrt(
            self.mean10min.UWND_MEAN.values ** 2
            + self.mean10min.VWND_MEAN.values ** 2)


# %%
ROOT = "/Users/nicholasforcone/Library/Mobile Documents/com~apple~CloudDocs/GustinessData/"
FILE = "sd1069_2019_c297_1b1e_6c16.nc"
FILE_ADCP = "SD1069adcp2019.cdf"
SD = SaildroneADCP("1069", ROOT + FILE, ROOT + FILE_ADCP)
SD.hourly_binned_mean()  # kind of slow

# %%
# visualize and verify binning algorithm
# time bin left edges
left = []
bins = SD.mean60min.time_bins.values
for t in range(len(bins)):
    left.append(bins[t].left)
np.asarray(left)

fig00, ax00 = plt.subplots()
ax00twin = ax00.twinx()
n = 10  # number of samples to plot
ymin = -10
ymax = 60
ax00.vlines(date2num(left[0:n]),
            ymin=ymin,
            ymax=ymax,
            color='darkorchid',
            alpha=0.8,
            label='Bin Edges')
ax00.vlines(date2num(SD.center_time[0:n]),
            ymin=ymin,
            ymax=ymax,
            linestyle='--',
            color='black',
            label='bin centers')
ax00.scatter(date2num(SD.ds.time)[0:n*60],  # wind speed raw data
             SD.ds.wind_speed[0:n*60],
             marker='^',
             s=2.5,
             color='tan')
ax00.scatter(date2num(SD.ds.time)[0:n*60],  # V wind
             SD.ds.VWND_MEAN[0:n*60],
             marker='^',
             s=2.5,
             color='red')
ax00twin.scatter(date2num(SD.ds.time)[0:n*60],  # UADCP raw data
                 SD.ds.UADCP[0:n*60],
                 marker='^',
                 s=2.5,
                 color='blue')
ax00.plot(date2num(SD.center_time)[0:n],  # wind speed 60 min mean
          SD.mean60min.wind_speed[0:n],
          marker='^',
          markerfacecolor='orange',
          markeredgecolor='orange',
          markersize=5,
          color='orange',
          label='HR Mean Wind Speed')
ax00.plot(date2num(SD.center_time)[0:n],  # V wind 60 min mean
          SD.mean60min.VWND_MEAN[0:n],
          marker='^',
          markerfacecolor='lightgreen',
          markeredgecolor='lightgreen',
          markersize=5,
          color='lightgreen',
          label='HR Mean V Wind')
ax00twin.plot(date2num(SD.center_time)[0:n],  # UADCP 60 min mean
              SD.mean60min.UADCP[0:n],
              marker='^',
              markerfacecolor='steelblue',
              markeredgecolor='steelblue',
              markersize=5,
              color='steelblue',
              label='HR Mean U Ocean')
ax00.xaxis.set_major_locator(HourLocator())
ax00.xaxis.set_major_formatter(DateFormatter('%H:%M'))
fig00.legend(fontsize='medium', ncol=4)
ax00.set_ylabel('$m/s$')
ax00twin.set_ylabel('$m/s$')

# %%
SD.ten_min_binned_mean()  # really slow

# %%
fig01, ax01 = plt.subplots()

ax01.scatter(date2num(SD.ds.time)[0:n*60],  # UADCP raw data
             SD.ds.UADCP[0:n*60],
             marker='^',
             s=2.5,
             color='blue')
ax01.plot(date2num(SD.center_time_10min)[0:n*6],  # UADCP 10 min mean
          SD.mean10min.UADCP[0:n*6],
          marker='^',
          markerfacecolor='steelblue',
          markeredgecolor='steelblue',
          markersize=5,
          color='steelblue',
          label='HR Mean U Ocean')
ax01.plot(date2num(SD.center_time)[0:n],  # UADCP 60 min mean
          SD.mean60min.UADCP[0:n],
          marker='^',
          markerfacecolor='orange',
          markeredgecolor='orange',
          markersize=5,
          color='orange',
          label='HR Mean U Ocean')
ax01.xaxis.set_major_locator(HourLocator())
ax01.xaxis.set_major_formatter(DateFormatter('%H:%M'))

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
    Ss=SD.mean60min.SAL_SBE37_MEAN.values)

# hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
# hlb = latent heat flux (W/m^2) ... positive for qair < qs
sensible = coare_out[:, 2]
latent = coare_out[:, 3]
thflx = sensible + latent  # positive cools the ocean (heats the atmosphere)

# U_HR - U_Ocean_HR
U_REL = SD.mean60min.UWND_MEAN.values - SD.mean60min.UADCP.values

# V_HR - V_Ocean_HR
V_REL = SD.mean60min.VWND_MEAN.values - SD.mean60min.VADCP.values

# ocean relative vector mean wind speed
vector_60min_ocean_relative = np.sqrt(U_REL ** 2 + V_REL ** 2)

# run coare with parameterized gustiness ug
coare_out_p = coare36vn_zrf_et(
    u=SD.vector_60min_mean_wind_speed,
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
    Ss=SD.mean60min.SAL_SBE37_MEAN.values)

# gustiness parameter from COARE
ug = coare_out_p[:, 47]
SD_missing_wind_var = (SD.scalar_60min_mean_wind_speed ** 2 -
                       SD.vector_60min_mean_wind_speed ** 2)

# hsb = sensible heat flux (W/m^2) ... positive for Tair < Tskin
# hlb = latent heat flux (W/m^2) ... positive for qair < qs
sensible_p = coare_out_p[:, 2]
latent_p = coare_out_p[:, 3]
thflx_p = sensible_p + latent_p  # positive cools the ocean

# %%
# measured missing wind variance and ug^2
fig02, ax02 = plt.subplots()

# missing wind variance colored by SST
sc02 = ax02.scatter(date2num(SD.center_time),
                    SD_missing_wind_var,
                    c=SD.mean60min.TEMP_SBE37_MEAN.values,
                    s=1.25,
                    cmap='coolwarm',
                    linewidth=0.7,
                    label=r"Measured Gustiness: $\langle U^{2} + V^{2} \rangle"
                    r"- (\langle U \rangle^2 + \langle V \rangle^2)"
                    r"\: \langle\rangle_{60 \, min}$")  # raw string

# ug^2
ax02.plot(date2num(SD.center_time),
          ug**2,
          color="black",
          linewidth=0.7,
          label="Parameterized Gustiness ($Ug^2$ from COARE)")

# properties
ax02.xaxis.set_major_locator(MonthLocator())
ax02.xaxis.set_major_formatter(DateFormatter('%b'))
fig02.colorbar(sc02, label='SST (degC)')
ax02.set_ylabel("$m^2 / s^2$")
ax02.legend(loc='upper left', fontsize='small')
ax02.set_title('Measured & Parameterized Wind Gustiness')
