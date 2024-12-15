# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT (https://tldrlegal.com/license/mit-license)
# Copyright (c) 2024, Paula Romero Jure
# Paper in preparation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy import stats

from scipy.stats import t

import xarray as xa

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Var names convention: varDescription_place_timeRange
# Time range eg: clim, monthlyclim, annual, interannual, enso

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions


def anomaly(ds):
    ds_anom = ds.groupby("time.month") - ds.groupby("time.month").mean("time")
    return ds_anom


def netCRE(lwds, swds, times=False, cc=None):
    """Calculates a net CRE kernel.
    If times, calculates Net CRE (or cloud-induced radiative anomaly R)

    Parameters:
    ----------
    lwds: xr.Dataset
        Dataset containing long-wave TOA rad values. Must be trimmed
    swds: xr.Dataset
        Dataset containing short-wave TOA rad values. Must be trimmed.
    times: bool
        Whether it will be timed by cloud area or fraction
    cc: xr.Dataset
        Cloud cover. The weights.

    Returns:
    -------
    Kernel (xr.Dataset)
    """
    lw_cre = lwds.toa_lw_clr_mon.mean("lon") - lwds.toa_lw_cldtyp_mon.mean("lon")
    sw_cre = swds.toa_sw_clr_mon.mean("lon") - swds.toa_sw_cldtyp_mon.mean("lon")
    K_trop = (lw_cre + sw_cre) / 100

    if times:
        cc_anom = anomaly(cc)
        return K_trop * cc_anom
    else:
        return K_trop

    def kernel(allsw, alllw):
        area_ep = alllw.cldarea_cldtyp_mon.mean("lon")

        RclrLW_ep = alllw.toa_lw_clr_mon.mean("lon")
        RovcLW_ep = alllw.toa_lw_cldtyp_mon.mean("lon")
        lwK_ep = (RclrLW_ep - RovcLW_ep) / 100

        RclrSW_ep = allsw.toa_sw_clr_mon.mean("lon")
        RovcSW_ep = allsw.toa_sw_cldtyp_mon.mean("lon")
        swK_ep = (RclrSW_ep - RovcSW_ep) / 100

        K_ep = lwK_ep + swK_ep
        return K_ep


class Feedbacks:

    tcrit = t.ppf(0.975, df=222 - 1)

    def __init__(self, area, k, gmst):
        self.area = area
        self.k = k
        self.gmst = gmst
        self.cc_anom = anomaly(self.area)
        self._cTot = self.area.sum(["press", "opt"])
        self._cTot_anom = self.cc_anom.sum(["press", "opt"])
        self._cc_ast = self.cc_anom - (self.area / self._cTot) * self._cTot_anom
        # aca area .sel(press=[4, 5, 6])

    def ctp(self):
        """Computes total feedback in CTP-tau object.

        Returns:
        -------
        feed_r, feed_std: xr.DataArray
            The feedback in pressure and optical depth bins (7x6).
            The standard error in pressure and optical depth bins (7x6).
        """
        R = self.k.mean("time") * self.cc_anom
        R_mean = R.mean(["lat"])  # weighted(weights=weights).

        feed = np.zeros([7, 6])
        feed_st = np.zeros([7, 6])

        for od in [0, 1, 2, 3, 4, 5]:
            for p in [0, 1, 2, 3, 4, 5, 6]:
                bints = R_mean.sel(press=p, opt=od)
                net_regress = stats.linregress(self.gmst, bints)
                feed[p, od] = net_regress.slope
                feed_st[p, od] = net_regress.stderr

        feed_r = xa.DataArray(
            feed,
            coords={
                "press": R_mean.press,
                "opt": R_mean.opt,
            },
        )

        feed_std = xa.DataArray(
            feed_st,
            coords={
                "press": R_mean.press,
                "opt": R_mean.opt,
            },
        )

        return feed_r, feed_std

    def total(self):
        total_feedback = self.sum(["press", "opt"])
        ci = self.sum(["press", "opt"])
        return total_feedback, ci

    # Decompositions
    # Aca necesito solo high clouds
    def amount(self, high_clouds=True):

        if high_clouds:
            hc = [4, 5, 6]
        else:
            hc = []

        K_0_hc = (((self.area.sel(press=hc) / self._cTot)) * self.k.sel(press=hc)).sum(
            ["opt", "press"]
        )
        Ramt_anom = K_0_hc * self._cTot_anom
        feed_amount = stats.linregress(self.gmst, Ramt_anom.mean("lat"))
        return feed_amount.slope, feed_amount.stderr

    def opticaldepth(self):
        # aca estaban cTot, cc_anom, cTot_anom, c_ast con press para hc
        ci_Tot = (self.area.sel(press=[4, 5, 6]) / self._cTot).sum("press")
        Ktau = self.k.sel(press=[4, 5, 6]) * ci_Tot.sum("opt")

        K_prima = (Ktau * ci_Tot).sum("opt")

        casi_R = K_prima * self.cc_ast.sum("opt")

        R_altitude = casi_R.sum("press")

        feedback_i = stats.linregress(self.gmst, R_altitude.mean("lat"))

        return feedback_i.slope, feedback_i.stderr

    def altitude(self):
        # aca estaban cTot, cc_anom, cTot_anom, c_ast con press para hc
        ci_Tot = (self.area.sel(press=[4, 5, 6]) / self._cTot).sum("opt")
        Ktau = self.k.sel(press=[4, 5, 6]) * ci_Tot.sum("press")

        K_prima = (Ktau * ci_Tot).sum("press")

        casi_R = K_prima * self.cc_ast.sum("press")

        R_altitude = casi_R.sum("opt")

        feedback_i = stats.linregress(self.gmst, R_altitude.mean("lat"))

        return feedback_i.slope, feedback_i.stderr

    def res(self):
        ct_Tot = (self.area.sel(press=[4, 5, 6]) / self._cTot).sum("press")
        Kp = self.k.sel(press=[4, 5, 6]) * ct_Tot.sum("opt")
        k_prima_press = (Kp * ct_Tot).sum("opt")

        cp_Tot = (self.area.sel(press=[4, 5, 6]) / self._cTot).sum("opt")
        Ktau = self.k.sel(press=[4, 5, 6]) * cp_Tot.sum("press")
        k_prima_tau = (Ktau * cp_Tot).sum("press")

        k_R = self.k - k_prima_press - k_prima_tau

        R_res = (k_R * self.cc_ast).sum(["press", "opt"])

        feed_res = stats.linregress(self.gmst, R_res.mean("lat"))
        return feed_res


def correlations(sst, var, plot=True):
    """Computes correlations for time series and plots both.
    lat, lon y times must be selected

    Parameters:
    ----------
    sst: xa
    La SST con lat y lon
    var:xa.DataSet
    EL dataset de la variable con todas las dimensiones"""
    # calcular anomalies para los dos
    odweight_itcz = var.mean(["opt", "press", "lat", "lon"])

    odweight_itcz_anom = odweight_itcz.groupby("time.month") - odweight_itcz.groupby(
        "time.month"
    ).mean("time")

    # plot
    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = "tab:red"
        (p1,) = odweight_itcz_anom.plot(
            linewidth=0.6,
            ax=ax1,
            color=color,
        )
        ax1.tick_params(
            axis="y",
            labelcolor=color,
        )

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = "tab:blue"
        (p2,) = sst.mean(["lat", "lon"]).plot(
            linewidth=0.6, ax=ax2, color=color
        )  # ax2.plot(t, data2, color=color)
        ax2.tick_params(
            axis="y",
            labelcolor=color,
        )

        plt.suptitle(
            "Thick high clouds OD and SST monthly anomalies - Lat=[0,20], Lon=[120,280]"
        )
        ax1.set_ylabel("OD")
        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        ax1.xaxis.set_major_locator(mdates.YearLocator(base=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax1.grid(visible=True, axis="x")

        plt.show()

    return xa.corr(sst, odweight_itcz_anom)
