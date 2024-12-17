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

import pandas as pd

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
    """Calculates a kernel"""

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

    def __repr__(self):
        return r"$\lambda$"

    def ctp(self):
        """Computes total feedback in CTP-tau object. Returns xarray DataArray.
        Still doesnt work only for hc

        Returns:
        -------
        feed_r, feed_std: xr.DataArray
            The feedback in pressure and optical depth bins (7x6).
            The standard error in pressure and optical depth bins (7x6).
        """
        R = self.k.mean("time") * self.cc_anom
        R_mean = R.mean(["lat"])  # weighted(weights=weights).

        optdim = [int(i) for i in R_mean.opt.data.tolist()]
        pressdim = [int(i) for i in R_mean.press.data.tolist()]

        feed = np.zeros([len(pressdim), len(optdim)])
        feed_st = np.zeros([len(pressdim), len(optdim)])

        for od in optdim:
            for p in pressdim:
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

        feed_stderr = xa.DataArray(
            feed_st,
            coords={
                "press": R_mean.press,
                "opt": R_mean.opt,
            },
        )

        total_feedback = feed_r.sum(["press", "opt"]).data
        ci = self.tcrit * feed_stderr.sum(["press", "opt"]).data

        print(f"Total feedback = {total_feedback} +- {ci}")
        return feed_r, feed_stderr

    def total(self, ctpobj):
        """Calculates total feedback of a CTP-tau object with its confidence interval.

        Parameters:
        ----------
        crp: xr.DataArray
            ctp feedback, with dimentions "press" and "opt"
        Returns:
        -------
        total_feedback, ci: float, float
            Total feedback wit CI = t_student * std error
        """
        total_feedback = ctpobj[0].sum(["press", "opt"]).data
        ci = self.tcrit * ctpobj[1].sum(["press", "opt"]).data
        print(f"Total feedback = {total_feedback} +- {ci}")
        return total_feedback, ci

    # Decompositions
    # Aca necesito solo high clouds
    def amount(self, high_clouds=True):

        hc = [4, 5, 6]
        self.area = self.area.sel(press=hc)
        self.k = self.k.sel(press=hc)

        if high_clouds:
            hc = [4, 5, 6]
        else:
            hc = []

        K_0_hc = (((self.area.sel(press=hc) / self._cTot)) * self.k.sel(press=hc)).sum(
            ["opt", "press"]
        )
        Ramt_anom = K_0_hc * self._cTot_anom
        feed_amount = stats.linregress(self.gmst, Ramt_anom.mean("lat"))
        return feed_amount.slope, self.tcrit * feed_amount.stderr

    def altitude(self):
        # aca estaban cTot, cc_anom, cTot_anom, c_ast con press para hc
        ci_Tot = (self.area.sel(press=[4, 5, 6]) / self._cTot).sum("press")
        Ktau = self.k.sel(press=[4, 5, 6]) * ci_Tot.sum("opt")

        K_prima = (Ktau * ci_Tot).sum("opt")

        casi_R = K_prima * self._cc_ast.sum("opt")

        R_altitude = casi_R.sum("press")

        feedback_i = stats.linregress(self.gmst, R_altitude.mean("lat"))

        return feedback_i.slope, self.tcrit * feedback_i.stderr

    def opticaldepth(self):
        # aca estaban cTot, cc_anom, cTot_anom, c_ast con press para hc
        ci_Tot = (self.area.sel(press=[4, 5, 6]) / self._cTot).sum("opt")
        Ktau = self.k.sel(press=[4, 5, 6]) * ci_Tot.sum("press")

        K_prima = (Ktau * ci_Tot).sum("press")

        casi_R = K_prima * self._cc_ast.sum("press")

        R_altitude = casi_R.sum("opt")

        feedback_i = stats.linregress(self.gmst, R_altitude.mean("lat"))

        return feedback_i.slope, self.tcrit * feedback_i.stderr

    def res(self):
        ct_Tot = (self.area.sel(press=[4, 5, 6]) / self._cTot).sum("press")
        Kp = self.k.sel(press=[4, 5, 6]) * ct_Tot.sum("opt")
        k_prima_press = (Kp * ct_Tot).sum("opt")

        cp_Tot = (self.area.sel(press=[4, 5, 6]) / self._cTot).sum("opt")
        Ktau = self.k.sel(press=[4, 5, 6]) * cp_Tot.sum("press")
        k_prima_tau = (Ktau * cp_Tot).sum("press")

        k_R = self.k - k_prima_press - k_prima_tau

        R_res = (k_R * self._cc_ast).sum(["press", "opt"])

        feed_res = stats.linregress(self.gmst, R_res.mean("lat"))
        return feed_res.slope, self.tcrit * feed_res.stderr


def plot(
    hclw,
    hcsw,
    hcnet,
    area="ITCZ",
    totallw=(-0.347357, 0.3275278),
    totalsw=(0.43392, 0.31945),
    totalnet=(0.0519, 0.0642),
    **kwargs,
):
    """Plots bar plot given the LW, Sw and Net feedback objects.

    Parameters:
    ----------
    sw_obj: Feedback
        Short-wave feedback object.
    lw_obj: Feedback
        Long-wave feedback object.
    net_obj: Feedback
       Net feedback object.

    Returns:
    -------
    None. Or pandas summary.
    """

    decompos = ("Total", "Amount", "Altitude", "Optical Depth", "Residual")

    sdc = {
        # "lw total": hclw.total(hclw.ctp()),
        "lw": [
            totallw,
            hclw.amount(),
            hclw.altitude(),
            hclw.opticaldepth(),
            hclw.res(),
        ],
        # "sw total": hcsw.total(),
        "sw": [
            totalsw,
            hcsw.amount(),
            hcsw.altitude(),
            hcsw.opticaldepth(),
            hcsw.res(),
        ],
        # "net total": hcnet.total(),
        "net": [
            totalnet,
            hcnet.amount(),
            hcnet.altitude(),
            hcnet.opticaldepth(),
            hcnet.res(),
        ],
    }

    feeds = {
        "LW": [sdc["lw"][i][0] for i in range(5)],
        "Net": [sdc["net"][i][0] for i in range(5)],
        "SW": [sdc["sw"][i][0] for i in range(5)],
    }

    ci_upper = {
        "LW": feeds["LW"] + np.array([sdc["lw"][i][1] for i in range(5)]),
        "Net": feeds["Net"] + np.array([sdc["net"][i][1] for i in range(5)]),
        "SW": feeds["SW"] + np.array([sdc["sw"][i][1] for i in range(5)]),
    }
    ci_lower = {
        "LW": feeds["LW"] - np.array([sdc["lw"][i][1] for i in range(5)]),
        "Net": feeds["Net"] - np.array([sdc["net"][i][1] for i in range(5)]),
        "SW": feeds["SW"] - np.array([sdc["sw"][i][1] for i in range(5)]),
    }

    colors = {"LW": "red", "Net": "black", "SW": "blue"}

    x = np.arange(len(decompos))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))

    for attribute, measurement in feeds.items():
        offset = width * multiplier

        # Calculate asymmetric error bars
        lower_errors = [m - ci for m, ci in zip(measurement, ci_lower[attribute])]
        upper_errors = [ci - m for m, ci in zip(measurement, ci_upper[attribute])]
        yerr_values = (lower_errors, upper_errors)

        rects = ax.bar(
            x + offset,
            measurement,
            width,
            label=attribute,
            yerr=yerr_values,
            capsize=5,
            color=colors[attribute],
            **kwargs,
        )

        for i, rect in enumerate(rects):
            lower_bound = ci_lower[attribute][i]
            upper_bound = ci_upper[attribute][i]

            if (
                lower_bound <= 0 <= upper_bound
            ):  # Not significant # Only apply to LW bars
                rect.set_facecolor("none")  # Remove filling for the first bar
                rect.set_edgecolor("red")

        ax.bar_label(rects, labels=[f"{value:.2f}" for value in measurement], padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r"Feedbacks ($Wm^{-2}K^{-1}$)")
    ax.set_title(f"{area} high cloud feedbacks")
    ax.legend(loc="upper right")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Total", "Amount", "Altitude", "Optical Depth", "Residual"])
    ax.axhline(0, color="black", linewidth=1)
    if "ylim" in kwargs:
        ax.set_ylim(kwargs["ylim"])

    plt.show()

    return None


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


def summary(hclw, hcsw, hcnet, totallw, totalsw, totalnet):
    rows = ["Total", "Amount", "Altitude", "Optical Depth", "Residual"]

    sdc = {
        # "lw total": hclw.total(hclw.ctp()),
        "lw": [
            totallw,
            hclw.amount(),
            hclw.altitude(),
            hclw.opticaldepth(),
            hclw.res(),
        ],
        # "sw total": hcsw.total(),
        "sw": [
            totalsw,
            hcsw.amount(),
            hcsw.altitude(),
            hcsw.opticaldepth(),
            hcsw.res(),
        ],
        # "net total": hcnet.total(),
        "net": [
            totalnet,
            hcnet.amount(),
            hcnet.altitude(),
            hcnet.opticaldepth(),
            hcnet.res(),
        ],
    }

    sumdf = pd.DataFrame.from_dict(sdc, orient="columns").rename(
        index={0: rows[0], 1: rows[1], 2: rows[2], 3: rows[3], 4: rows[4]}
    )
    sumdf
    for col in sumdf.columns:
        sumdf[[f"{col}_mean", f"{col}_ci"]] = sumdf[col].apply(pd.Series)

    sumdf.drop(columns=["lw", "sw", "net"], inplace=True)
    sumdf.columns
    for col in sumdf.columns:
        sumdf[col] = sumdf[col].map(lambda x: f"{x:.4f}")

    return sumdf
