# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

from scipy import stats

import xarray as xa

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Var names convention
# varDescription_place_timeRange
# Time range eg: clim, monthlyclim, annual, interannual, enso
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def anomaly(ds):
    ds_anom = ds.groupby("time.month") - ds.groupby("time.month").mean("time")
    return ds_anom


def netCRE(lwds, swds, times=False, cc=None):
    """Calculates a net CRE kernel.
    If times, calculates Net CRE (or R)
    Arguments:
    ----------
    lwds: xr.Dataset
        Dataset containing long-wave TOA rad values. Must be trimmed
    swds: xr.Dataset
        Dataset containing short-wave TOA rad values. Must be trimmed.
    space_time: list
        list containing selected
    """
    lw_cre = lwds.toa_lw_clr_mon.mean("lon") - lwds.toa_lw_cldtyp_mon.mean("lon")
    sw_cre = swds.toa_sw_clr_mon.mean("lon") - swds.toa_sw_cldtyp_mon.mean("lon")
    K_trop = (lw_cre + sw_cre) / 100

    if times:
        cc_anom = anomaly(cc)
        R = K_trop * cc_anom
    else:
        return K_trop


def feedback_amount(area, K, sst_anom):
    cTot = area.sum(["press", "opt"])
    cc_anom = area.groupby("time.month") - area.groupby("time.month").mean("time")
    cTot_anom = cc_anom.sum(["press", "opt"])

    K_0_hc = (((area / cTot)) * K).sum(["opt", "press"])
    Ramt_anom = K_0_hc * cTot_anom
    return stats.linregress(sst_anom, Ramt_anom.mean("lat")).slope


def hc_feedbacks(area, K, sst_anom, which):

    i = ["press", "opt"]
    if which not in i:
        raise NameError(f"Decompsitions only in {i} dims")

    opp = i[i.index(which) - 1]

    cTot = area.sel(press=[4, 5, 6]).sum(["press", "opt"])
    cc_anom = area.sel(press=[4, 5, 6]).groupby("time.month") - area.sel(
        press=[4, 5, 6]
    ).groupby("time.month").mean("time")
    cTot_anom = cc_anom.sel(press=[4, 5, 6]).sum(["press", "opt"])

    cc_ast = cc_anom - (area.sel(press=[4, 5, 6]) / cTot) * cTot_anom

    ci_Tot = (area.sel(press=[4, 5, 6]) / cTot).sum(which)
    Ktau = K.sel(press=[4, 5, 6]) * ci_Tot.sum(opp)

    K_prima = (Ktau * ci_Tot).sum(opp)

    casi_R = K_prima * cc_ast.sum(opp)

    R_altitude = casi_R.sum(which)

    feedback_i = stats.linregress(sst_anom, R_altitude.mean("lat")).slope

    return feedback_i


def correlations(sst, var):
    """Calcula correlaciones para time series y plotea las dos juntas.
    lat, lon y times must be selected
    INputs
    -------
    sst: xa
    La SST con lat y lon
    var:xa.DataSet
    EL dataset de la variable con todas las dimensiones"""
    # calcular anomalies para los dos
    odweight_itcz = (
        odtropics.sel(opt=[4, 5], lat=slice(0, 20))
        .weighted(area_tropics.sel(opt=[4, 5], lat=slice(0, 20)))
        .mean(["opt", "press", "lat", "lon"])
    )
    odweight_itcz_anom = odweight_itcz.groupby("time.month") - odweight_itcz.groupby(
        "time.month"
    ).mean("time")

    # plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:red"
    (p1,) = odweight_itcz_anom.plot(
        linewidth=0.6,
        ax=ax1,
        color=color,
    )
    ax1.tick_params(axis="y", labelcolor=color, **tkw)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = "tab:blue"
    (p2,) = (
        sst_anom_latlon.sel(lat=slice(20.5, -0.5), lon=ep)
        .mean(["lat", "lon"])
        .plot(linewidth=0.6, ax=ax2, color=color)
    )  # ax2.plot(t, data2, color=color)
    ax2.tick_params(axis="y", labelcolor=color, **tkw)

    plt.suptitle(
        "Thick high clouds OD and SST monthly anomalies - Lat=[0,20], Lon=[120,280]"
    )
    ax1.set_ylabel("OD")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    ax1.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.grid(visible=True, axis="x")

    plt.show()
    return corr
