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


def feedback(
    vards,
    gmst,
):
    feed = vards.mean(["lat"])
    return feed
