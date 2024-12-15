# !/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT (https://tldrlegal.com/license/mit-license)
# Copyright (c) 2024, Paula Romero Jure
# Paper in preparation

import numpy as np

import pandas as pd

import xarray as xa

import methods

# Opening data

all0205 = xa.open_dataset(
    "../daa/atropics/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_Subset_200207-200505.nc",
    engine="netcdf4",
)
all0508 = xa.open_dataset(
    "../daa/atropics/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_Subset_200506-200804.nc",
    engine="netcdf4",
)
all0811 = xa.open_dataset(
    "../daa/atropics/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_Subset_200805-201103.nc",
    engine="netcdf4",
)
all1114 = xa.open_dataset(
    "../daa/atropics/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_Subset_201104-201402.nc",
    engine="netcdf4",
)
all1417 = xa.open_dataset(
    "../daa/atropics/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_Subset_201403-201701.nc",
    engine="netcdf4",
)
all1821 = xa.open_dataset(
    "../daa/atropics/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_Subset_201702-201912.nc",
    engine="netcdf4",
)
all2123 = xa.open_dataset(
    "../daa/atropics/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_Subset_202001-202211.nc",
    engine="netcdf4",
)
all2223 = xa.open_dataset(
    "../daa/atropics/CERES_FluxByCldTyp-MON_Terra-Aqua-MODIS_Ed4.1_Subset_202212-202302.nc",
    engine="netcdf4",
)

var = "toa_lw"
alllw = xa.concat(
    [
        all0205[["cldarea_cldtyp_mon", f"{var}_cldtyp_mon", "toa_lw_clr_mon"]],
        all0508[["cldarea_cldtyp_mon", f"{var}_cldtyp_mon", "toa_lw_clr_mon"]],
        all0811[["cldarea_cldtyp_mon", f"{var}_cldtyp_mon", "toa_lw_clr_mon"]],
        all1114[["cldarea_cldtyp_mon", f"{var}_cldtyp_mon", "toa_lw_clr_mon"]],
        all1417[["cldarea_cldtyp_mon", f"{var}_cldtyp_mon", "toa_lw_clr_mon"]],
        all1821[["cldarea_cldtyp_mon", f"{var}_cldtyp_mon", "toa_lw_clr_mon"]],
        all2123[["cldarea_cldtyp_mon", f"{var}_cldtyp_mon", "toa_lw_clr_mon"]],
    ],
    dim="time",
)

var2 = "toa_sw"
allsw = xa.concat(
    [
        all0205[["cldarea_cldtyp_mon", f"{var2}_cldtyp_mon", "toa_sw_clr_mon"]],
        all0508[["cldarea_cldtyp_mon", f"{var2}_cldtyp_mon", "toa_sw_clr_mon"]],
        all0811[["cldarea_cldtyp_mon", f"{var2}_cldtyp_mon", "toa_sw_clr_mon"]],
        all1114[["cldarea_cldtyp_mon", f"{var2}_cldtyp_mon", "toa_sw_clr_mon"]],
        all1417[["cldarea_cldtyp_mon", f"{var2}_cldtyp_mon", "toa_sw_clr_mon"]],
        all1821[["cldarea_cldtyp_mon", f"{var2}_cldtyp_mon", "toa_sw_clr_mon"]],
        all2123[["cldarea_cldtyp_mon", f"{var2}_cldtyp_mon", "toa_sw_clr_mon"]],
    ],
    dim="time",
)

# Gistemp GMST anomalies
gmst1850 = pd.read_csv("../daa/GLB.Ts+dSST.csv", skiprows=[0])

gmst = gmst1850.loc[gmst1850["Year"] >= 2002]
gmst = gmst.loc[gmst["Year"] < 2021]
lista_gmst = (
    gmst[
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    ]
    .values.flatten()
    .tolist()
)
# lista_gmst, len(lista_gmst)
lista_gmst[6:]
# NOT the same as meaqning across all the values, which gives:0.7340974 , 0.23327379, 0.95729756, 0.8614268 , 0.7519227...
gmst_anom_raghu = np.array(lista_gmst[6:], dtype=np.float16)

# Defining time and area slices
trop = slice(-30, 30)
ragutime = slice("2002-07-15T00:00:00.000000000", "2020-12-15T00:00:00.000000000")
itcz = slice(-0.5, 20.5)  # slice(5,15)
trop = slice(-30, 30)
wp = slice(120, 160)
cp = slice(160, 200)
ep = slice(210, 260)

# Computing kernels
net_k = methods.kernel(
    allsw.sel(lat=trop, time=ragutime), alllw.sel(lat=trop, time=ragutime)
)

area_trop = alllw.cldarea_cldtyp_mon.sel(lat=trop, time=ragutime).mean("lon")

objfeed = methods.Feedbacks(area=area_trop, k=net_k, gmst=gmst_anom_raghu)

print(objfeed)

print(objfeed.ctp)

# objfeed.total(objfeed.ctp)
