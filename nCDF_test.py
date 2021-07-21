from netCDF4 import Dataset
import os

nc_path = os.path.join(os.getcwd(), "data", "nc")

rootgrp = Dataset(
    os.path.join(nc_path, "SNPP_VIIRS.20210705T140600.L2.SST3.NRT.nc"),
    "r+",
    format="NETCDF4",
)

print(rootgrp)
rootgrp.close()
