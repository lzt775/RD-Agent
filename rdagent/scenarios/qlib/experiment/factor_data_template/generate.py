import os

import qlib
from qlib.data import D


def generate():
    qlib.init(provider_uri=os.environ.get("RD_QLIB_PROVIDER", "~/.qlib/qlib_data/cn_data"), kernels=1)
    fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
    instruments = D.instruments()
    data = D.features(instruments, fields, start_time="2008-12-29", freq="day").swaplevel().sort_index()
    data.to_hdf("daily_pv_all.h5", key="data")
    names = data.index.get_level_values("instrument").unique()[:100]
    sample = data.loc["2018-01-01":"2019-12-31"]
    debug = sample.loc[sample.index.get_level_values("instrument").isin(names)]
    if debug.empty:
        raise ValueError("The Qlib dataset must cover the official 2018–2019 debug period")
    debug.to_hdf("daily_pv_debug.h5", key="data")


if __name__ == "__main__":
    generate()
