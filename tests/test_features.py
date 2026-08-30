import pandas as pd
import numpy as np
from src.features import add_ratios


def test_ratios_produce_no_infinities():
    df = pd.DataFrame({
        "AMT_CREDIT": [100.0, 200.0],
        "AMT_INCOME_TOTAL": [50.0, 0.0],     # zero income
        "AMT_ANNUITY": [10.0, 20.0],
        "AMT_GOODS_PRICE": [90.0, 180.0],
        "CNT_FAM_MEMBERS": [2.0, 1.0],
    })
    out = add_ratios(df)
    assert not np.isinf(out.select_dtypes("number")).any().any()


def test_ratios_do_not_drop_rows():
    df = pd.DataFrame({
        "AMT_CREDIT": [100.0], "AMT_INCOME_TOTAL": [50.0],
        "AMT_ANNUITY": [10.0], "AMT_GOODS_PRICE": [90.0],
        "CNT_FAM_MEMBERS": [2.0],
    })
    assert len(add_ratios(df)) == len(df)