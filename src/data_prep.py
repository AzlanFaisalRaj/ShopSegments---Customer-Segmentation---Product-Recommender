"""Data preparation: load Online Retail II, clean transactions, build RFM features."""
from pathlib import Path

import pandas as pd

PRODUCT_CODE_REGEX = r"^\d{5}"   # real products start with 5 digits (e.g. 85123A); POST, M, D, ... do not


def load_data(excel_path):
    """Read both sheets of Online Retail II and combine them into one DataFrame."""
    sheets = pd.read_excel(Path(excel_path), sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    # Invoice / StockCode / Description mix numbers and text in the raw file
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].where(df["Description"].isna(), df["Description"].astype(str))
    return df


def clean_transactions(df_raw, return_log=False):
    """Return clean purchase rows with LineTotal and an is_product flag.

    Steps: drop duplicates, drop null date/qty/price, drop missing Customer ID,
    drop cancelled invoices (prefix 'C'), drop Quantity <= 0 or Price <= 0.
    """
    df = df_raw.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    log = [("start (combined raw)", len(df))]

    df = df.drop_duplicates()
    log.append(("drop exact duplicates", len(df)))
    df = df.dropna(subset=["InvoiceDate", "Quantity", "Price"])
    log.append(("drop null date/qty/price", len(df)))
    df = df.dropna(subset=["Customer ID"])
    log.append(("drop missing Customer ID", len(df)))
    df = df[~df["Invoice"].str.startswith("C")]
    log.append(("drop cancelled invoices", len(df)))
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    log.append(("drop Quantity<=0 or Price<=0", len(df)))

    df["Customer ID"] = df["Customer ID"].astype(int)
    df["LineTotal"] = df["Quantity"] * df["Price"]
    df["is_product"] = df["StockCode"].str.match(PRODUCT_CODE_REGEX)
    df = df.reset_index(drop=True)

    if return_log:
        log_df = pd.DataFrame(log, columns=["step", "rows_after"])
        log_df["rows_removed"] = (-log_df["rows_after"].diff()).fillna(0).astype(int)
        return df, log_df
    return df


def product_rows(df):
    """Keep only real products (drops postage, manual entries, bank charges...)."""
    return df[df["is_product"]].copy()


def build_rfm(df, snapshot_date=None):
    """One row per customer: Recency (days), Frequency (distinct invoices), Monetary (GBP).

    Snapshot date = last transaction date + 1 day. Dates are normalised so a customer
    who bought on the last day gets Recency 1, not 0.
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    g = df.groupby("Customer ID")
    rfm = pd.DataFrame({
        "Recency": (snapshot_date - g["InvoiceDate"].max().dt.normalize()).dt.days,
        "Frequency": g["Invoice"].nunique(),
        "Monetary": g["LineTotal"].sum(),
    }).reset_index().rename(columns={"Customer ID": "CustomerID"})
    return rfm
