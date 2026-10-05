import argparse
import os

import pandas as pd


def prepare_data(rawdir, outdir):

    print("[prepare] building dataset...")

    # ---------------------------------------------------------
    # NOAA anomaly database
    # ---------------------------------------------------------

    anomaly_path = os.path.join(
        rawdir,
        "NOAA_Info",
        "anom5j.xls"
    )

    df = pd.read_excel(anomaly_path)

    df["BIRD"] = (
        df["BIRD"]
        .astype(str)
        .str.replace("-", " ", regex=False)
        .str.strip()
        .str.lower()
    )

    df["ADATE"] = pd.to_datetime(
        df["ADATE"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["BIRD", "ADATE"]
    )

    # ---------------------------------------------------------
    # CelesTrak SATCAT
    # ---------------------------------------------------------

    satcat_path = os.path.join(
        rawdir,
        "satcat.csv"
    )

    sc = pd.read_csv(satcat_path)

    required_columns = [
        "OBJECT_NAME",
        "PERIGEE",
        "INCLINATION"
    ]

    missing = [
        col
        for col in required_columns
        if col not in sc.columns
    ]

    if missing:
        raise ValueError(
            f"CelesTrak SATCAT is missing columns: {missing}"
        )

    sc = sc[
        [
            "OBJECT_NAME",
            "PERIGEE",
            "INCLINATION"
        ]
    ].copy()

    sc.columns = [
        "Name",
        "Perigee (km)",
        "Inclination (deg)"
    ]

    sc["Name"] = (
        sc["Name"]
        .astype(str)
        .str.replace("-", " ", regex=False)
        .str.strip()
        .str.lower()
    )

    # ---------------------------------------------------------
    # Sunspot data
    # ---------------------------------------------------------

    sunspot_path = os.path.join(
        rawdir,
        "sunspot.csv"
    )

    ss = pd.read_csv(
        sunspot_path,
        sep=";",
        header=None
    )

    ss = ss.iloc[:, :5].copy()

    ss.columns = [
        "Year",
        "Month",
        "Day",
        "DecimalYear",
        "SSN"
    ]

    ss["Year"] = pd.to_numeric(
        ss["Year"],
        errors="coerce"
    )

    ss["Month"] = pd.to_numeric(
        ss["Month"],
        errors="coerce"
    )

    ss["Day"] = pd.to_numeric(
        ss["Day"],
        errors="coerce"
    )

    ss["SSN"] = pd.to_numeric(
        ss["SSN"],
        errors="coerce"
    )

    ss["T0"] = pd.to_datetime(
        dict(
            year=ss["Year"],
            month=ss["Month"],
            day=ss["Day"]
        ),
        errors="coerce"
    )

    ss = ss[
        ["SSN", "T0"]
    ].dropna()

    # ---------------------------------------------------------
    # Satellites with multiple anomalies
    # ---------------------------------------------------------

    counts = df["BIRD"].value_counts()

    anomaly_satellites = counts[
        counts > 1
    ].index

    anom = df[
        df["BIRD"].isin(
            anomaly_satellites
        )
    ].copy()

    # ---------------------------------------------------------
    # Match NOAA satellites with CelesTrak
    # ---------------------------------------------------------

    newsc = sc[
        sc["Name"].isin(
            anom["BIRD"].unique()
        )
    ].copy()

    newanom = anom[
        anom["BIRD"].isin(
            newsc["Name"].unique()
        )
    ][
        ["BIRD", "ADATE"]
    ].copy()

    # ---------------------------------------------------------
    # GPS anomaly records retained from original pipeline
    # ---------------------------------------------------------

    gps_names = [
        "gps 5111",
        "gps 5112",
        "gps 5113",
        "gps 5114",
        "gps 5118",
        "gps 9794",
        "gps svn11"
    ]

    gpsanom = anom[
        anom["BIRD"].isin(gps_names)
    ][
        ["BIRD", "ADATE"]
    ].copy()

    finalanom = pd.concat(
        [
            newanom,
            gpsanom
        ],
        ignore_index=True
    )

    # Remove duplicate records that may have entered
    # through both matching paths.
    finalanom = finalanom.drop_duplicates(
        subset=[
            "BIRD",
            "ADATE"
        ]
    )

    # ---------------------------------------------------------
    # Calculate TTE
    # ---------------------------------------------------------

    rows = []

    for satellite in finalanom["BIRD"].unique():

        group = (
            finalanom[
                finalanom["BIRD"] == satellite
            ]
            .sort_values("ADATE")
            .reset_index(drop=True)
        )

        for i in range(len(group) - 1):

            tte = (
                group.loc[i + 1, "ADATE"]
                - group.loc[i, "ADATE"]
            ).days

            rows.append(
                [
                    satellite,
                    tte,
                    group.loc[i, "ADATE"]
                ]
            )

    data = pd.DataFrame(
        rows,
        columns=[
            "BIRD",
            "TTE",
            "T0"
        ]
    )

    # ---------------------------------------------------------
    # TTE filtering
    # ---------------------------------------------------------

    data = data[
        (data["TTE"] > 0) &
        (data["TTE"] < 365)
    ]

    counts = data["BIRD"].value_counts()

    data = data[
        data["BIRD"].isin(
            counts[counts > 20].index
        )
    ]

    data = data[
        ~data["BIRD"].isin(
            [
                "scatha",
                "ecs 1"
            ]
        )
    ]

    # ---------------------------------------------------------
    # Merge sunspot feature
    # ---------------------------------------------------------

    data = pd.merge(
        data,
        ss,
        on="T0",
        how="inner"
    )

    # ---------------------------------------------------------
    # Starting month feature
    # ---------------------------------------------------------

    data["Month"] = data["T0"].dt.month

    data = data.sort_values(
        "TTE"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Satellite orbital features
    # ---------------------------------------------------------

    final_satellites = data[
        "BIRD"
    ].unique()

    # Direct name matches
    satfeat = sc[
        sc["Name"].isin(
            final_satellites
        )
    ].copy()

    # ---------------------------------------------------------
    # GPS name mapping
    #
    # NOAA:
    #   gps 5111
    #   gps 5112
    #   gps 5113
    #   gps 5114
    #   gps 5118
    #   gps 9794
    #
    # CelesTrak:
    #   NAVSTAR 1 (OPS 5111)
    #   NAVSTAR 2 (OPS 5112)
    #   NAVSTAR 3 (OPS 5113)
    #   NAVSTAR 4 (OPS 5114)
    #   NAVSTAR 6 (OPS 5118)
    #   NAVSTAR 8 (OPS 9794)
    # ---------------------------------------------------------

    gps_mapping = {
        "gps 5111": "navstar 1 (ops 5111)",
        "gps 5112": "navstar 2 (ops 5112)",
        "gps 5113": "navstar 3 (ops 5113)",
        "gps 5114": "navstar 4 (ops 5114)",
        "gps 5118": "navstar 6 (ops 5118)",
        "gps 9794": "navstar 8 (ops 9794)"
    }

    for gps_name, celestrak_name in gps_mapping.items():

        if gps_name in final_satellites:

            match = sc[
                sc["Name"] == celestrak_name
            ].copy()

            if len(match) > 0:

                match["Name"] = gps_name

                satfeat = pd.concat(
                    [
                        satfeat,
                        match
                    ],
                    ignore_index=True
                )

    # ---------------------------------------------------------
    # Clean satellite feature table
    # ---------------------------------------------------------

    satfeat = satfeat[
        [
            "Name",
            "Perigee (km)",
            "Inclination (deg)"
        ]
    ].drop_duplicates(
        subset=["Name"]
    )

    # ---------------------------------------------------------
    # Final outputs
    # ---------------------------------------------------------

    os.makedirs(
        outdir,
        exist_ok=True
    )

    data.to_csv(
        os.path.join(
            outdir,
            "data.csv"
        ),
        index=False
    )

    satfeat.to_csv(
        os.path.join(
            outdir,
            "satfeat.csv"
        ),
        index=False
    )

    print(
        f"[prepare] wrote data.csv "
        f"({len(data)} events)"
    )

    print(
        f"[prepare] wrote satfeat.csv "
        f"({len(satfeat)} satellites)"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rawdir",
        default="raw"
    )

    parser.add_argument(
        "--outdir",
        default="."
    )

    args = parser.parse_args()

    prepare_data(
        args.rawdir,
        args.outdir
    )