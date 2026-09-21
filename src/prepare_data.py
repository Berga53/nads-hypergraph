"""Build clean annual hypergraph incidences and real company edge parameters.

Inputs
------
``data/raw/rete_IPL.dta``
    Municipality-company participations used to build the hypergraph.
``data/raw/dataset_partecipate.xlsx``
    Annual company accounts and employment information.
``data/raw/Anagrafe_comuni.csv`` and ``data/raw/popolazione.csv``
    Municipality fiscal-code to BDAP mapping and annual population values.

Outputs
-------
``data/processed/rete_YYYY.csv``
    One incidence per municipality-company pair with the ownership quota.
``data/processed/hyperedge_parameters.csv``
    One row per company/hyperedge/year, including the single GIP parameter and
    enough source-year information to audit missing-data treatment.
``data/processed/node_parameters.csv``
    One row per municipality/node/year with its BDAP key and population weight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


NETWORK_COLUMNS = [
    "anno",
    "codfisc",
    "codfisc_ente",
    "denominazione_ente",
    "quota_totale_ente",
]

COMPANY_COLUMNS = [
    "anno",
    "codfisc",
    "denominazione_imp",
    "fatturato",
    "fatturato_mef",
    "attivo",
    "va",
    "dipendenti",
    "occ_tot_inps",
    "ndip_bil",
    "addetti_mef",
]

SIZE_METRICS = [
    "turnover_keur",
    "assets_keur",
    "employees",
    "value_added_keur",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare annual hypergraphs and one real parameter per hyperedge."
    )
    parser.add_argument(
        "--network", type=Path, default=RAW_DATA_DIR / "rete_IPL.dta"
    )
    parser.add_argument(
        "--companies",
        type=Path,
        default=RAW_DATA_DIR / "dataset_partecipate.xlsx",
    )
    parser.add_argument(
        "--municipality-registry",
        type=Path,
        default=RAW_DATA_DIR / "Anagrafe_comuni.csv",
        help="BDAP registry containing the CF to Id_Ente crosswalk.",
    )
    parser.add_argument(
        "--population",
        type=Path,
        default=RAW_DATA_DIR / "popolazione.csv",
        help="Wide population file indexed by BDAP Id_Ente.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROCESSED_DATA_DIR
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset such as --years 2015 2018; default uses all years >= 2010.",
    )
    parser.add_argument(
        "--imputation-direction",
        choices=("nearest", "past"),
        default="nearest",
        help="Nearest same-company value, or earlier values only for look-ahead safety.",
    )
    parser.add_argument(
        "--max-imputation-gap",
        type=int,
        default=None,
        help="Optional maximum source-target distance in years.",
    )
    args = parser.parse_args()
    if args.max_imputation_gap is not None and args.max_imputation_gap < 0:
        parser.error("--max-imputation-gap must be non-negative")
    return args


def _numeric_identifier(values: pd.Series, label: str) -> pd.Series:
    identifiers = pd.to_numeric(values, errors="coerce").round().astype("Int64")
    if identifiers.isna().any():
        raise ValueError(
            f"{label} contains {int(identifiers.isna().sum()):,} invalid identifiers"
        )
    return identifiers.astype("int64")


def load_network(network_path: Path, years: list[int] | None) -> pd.DataFrame:
    network = pd.read_stata(
        network_path,
        columns=NETWORK_COLUMNS,
        convert_categoricals=False,
    )
    network = network.rename(
        columns={
            "anno": "year",
            "codfisc": "company_id",
            "codfisc_ente": "municipality_id",
            "denominazione_ente": "municipality_name",
            "quota_totale_ente": "ownership_share",
        }
    )
    network["year"] = pd.to_numeric(network["year"], errors="coerce").astype("Int64")
    network = network.dropna(subset=["year"]).copy()
    network["year"] = network["year"].astype("int16")
    network = network.loc[network["year"].ge(2010)].copy()
    if years:
        network = network.loc[network["year"].isin(years)].copy()
    if network.empty:
        raise ValueError("No network observations remain after the year filter")

    network["company_id"] = _numeric_identifier(network["company_id"], "codfisc")
    network["municipality_id"] = _numeric_identifier(
        network["municipality_id"], "codfisc_ente"
    )
    network["ownership_share"] = pd.to_numeric(
        network["ownership_share"], errors="coerce"
    )
    if network["ownership_share"].isna().any():
        raise ValueError("quota_totale_ente contains missing or non-numeric values")

    duplicate_count = int(
        network.duplicated(["year", "company_id", "municipality_id"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"Network contains {duplicate_count:,} duplicate year/company/municipality rows"
        )
    return network.sort_values(
        ["year", "municipality_id", "company_id"], kind="stable"
    ).reset_index(drop=True)


def _first_non_missing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].bfill(axis=1).iloc[:, 0]


def load_companies(workbook_path: Path) -> pd.DataFrame:
    company = pd.read_excel(
        workbook_path,
        sheet_name="Sheet1",
        usecols=COMPANY_COLUMNS,
        engine="openpyxl",
    )
    company = company.rename(
        columns={
            "anno": "year",
            "codfisc": "company_id",
            "denominazione_imp": "company_name",
            "attivo": "assets_keur",
            "va": "value_added_keur",
        }
    )
    company["year"] = pd.to_numeric(company["year"], errors="coerce").astype("Int64")
    company = company.dropna(subset=["year"]).copy()
    company["year"] = company["year"].astype("int16")
    company["company_id"] = _numeric_identifier(company["company_id"], "codfisc")
    if company.duplicated(["year", "company_id"]).any():
        raise ValueError("Company workbook contains duplicate company-year rows")

    numeric_columns = [
        "fatturato",
        "fatturato_mef",
        "assets_keur",
        "value_added_keur",
        "dipendenti",
        "occ_tot_inps",
        "ndip_bil",
        "addetti_mef",
    ]
    company[numeric_columns] = company[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Accounts are in thousands of euros; the MEF turnover field is in euros.
    company["turnover_keur"] = company["fatturato"].combine_first(
        company["fatturato_mef"] / 1_000.0
    )
    company["turnover_source"] = np.select(
        [company["fatturato"].notna(), company["fatturato_mef"].notna()],
        ["financial_statements", "MEF"],
        default=None,
    )

    employee_columns = ["dipendenti", "occ_tot_inps", "ndip_bil", "addetti_mef"]
    company["employees"] = _first_non_missing(company, employee_columns)
    company["employees_source"] = np.select(
        [company[column].notna() for column in employee_columns],
        ["combined", "INPS", "financial_statements", "MEF"],
        default=None,
    )

    keep = [
        "year",
        "company_id",
        "company_name",
        *SIZE_METRICS,
        "turnover_source",
        "employees_source",
    ]
    company = company[keep]
    company["company_record_exact"] = True
    return company


def _string_identifier(
    values: pd.Series,
    label: str,
    *,
    width: int | None = None,
) -> pd.Series:
    identifiers = values.astype("string").str.strip()
    identifiers = identifiers.str.replace(r"\.0+$", "", regex=True)
    invalid = identifiers.isna() | identifiers.eq("") | identifiers.eq("<NA>")
    if invalid.any():
        raise ValueError(
            f"{label} contains {int(invalid.sum()):,} missing identifiers"
        )
    if width is not None:
        identifiers = identifiers.str.zfill(width)
    return identifiers


def load_population(
    registry_path: Path,
    population_path: Path,
    years: list[int],
) -> pd.DataFrame:
    """Return yearly population after the exact CF -> BDAP crosswalk.

    The network identifies municipalities by fiscal code, whereas the population
    file uses ``Id_Ente`` from the BDAP registry. Identifiers deliberately remain
    strings during the join so leading zeroes cannot be lost.
    """
    registry = pd.read_csv(
        registry_path,
        sep=";",
        encoding="latin-1",
        dtype="string",
        usecols=["Id_Ente", "CF", "Denominazione"],
    ).rename(
        columns={
            "Id_Ente": "municipality_bdap_id",
            "CF": "municipality_tax_code",
            "Denominazione": "registry_municipality_name",
        }
    )
    registry["municipality_bdap_id"] = _string_identifier(
        registry["municipality_bdap_id"], "Anagrafe Id_Ente"
    )
    registry["municipality_tax_code"] = _string_identifier(
        registry["municipality_tax_code"], "Anagrafe CF", width=11
    )

    population = pd.read_csv(population_path, dtype="string").rename(
        columns={"BDAP": "municipality_bdap_id"}
    )
    if "municipality_bdap_id" not in population:
        raise ValueError("Population file must contain a BDAP column")
    population["municipality_bdap_id"] = _string_identifier(
        population["municipality_bdap_id"], "Population BDAP"
    )
    if population["municipality_bdap_id"].duplicated().any():
        raise ValueError("Population file contains duplicate BDAP identifiers")

    year_columns = [str(int(year)) for year in sorted(set(years))]
    missing_years = [year for year in year_columns if year not in population]
    if missing_years:
        raise ValueError(
            "Population file is missing requested year column(s): "
            + ", ".join(missing_years)
        )

    # The full registry contains many kinds of public bodies and fiscal codes can
    # recur there. Restricting it to the BDAP IDs in the population table isolates
    # the municipality crosswalk, which must be one-to-one.
    municipality_registry = registry.loc[
        registry["municipality_bdap_id"].isin(population["municipality_bdap_id"])
    ].copy()
    if municipality_registry["municipality_bdap_id"].duplicated().any():
        raise ValueError("Anagrafe contains duplicate population BDAP identifiers")
    if municipality_registry["municipality_tax_code"].duplicated().any():
        raise ValueError("Anagrafe maps a municipality fiscal code to multiple BDAP IDs")

    population_long = population.melt(
        id_vars="municipality_bdap_id",
        value_vars=year_columns,
        var_name="year",
        value_name="population",
    )
    population_long["year"] = population_long["year"].astype("int16")
    population_long["population"] = pd.to_numeric(
        population_long["population"], errors="coerce"
    )
    invalid_population = (
        population_long["population"].isna()
        | ~np.isfinite(population_long["population"])
        | population_long["population"].le(0)
    )
    if invalid_population.any():
        raise ValueError(
            "Population file contains "
            f"{int(invalid_population.sum()):,} missing, non-finite, or non-positive values"
        )

    joined = population_long.merge(
        municipality_registry,
        on="municipality_bdap_id",
        how="left",
        validate="m:1",
    )
    if joined["municipality_tax_code"].isna().any():
        raise ValueError(
            "Anagrafe does not contain every BDAP identifier in the population file"
        )
    return joined[
        [
            "year",
            "municipality_tax_code",
            "municipality_bdap_id",
            "registry_municipality_name",
            "population",
        ]
    ].sort_values(["year", "municipality_tax_code"], kind="stable")


def _nearest_fill(
    panel: pd.DataFrame,
    column: str,
    direction: str,
    max_gap: int | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    groups = panel.groupby("company_id", sort=False, observed=True)
    observed = panel[column].notna()
    observed_year = panel["year"].where(observed)
    previous_value = groups[column].ffill()
    previous_year = observed_year.groupby(panel["company_id"], sort=False).ffill()

    if direction == "past":
        filled = previous_value
        source_year = previous_year
    else:
        next_value = groups[column].bfill()
        next_year = observed_year.groupby(panel["company_id"], sort=False).bfill()
        previous_distance = panel["year"] - previous_year
        next_distance = next_year - panel["year"]
        use_previous = previous_value.notna() & (
            next_value.isna() | previous_distance.le(next_distance)
        )
        filled = previous_value.where(use_previous, next_value)
        source_year = previous_year.where(use_previous, next_year)

    distance = (panel["year"] - source_year).abs()
    if max_gap is not None:
        allowed = distance.le(max_gap)
        filled = filled.where(allowed)
        source_year = source_year.where(allowed)
        distance = distance.where(allowed)
    return filled, source_year.astype("Int16"), distance.astype("Int16")


def build_company_panel(
    edge_keys: pd.DataFrame,
    company: pd.DataFrame,
    direction: str,
    max_gap: int | None,
) -> pd.DataFrame:
    company_ids = np.sort(edge_keys["company_id"].unique())
    source = company.loc[company["company_id"].isin(company_ids)].copy()
    observed_company_ids = set(source["company_id"].unique())

    years = np.arange(
        min(int(edge_keys["year"].min()), int(source["year"].min())),
        max(int(edge_keys["year"].max()), int(source["year"].max())) + 1,
        dtype=np.int16,
    )
    panel = pd.MultiIndex.from_product(
        [company_ids, years], names=["company_id", "year"]
    ).to_frame(index=False)
    panel = panel.merge(source, on=["company_id", "year"], how="left", validate="1:1")
    panel["company_ever_in_workbook"] = panel["company_id"].isin(observed_company_ids)
    panel["company_record_exact"] = panel["company_record_exact"].eq(True)
    panel = panel.sort_values(["company_id", "year"], kind="stable").reset_index(drop=True)

    company_name, _, _ = _nearest_fill(panel, "company_name", direction, max_gap)
    panel["company_name"] = company_name

    for metric in SIZE_METRICS:
        value, source_year, distance = _nearest_fill(panel, metric, direction, max_gap)
        panel[metric] = value
        panel[f"{metric}_source_year"] = source_year
        panel[f"{metric}_year_distance"] = distance
        panel[f"{metric}_imputed"] = value.notna() & distance.gt(0)

    for source_column in ["turnover_source", "employees_source"]:
        value, _, _ = _nearest_fill(panel, source_column, direction, max_gap)
        panel[source_column] = value

    return edge_keys.merge(
        panel, on=["year", "company_id"], how="left", validate="1:1"
    )


def _gap_confidence(distance: pd.Series) -> pd.Series:
    distance = pd.to_numeric(distance, errors="coerce")
    confidence = pd.Series(np.nan, index=distance.index, dtype=float)
    confidence.loc[distance.eq(0)] = 1.00
    confidence.loc[distance.eq(1)] = 0.90
    confidence.loc[distance.eq(2)] = 0.80
    confidence.loc[distance.between(3, 5)] = 0.60
    confidence.loc[distance.gt(5)] = 0.35
    return confidence


def add_edge_parameters(panel: pd.DataFrame) -> pd.DataFrame:
    percentile_columns: list[str] = []
    confidence_columns: list[str] = []
    for metric in SIZE_METRICS:
        valid_value = panel[metric].where(panel[metric].ge(0))
        percentile_column = f"{metric}_percentile_year"
        panel[percentile_column] = valid_value.groupby(panel["year"]).rank(
            method="average", pct=True
        ) * 100
        percentile_columns.append(percentile_column)

        confidence_column = f"_{metric}_confidence"
        panel[confidence_column] = _gap_confidence(
            panel[f"{metric}_year_distance"]
        ).where(valid_value.notna())
        confidence_columns.append(confidence_column)

    panel["company_size_indicator_count"] = panel[percentile_columns].notna().sum(axis=1)
    panel["company_size_score_0_100"] = panel[percentile_columns].mean(axis=1)
    panel["company_size_data_confidence_0_1"] = panel[confidence_columns].mean(axis=1)

    count = panel["company_size_indicator_count"]
    confidence = panel["company_size_data_confidence_0_1"]
    panel["company_size_data_quality"] = np.select(
        [
            count.eq(0),
            count.ge(3) & confidence.ge(0.80),
            count.ge(2) & confidence.ge(0.60),
        ],
        ["missing", "high", "medium"],
        default="low",
    )

    size_0_1 = panel["company_size_score_0_100"].div(100)
    neutral_size = size_0_1.groupby(panel["year"]).transform("median")
    if neutral_size.isna().any():
        raise ValueError("At least one year has no usable company-size information")
    filled_size = size_0_1.fillna(neutral_size)
    filled_confidence = confidence.fillna(0).clip(lower=0, upper=1)
    panel["edge_parameter_base_0_1"] = (
        filled_confidence * filled_size
        + (1.0 - filled_confidence) * neutral_size
    )
    parameter_mean = panel.groupby("year")["edge_parameter_base_0_1"].transform("mean")
    panel["edge_parameter_mean_1"] = panel["edge_parameter_base_0_1"] / parameter_mean
    panel["edge_parameter_basis"] = np.select(
        [size_0_1.isna(), filled_confidence.lt(1)],
        ["year_median_no_company_size", "company_size_confidence_shrunk"],
        default="company_size_direct",
    )
    panel.drop(columns=confidence_columns, inplace=True)
    return panel


def build_parameters(
    network: pd.DataFrame,
    company: pd.DataFrame,
    direction: str,
    max_gap: int | None,
) -> pd.DataFrame:
    edge_size = (
        network.groupby(["year", "company_id"], sort=True)["municipality_id"]
        .nunique()
        .rename("municipality_count")
        .reset_index()
    )
    edge_keys = edge_size[["year", "company_id"]]
    panel = build_company_panel(edge_keys, company, direction, max_gap)
    panel = panel.merge(edge_size, on=["year", "company_id"], how="left", validate="1:1")
    panel = add_edge_parameters(panel)
    panel["company_tax_code"] = panel["company_id"].astype(str).str.zfill(11)
    panel["company_name"] = panel["company_name"].fillna("UNMATCHED COMPANY")

    leading = [
        "year",
        "company_id",
        "company_tax_code",
        "company_name",
        "municipality_count",
        "edge_parameter_base_0_1",
        "edge_parameter_mean_1",
        "edge_parameter_basis",
        "company_size_score_0_100",
        "company_size_indicator_count",
        "company_size_data_confidence_0_1",
        "company_size_data_quality",
        "company_ever_in_workbook",
        "company_record_exact",
        *SIZE_METRICS,
        "turnover_source",
        "employees_source",
    ]
    for metric in SIZE_METRICS:
        leading.extend(
            [
                f"{metric}_source_year",
                f"{metric}_year_distance",
                f"{metric}_imputed",
                f"{metric}_percentile_year",
            ]
        )
    return panel[leading].sort_values(["year", "company_id"], kind="stable")


def build_node_parameters(
    network: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:
    """Build one auditable population-derived weight per node and year."""
    nodes = (
        network[["year", "municipality_id", "municipality_name"]]
        .drop_duplicates(["year", "municipality_id"])
        .copy()
    )
    nodes["municipality_tax_code"] = (
        nodes["municipality_id"].astype(str).str.zfill(11)
    )
    nodes = nodes.merge(
        population,
        on=["year", "municipality_tax_code"],
        how="left",
        validate="1:1",
    )
    nodes["population_data_available"] = nodes["population"].notna()

    # Historic/ceased municipalities are not always present in the current BDAP
    # population table. Keep the raw value missing for auditability, but give the
    # model a neutral within-year median so every graph node remains usable.
    year_median = nodes.groupby("year")["population"].transform("median")
    if year_median.isna().any():
        raise ValueError("At least one network year has no matched population data")
    nodes["node_weight_population"] = nodes["population"].fillna(year_median)
    weight_mean = nodes.groupby("year")["node_weight_population"].transform("mean")
    nodes["node_weight_mean_1"] = nodes["node_weight_population"] / weight_mean
    nodes["node_weight_basis"] = np.where(
        nodes["population_data_available"],
        "population_direct",
        "year_median_no_bdap_match",
    )

    leading = [
        "year",
        "municipality_id",
        "municipality_tax_code",
        "municipality_bdap_id",
        "municipality_name",
        "registry_municipality_name",
        "population",
        "population_data_available",
        "node_weight_population",
        "node_weight_mean_1",
        "node_weight_basis",
    ]
    return nodes[leading].sort_values(
        ["year", "municipality_id"], kind="stable"
    )


def build_diagnostics(
    parameters: pd.DataFrame,
    node_parameters: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for year, frame in parameters.groupby("year", sort=True):
        row: dict[str, float | int] = {
            "year": int(year),
            "hyperedges": len(frame),
            "exact_company_record_pct": frame["company_record_exact"].mean() * 100,
            "company_ever_in_workbook_pct": frame["company_ever_in_workbook"].mean()
            * 100,
            "company_size_available_pct": frame["company_size_score_0_100"].notna().mean()
            * 100,
            "neutral_parameter_fallback_pct": frame["edge_parameter_basis"]
            .eq("year_median_no_company_size")
            .mean()
            * 100,
            "edge_parameter_mean": frame["edge_parameter_mean_1"].mean(),
        }
        for metric in SIZE_METRICS:
            row[f"{metric}_available_pct"] = frame[metric].notna().mean() * 100
            row[f"{metric}_exact_pct"] = (
                frame[metric].notna() & frame[f"{metric}_year_distance"].eq(0)
            ).mean() * 100
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    node_diagnostics = (
        node_parameters.groupby("year", sort=True)
        .agg(
            nodes=("municipality_id", "size"),
            population_available_pct=("population_data_available", "mean"),
            node_weight_mean=("node_weight_mean_1", "mean"),
        )
        .reset_index()
    )
    node_diagnostics["population_available_pct"] *= 100
    return diagnostics.merge(node_diagnostics, on="year", how="left", validate="1:1")


def write_outputs(
    network: pd.DataFrame,
    parameters: pd.DataFrame,
    node_parameters: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for year, frame in network.groupby("year", sort=True):
        annual = frame[
            ["municipality_id", "municipality_name", "company_id", "ownership_share"]
        ].rename(
            columns={
                "municipality_id": "CF Comune",
                "municipality_name": "Comune",
                "company_id": "CF Partecipata",
                "ownership_share": "Quota",
            }
        )
        annual.to_csv(output_dir / f"rete_{int(year)}.csv", index=False)

    parameters.to_csv(
        output_dir / "hyperedge_parameters.csv",
        index=False,
        float_format="%.8g",
    )
    node_parameters.to_csv(
        output_dir / "node_parameters.csv",
        index=False,
        float_format="%.8g",
    )
    diagnostics = build_diagnostics(parameters, node_parameters)
    diagnostics.to_csv(
        output_dir / "data_quality_by_year.csv", index=False, float_format="%.3f"
    )
    summary = {
        "years": [int(value) for value in sorted(network["year"].unique())],
        "incidences": int(len(network)),
        "hyperedge_year_rows": int(len(parameters)),
        "unique_companies": int(parameters["company_id"].nunique()),
        "node_year_rows": int(len(node_parameters)),
        "unique_municipalities": int(node_parameters["municipality_id"].nunique()),
        "population_available_pct": round(
            float(node_parameters["population_data_available"].mean() * 100), 3
        ),
        "exact_company_record_pct": round(
            float(parameters["company_record_exact"].mean() * 100), 3
        ),
        "companies_never_in_workbook": int(
            parameters.loc[~parameters["company_ever_in_workbook"], "company_id"].nunique()
        ),
        "company_size_available_pct": round(
            float(parameters["company_size_score_0_100"].notna().mean() * 100), 3
        ),
        "imputation_direction": args.imputation_direction,
        "max_imputation_gap": args.max_imputation_gap,
        "parameter_definition": (
            "confidence-adjusted mean of within-year turnover, assets, employees, "
            "and value-added percentiles"
        ),
        "node_weight_definition": (
            "annual population mapped from municipality fiscal code to BDAP Id_Ente; "
            "unmatched historic municipalities use the within-year median"
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    print(f"Loading network: {args.network}")
    network = load_network(args.network, args.years)
    print(
        f"Loaded {len(network):,} incidences across "
        f"{network['year'].nunique()} year(s)."
    )

    print(f"Loading company attributes: {args.companies}")
    company = load_companies(args.companies)
    print(f"Loaded {len(company):,} company-year records.")

    network_years = [int(value) for value in sorted(network["year"].unique())]
    print(
        "Loading municipality populations: "
        f"{args.municipality_registry} -> {args.population}"
    )
    population = load_population(
        args.municipality_registry,
        args.population,
        network_years,
    )
    print(f"Loaded {len(population):,} mapped municipality-year populations.")

    parameters = build_parameters(
        network,
        company,
        direction=args.imputation_direction,
        max_gap=args.max_imputation_gap,
    )
    node_parameters = build_node_parameters(network, population)
    write_outputs(network, parameters, node_parameters, args.output_dir, args)
    print(f"Wrote clean data to {args.output_dir}")


if __name__ == "__main__":
    main()
