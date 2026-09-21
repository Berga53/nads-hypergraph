"""ISTAT geographical classifications for municipality-network partitions."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "raw" / "Anagrafe_comuni.csv"

ISTAT_MACROAREAS = (
    "NORD-OVEST",
    "NORD-EST",
    "CENTRO",
    "SUD",
    "ISOLE",
)
ISTAT_MACROAREA_CODES = {
    "NORD-OVEST": 1,
    "NORD-EST": 2,
    "CENTRO": 3,
    "SUD": 4,
    "ISOLE": 5,
}


def normalize_macroarea(value: str) -> str:
    """Return the canonical registry label for a CLI/user macro-area value."""
    normalized = value.strip().upper().replace("_", "-").replace(" ", "-")
    if normalized not in ISTAT_MACROAREA_CODES:
        choices = ", ".join(ISTAT_MACROAREAS)
        raise ValueError(f"Unknown ISTAT macro-area {value!r}; choose one of {choices}")
    return normalized


def load_municipality_geography(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> pd.DataFrame:
    """Load one validated ISTAT macro-area and region per municipality fiscal code.

    The BDAP registry includes multiple records for some fiscal codes. Historic
    municipality-code changes can therefore produce several registry rows, but
    their region and macro-area must agree before they are collapsed here.
    """
    source_columns = [
        "CF",
        "Codice_Regione",
        "Dizione_Regione",
        "Codice_Zona",
        "Dizione_zona",
    ]
    registry = pd.read_csv(
        registry_path,
        sep=";",
        encoding="latin-1",
        dtype="string",
        usecols=source_columns,
    )
    registry[source_columns] = registry[source_columns].apply(
        lambda column: column.str.strip()
    )
    registry["municipality_id"] = pd.to_numeric(
        registry["CF"].str.replace(r"\.0+$", "", regex=True), errors="coerce"
    ).astype("Int64")
    registry["istat_macroarea_code"] = pd.to_numeric(
        registry["Codice_Zona"], errors="coerce"
    ).astype("Int64")
    registry = registry.dropna(
        subset=[
            "municipality_id",
            "Codice_Regione",
            "Dizione_Regione",
            "istat_macroarea_code",
            "Dizione_zona",
        ]
    ).copy()
    registry["istat_macroarea"] = registry["Dizione_zona"].map(normalize_macroarea)

    stable_columns = [
        "Codice_Regione",
        "Dizione_Regione",
        "istat_macroarea_code",
        "istat_macroarea",
    ]
    ambiguities = registry.groupby("municipality_id")[stable_columns].nunique()
    ambiguous_ids = ambiguities.index[ambiguities.gt(1).any(axis=1)]
    if len(ambiguous_ids):
        examples = ", ".join(str(int(value)) for value in ambiguous_ids[:5])
        raise ValueError(
            "Municipality fiscal codes map to conflicting ISTAT geography; "
            f"examples: {examples}"
        )

    geography = (
        registry.drop_duplicates("municipality_id", keep="first")
        .rename(
            columns={
                "Codice_Regione": "istat_region_code",
                "Dizione_Regione": "istat_region",
            }
        )[
            [
                "municipality_id",
                "istat_region_code",
                "istat_region",
                "istat_macroarea_code",
                "istat_macroarea",
            ]
        ]
        .sort_values("municipality_id", kind="stable")
        .reset_index(drop=True)
    )
    geography["municipality_id"] = geography["municipality_id"].astype("int64")
    geography["istat_macroarea_code"] = geography["istat_macroarea_code"].astype(
        "int8"
    )
    expected_codes = geography["istat_macroarea"].map(ISTAT_MACROAREA_CODES)
    if not expected_codes.eq(geography["istat_macroarea_code"]).all():
        raise ValueError("The registry contains inconsistent ISTAT macro-area labels/codes")
    return geography


def load_municipality_istat_codes(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> pd.DataFrame:
    """Load municipality fiscal-code to ISTAT municipality-code candidates.

    Multiple codes are deliberately retained for fiscal codes whose municipality
    changed over time. A dated boundary file can then select the code that exists
    in that particular map vintage.
    """
    registry = pd.read_csv(
        registry_path,
        sep=";",
        encoding="latin-1",
        dtype="string",
        usecols=["CF", "Codice_ISTAT_Comune"],
    )
    registry["municipality_id"] = pd.to_numeric(
        registry["CF"].str.strip().str.replace(r"\.0+$", "", regex=True),
        errors="coerce",
    ).astype("Int64")
    registry["istat_municipality_code"] = (
        registry["Codice_ISTAT_Comune"].str.strip().str.zfill(6)
    )
    valid_code = registry["istat_municipality_code"].str.fullmatch(r"\d{6}", na=False)
    codes = (
        registry.loc[
            registry["municipality_id"].notna() & valid_code,
            ["municipality_id", "istat_municipality_code"],
        ]
        .drop_duplicates()
        .sort_values(
            ["municipality_id", "istat_municipality_code"], kind="stable"
        )
        .reset_index(drop=True)
    )
    codes["municipality_id"] = codes["municipality_id"].astype("int64")
    return codes


def normalize_municipality_label(value: object) -> str:
    """Normalize municipality names for conservative exact-name matching."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode().upper()
    text = re.sub(r"^COMUNE\s+DI\s+", "", text)
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def match_municipalities_to_boundaries(
    municipalities: pd.DataFrame,
    boundaries: pd.DataFrame,
    registry_codes: pd.DataFrame,
) -> pd.DataFrame:
    """Return a conservative one-to-one municipality-to-boundary crosswalk.

    Registry ISTAT codes are used first. Conflicting code matches are retained
    only when the municipality and boundary names agree. Remaining records are
    matched by normalized name within the same ISTAT region, which handles
    isolated bad or historical registry codes without allowing cross-region
    homonyms.
    """
    municipality_columns = (
        "municipality_id",
        "municipality_name",
        "istat_region_code",
    )
    boundary_columns = (
        "istat_municipality_code",
        "shape_name",
        "istat_region_code",
    )
    code_columns = ("municipality_id", "istat_municipality_code")
    missing_municipality = set(municipality_columns) - set(municipalities.columns)
    missing_boundary = set(boundary_columns) - set(boundaries.columns)
    missing_code = set(code_columns) - set(registry_codes.columns)
    if missing_municipality or missing_boundary or missing_code:
        raise ValueError(
            "Boundary matching inputs are missing columns: "
            f"municipalities={sorted(missing_municipality)}, "
            f"boundaries={sorted(missing_boundary)}, "
            f"registry_codes={sorted(missing_code)}"
        )

    municipality_view = municipalities[list(municipality_columns)].copy()
    boundary_view = boundaries[list(boundary_columns)].copy()
    code_view = registry_codes[list(code_columns)].copy()
    if municipality_view["municipality_id"].duplicated().any():
        raise ValueError("Municipality rows must be unique by municipality_id")
    if boundary_view["istat_municipality_code"].duplicated().any():
        raise ValueError("Boundary rows must be unique by ISTAT municipality code")

    municipality_view["municipality_id"] = pd.to_numeric(
        municipality_view["municipality_id"], errors="raise"
    ).astype("int64")
    code_view["municipality_id"] = pd.to_numeric(
        code_view["municipality_id"], errors="raise"
    ).astype("int64")
    for frame in (municipality_view, boundary_view):
        frame["istat_region_code"] = (
            frame["istat_region_code"].astype("string").str.strip().str.zfill(2)
        )
    boundary_view["istat_municipality_code"] = (
        boundary_view["istat_municipality_code"]
        .astype("string")
        .str.strip()
        .str.zfill(6)
    )
    code_view["istat_municipality_code"] = (
        code_view["istat_municipality_code"]
        .astype("string")
        .str.strip()
        .str.zfill(6)
    )
    municipality_view["normalized_name"] = municipality_view[
        "municipality_name"
    ].map(normalize_municipality_label)
    boundary_view["normalized_name"] = boundary_view["shape_name"].map(
        normalize_municipality_label
    )

    code_candidates = code_view.merge(
        municipality_view,
        on="municipality_id",
        how="inner",
        validate="many_to_one",
    ).merge(
        boundary_view,
        on="istat_municipality_code",
        how="inner",
        suffixes=("_municipality", "_shape"),
        validate="many_to_one",
    )
    code_candidates["name_match"] = code_candidates[
        "normalized_name_municipality"
    ].eq(code_candidates["normalized_name_shape"])
    ambiguous_code = (
        code_candidates["municipality_id"].duplicated(keep=False)
        | code_candidates["istat_municipality_code"].duplicated(keep=False)
    )
    code_matches = pd.concat(
        [
            code_candidates.loc[~ambiguous_code],
            code_candidates.loc[ambiguous_code & code_candidates["name_match"]],
        ],
        ignore_index=True,
    )
    code_matches = code_matches.loc[
        ~code_matches["municipality_id"].duplicated(keep=False)
        & ~code_matches["istat_municipality_code"].duplicated(keep=False)
    ].copy()
    code_matches["match_basis"] = "istat_code"

    unmatched_municipalities = municipality_view.loc[
        ~municipality_view["municipality_id"].isin(code_matches["municipality_id"])
    ]
    unused_boundaries = boundary_view.loc[
        ~boundary_view["istat_municipality_code"].isin(
            code_matches["istat_municipality_code"]
        )
    ]
    name_candidates = unmatched_municipalities.merge(
        unused_boundaries,
        on=["istat_region_code", "normalized_name"],
        how="inner",
        suffixes=("_municipality", "_shape"),
        validate="many_to_many",
    )
    name_matches = name_candidates.loc[
        ~name_candidates["municipality_id"].duplicated(keep=False)
        & ~name_candidates["istat_municipality_code"].duplicated(keep=False)
    ].copy()
    name_matches["match_basis"] = "region_and_name"

    matches = pd.concat(
        [
            code_matches[
                ["municipality_id", "istat_municipality_code", "match_basis"]
            ],
            name_matches[
                ["municipality_id", "istat_municipality_code", "match_basis"]
            ],
        ],
        ignore_index=True,
    ).sort_values("municipality_id", kind="stable")
    if (
        matches["municipality_id"].duplicated().any()
        or matches["istat_municipality_code"].duplicated().any()
    ):
        raise ValueError("Municipality-to-boundary crosswalk is not one-to-one")
    return matches.reset_index(drop=True)


def geography_fingerprint(geography: pd.DataFrame) -> str:
    """Hash the canonical municipality-to-geography mapping used by a run."""
    required = [
        "municipality_id",
        "istat_region_code",
        "istat_region",
        "istat_macroarea_code",
        "istat_macroarea",
    ]
    missing = sorted(set(required) - set(geography.columns))
    if missing:
        raise ValueError(f"Geography mapping is missing columns: {missing}")
    canonical = geography[required].sort_values("municipality_id", kind="stable")
    digest = hashlib.sha256()
    digest.update(canonical.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()
