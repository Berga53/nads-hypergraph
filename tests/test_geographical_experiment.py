"""Synthetic checks for ISTAT mapping and municipality-induced experiments."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import run_experiment as grid
from scripts import run_geographical_experiment as geographical
from src.geography import (
    geography_fingerprint,
    load_municipality_geography,
    load_municipality_istat_codes,
    match_municipalities_to_boundaries,
)


class GeographicalExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data" / "processed"
        self.data.mkdir(parents=True)
        pd.DataFrame(
            {
                "CF Comune": [10, 20, 20, 30],
                "CF Partecipata": [100, 100, 200, 200],
                "Quota": [1.0, 1.0, 1.0, 1.0],
            }
        ).to_csv(self.data / "rete_2023.csv", index=False)
        pd.DataFrame(
            {
                "year": [2023, 2023],
                "company_id": [100, 200],
                "company_name": ["A", "B"],
                "company_size_score_0_100": [50, 50],
                "company_size_data_confidence_0_1": [1, 1],
                "company_size_data_quality": ["complete", "complete"],
                "edge_parameter_base_0_1": [0.5, 0.5],
                "edge_parameter_basis": ["observed", "observed"],
            }
        ).to_csv(self.data / "hyperedge_parameters.csv", index=False)
        pd.DataFrame(
            {
                "year": [2023, 2023, 2023],
                "municipality_id": [10, 20, 30],
                "municipality_name": ["X", "Y", "Z"],
                "municipality_tax_code": ["00010", "00020", "00030"],
                "municipality_bdap_id": ["010", "020", "030"],
                "population": [10, 20, 30],
                "population_data_available": [True, True, True],
                "node_weight_population": [10, 20, 30],
                "node_weight_basis": ["observed", "observed", "observed"],
            }
        ).to_csv(self.data / "node_parameters.csv", index=False)
        self.geography = pd.DataFrame(
            {
                "municipality_id": [10, 20, 30],
                "istat_region_code": ["01", "01", "16"],
                "istat_region": ["PIEMONTE", "PIEMONTE", "PUGLIA"],
                "istat_macroarea_code": [1, 1, 4],
                "istat_macroarea": ["NORD-OVEST", "NORD-OVEST", "SUD"],
            }
        )

    def test_registry_duplicates_collapse_only_when_geography_agrees(self):
        registry = self.root / "registry.csv"
        rows = pd.DataFrame(
            {
                "CF": ["00000000010", "00000000010", "00000000020"],
                "Codice_Regione": ["01", "01", "16"],
                "Dizione_Regione": ["PIEMONTE", "PIEMONTE", "PUGLIA"],
                "Codice_Zona": ["1", "1", "4"],
                "Dizione_zona": ["NORD-OVEST", "NORD-OVEST", "SUD"],
            }
        )
        rows.to_csv(registry, sep=";", encoding="latin-1", index=False)
        loaded = load_municipality_geography(registry)
        self.assertEqual(loaded["municipality_id"].tolist(), [10, 20])
        self.assertEqual(len(geography_fingerprint(loaded)), 64)

        rows.loc[1, "Dizione_zona"] = "SUD"
        rows.loc[1, "Codice_Zona"] = "4"
        rows.to_csv(registry, sep=";", encoding="latin-1", index=False)
        with self.assertRaisesRegex(ValueError, "conflicting ISTAT geography"):
            load_municipality_geography(registry)

    def test_map_codes_retain_historical_istat_candidates(self):
        registry = self.root / "registry.csv"
        pd.DataFrame(
            {
                "CF": ["00000000010", "00000000010", "00000000020"],
                "Codice_ISTAT_Comune": ["001001", "001999", "72006"],
            }
        ).to_csv(registry, sep=";", encoding="latin-1", index=False)

        loaded = load_municipality_istat_codes(registry)

        self.assertEqual(
            loaded.to_dict("records"),
            [
                {
                    "municipality_id": 10,
                    "istat_municipality_code": "001001",
                },
                {
                    "municipality_id": 10,
                    "istat_municipality_code": "001999",
                },
                {
                    "municipality_id": 20,
                    "istat_municipality_code": "072006",
                },
            ],
        )

    def test_boundary_matching_resolves_code_collisions_and_name_fallbacks(self):
        municipalities = pd.DataFrame(
            {
                "municipality_id": [10, 20, 30],
                "municipality_name": [
                    "COMUNE DI CAINES",
                    "COMUNE DI RIFIANO",
                    "COMUNE DI NEW-TOWN",
                ],
                "istat_region_code": [21, 21, 1],
            }
        )
        boundaries = pd.DataFrame(
            {
                "istat_municipality_code": ["021014", "021073", "001003"],
                "shape_name": ["Caines", "Rifiano", "New Town"],
                "istat_region_code": [21, 21, 1],
            }
        )
        registry_codes = pd.DataFrame(
            {
                "municipality_id": [10, 20, 30],
                "istat_municipality_code": ["021073", "021073", "099999"],
            }
        )

        matches = match_municipalities_to_boundaries(
            municipalities, boundaries, registry_codes
        )

        self.assertEqual(
            matches.to_dict("records"),
            [
                {
                    "municipality_id": 10,
                    "istat_municipality_code": "021014",
                    "match_basis": "region_and_name",
                },
                {
                    "municipality_id": 20,
                    "istat_municipality_code": "021073",
                    "match_basis": "istat_code",
                },
                {
                    "municipality_id": 30,
                    "istat_municipality_code": "001003",
                    "match_basis": "region_and_name",
                },
            ],
        )

    def test_model_loader_builds_an_induced_network_and_distinct_identity(self):
        config = grid.ExperimentConfig(seed_budget=1)
        full = grid.load_year_model(2023, config, data_dir=self.data)
        north_west = grid.load_year_model(
            2023,
            config,
            data_dir=self.data,
            municipality_ids=[10, 20],
            partition_identity="NORD-OVEST:test",
        )
        self.assertEqual(full["incidence"].shape, (3, 2))
        self.assertEqual(north_west["incidence"].shape, (2, 1))
        self.assertEqual(north_west["incidence_count"], 2)
        self.assertEqual(set(north_west["node_ids"]), {10, 20})
        self.assertNotEqual(full["input_fingerprint"], north_west["input_fingerprint"])
        changed_identity = grid.load_year_model(
            2023,
            config,
            data_dir=self.data,
            municipality_ids=[10, 20],
            partition_identity="NORD-OVEST:changed",
        )
        self.assertNotEqual(
            north_west["input_fingerprint"], changed_identity["input_fingerprint"]
        )

    def test_geographical_runner_checkpoints_and_resumes(self):
        output = self.root / "geographical.csv"
        args = [
            "--years",
            "2023",
            "--macroareas",
            "nord_ovest",
            "--alphas",
            "0.1",
            "--seed-budgets",
            "1",
            "--search-seconds",
            "0.08",
            "--max-iter",
            "100",
            "--output",
            str(output),
        ]
        with (
            patch.object(geographical, "DATA_DIR", self.data),
            patch.object(
                geographical,
                "load_municipality_geography",
                return_value=self.geography,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            geographical.main(args)
            before = output.read_bytes()
            with patch.object(
                geographical, "run_year", side_effect=AssertionError("must resume")
            ):
                geographical.main(args)
        self.assertEqual(output.read_bytes(), before)

        saved = grid.load_results(output)
        geographical._validate_geographical_results(saved)
        self.assertEqual(saved["macro_area"].tolist(), ["NORD-OVEST"])
        metadata = json.loads(saved.loc[0, "graph_metadata"])
        self.assertEqual(metadata["macro_area"], "NORD-OVEST")
        self.assertEqual(metadata["nodes"], 2)
        self.assertEqual(metadata["hyperedges"], 1)
        self.assertEqual(metadata["partition_rule"], "municipality_induced_then_min_edge_size")


if __name__ == "__main__":
    unittest.main()
