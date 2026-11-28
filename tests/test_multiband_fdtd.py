import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multiband_fdtd.engine import (
    BandResult, FrequencyBand, reconstruct_response, run_multiband,
    run_multiband_auto,
)
from multiband_fdtd.planner import BandCost, select_fastest_band_sequence
from multiband_fdtd.characterization import (
    characterize_frequency_intervals, generate_candidate_bands,
    characterize_and_select_intervals, load_interval_plan,
    _load_cached_interval_plan, _store_cached_interval_plan,
)



class ActiveVariableTests(unittest.TestCase):
    @patch("multiband_fdtd.engine._run_band")
    def test_forwards_candidate_variables_to_every_band(self, run_band):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\n", encoding="utf-8")
            frequencies = np.linspace(0.1e9, 1.0e9, 2)
            run_band.return_value = BandResult(
                FrequencyBand(0.1, 1.0, 2),
                frequencies,
                np.array([0.2 + 0j, 0.3 + 0j]),
                np.array([50.0 + 0j, 55.0 + 0j]),
                1.0, 1000, "mock",
            )
            active_vars = {"arm_length": 0.042}
            run_multiband(
                config,
                Path(temp) / "run",
                [FrequencyBand(0.1, 1.0, 2)],
                profile="optimization",
                active_vars=active_vars,
                record_far_field=True,
            )
            run_band.assert_called_once()
            self.assertEqual(
                run_band.call_args.kwargs["active_vars"], active_vars
            )
            self.assertTrue(run_band.call_args.kwargs["record_far_field"])


class MultiBandPlannerTests(unittest.TestCase):
    def test_selects_fastest_measured_sequence(self):
        candidates = [
            BandCost(0.1, 0.6, 4.0),
            BandCost(0.5, 1.5, 3.0),
            BandCost(1.4, 3.0, 4.0),
            BandCost(0.1, 1.5, 12.0),
        ]
        selected = select_fastest_band_sequence(
            candidates, (0.1, 3.0), minimum_overlap_ghz=0.1
        )
        self.assertEqual(
            [(x.fstart_ghz, x.fstop_ghz) for x in selected],
            [(0.1, 0.6), (0.5, 1.5), (1.4, 3.0)],
        )

    def test_rejects_gap(self):
        candidates = [BandCost(0.1, 0.5, 2.0), BandCost(0.6, 1.0, 2.0)]
        with self.assertRaises(ValueError):
            select_fastest_band_sequence(candidates, (0.1, 1.0))

    def test_reconstructs_overlapping_complex_responses(self):
        bands = []
        for low, high in ((0.1, 0.6), (0.5, 1.0)):
            freq = np.linspace(low, high, 101) * 1e9
            response = 0.3 * np.exp(1j * 2 * np.pi * freq / 1e9)
            bands.append(BandResult(
                FrequencyBand(low, high, 101), freq, response,
                np.full(freq.shape, 75.0 + 0j), 1.0, None, "test",
            ))
        result = reconstruct_response(bands, (0.1, 1.0), 181)
        self.assertEqual(len(result["freq"]), 181)
        self.assertTrue(np.all(np.isfinite(result["s11"])))
        self.assertIn("s11_raw", result)
        self.assertTrue(np.allclose(result["s11"], result["s11_raw"]))
        self.assertTrue(np.allclose(np.abs(result["s11"]), 0.3, atol=1e-3))


class CandidateGenerationTests(unittest.TestCase):
    def test_generates_grid_candidates_without_full_range(self):
        bands = generate_candidate_bands((0.1, 1.1), interval_widths_ghz=(0.5, 1.0), boundary_step_ghz=0.25, n_pts=41)
        pairs = {(b.fstart_ghz, b.fstop_ghz) for b in bands}
        self.assertNotIn((0.1, 1.1), pairs)
        self.assertIn((0.35, 0.85), pairs)
        self.assertTrue(all(b.n_pts == 41 for b in bands))
        self.assertTrue(all(0.1 <= b.fstart_ghz < b.fstop_ghz <= 1.1 for b in bands))

    def test_candidate_limit_without_full_range(self):
        bands = generate_candidate_bands((0.1, 3.0), interval_widths_ghz=(0.5, 1.0, 1.5), boundary_step_ghz=0.1, max_candidates=7)
        self.assertLessEqual(len(bands), 7)
        self.assertNotIn((0.1, 3.0), {(b.fstart_ghz, b.fstop_ghz) for b in bands})

    def test_single_candidate_limit(self):
        bands = generate_candidate_bands((0.1, 1.0), max_candidates=1)
        self.assertEqual(len(bands), 1)
        self.assertLess(bands[0].fstop_ghz - bands[0].fstart_ghz, 1.0)

    def test_rejects_invalid_generation_settings(self):
        with self.assertRaises(ValueError):
            generate_candidate_bands((0.0, 1.0))
        with self.assertRaises(ValueError):
            generate_candidate_bands((0.1, 1.0), interval_widths_ghz=())
        with self.assertRaises(ValueError):
            generate_candidate_bands((0.1, 1.0), max_candidates=0)


class IntegratedPlanningTests(unittest.TestCase):
    @patch("multiband_fdtd.characterization.characterize_frequency_intervals")
    def test_measures_candidates_selects_and_persists_plan(self, characterize):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "characterization.json"
            manifest = {"intervals": []}
            def fake_characterize(config, output, bands, **kwargs):
                records = [
                    BandCost(
                        band.fstart_ghz, band.fstop_ghz,
                        1.0 + (band.fstop_ghz - band.fstart_ghz),
                        1000, None,
                    )
                    for band in bands
                ]
                return {
                    "records": records,
                    "manifest": manifest,
                    "manifest_path": str(manifest_path),
                    "output_dir": str(temp),
                }
            characterize.side_effect = fake_characterize
            result = characterize_and_select_intervals(
                "design.yaml", Path(temp) / "unused",
                (0.1, 1.1),
                interval_widths_ghz=(0.5, 1.0),
                boundary_step_ghz=0.25,
                n_pts=31,
                max_candidates=20,
                minimum_overlap_ghz=0.1,
            )
            self.assertGreater(len(result["candidates"]), 0)
            self.assertTrue(result["selected"])
            self.assertTrue(result["selected_bands"])
            self.assertTrue(all(isinstance(x, FrequencyBand) for x in result["selected_bands"]))
            self.assertEqual(
                result["selection"]["frequency_range_ghz"], [0.1, 1.1]
            )
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("selection", saved)
            self.assertEqual(
                len(saved["selection"]["selected_intervals"]),
                len(result["selected"]),
            )
            characterize.assert_called_once()



class IntervalPlanCacheTests(unittest.TestCase):
    def test_round_trips_a_compatible_cached_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\\n", encoding="utf-8")
            digest = hashlib.sha256(config.read_bytes()).hexdigest()
            cache_path = Path(temp) / "cache" / ("a" * 64 + ".json")
            manifest = {
                "plan_version": 1,
                "config_sha256": digest,
                "candidate_generation": {"n_pts": 41},
                "intervals": [
                    {
                        "fstart_ghz": 0.1, "fstop_ghz": 0.6,
                        "runtime_seconds": 1.0, "mesh_cells": 1000,
                    },
                    {
                        "fstart_ghz": 0.5, "fstop_ghz": 1.0,
                        "runtime_seconds": 1.2, "mesh_cells": 1200,
                    },
                ],
                "selection": {
                    "frequency_range_ghz": [0.1, 1.0],
                    "selected_intervals": [
                        {
                            "fstart_ghz": 0.1, "fstop_ghz": 0.6,
                            "runtime_seconds": 1.0, "mesh_cells": 1000,
                        },
                        {
                            "fstart_ghz": 0.5, "fstop_ghz": 1.0,
                            "runtime_seconds": 1.2, "mesh_cells": 1200,
                        },
                    ],
                },
            }
            _store_cached_interval_plan(cache_path, manifest)
            loaded = _load_cached_interval_plan(
                cache_path, config_path=config,
                fingerprint=cache_path.stem, n_pts=41,
            )
            self.assertIsNotNone(loaded)
            self.assertEqual(
                [(b.fstart_ghz, b.fstop_ghz)
                 for b in loaded["selected_bands"]],
                [(0.1, 0.6), (0.5, 1.0)],
            )
            self.assertEqual(len(loaded["records"]), 2)

    def test_rejects_cache_for_changed_design(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\\n", encoding="utf-8")
            digest = hashlib.sha256(config.read_bytes()).hexdigest()
            cache_path = Path(temp) / "cache" / ("b" * 64 + ".json")
            manifest = {
                "plan_version": 1,
                "config_sha256": digest,
                "candidate_generation": {"n_pts": 41},
                "intervals": [
                    {
                        "fstart_ghz": 0.1, "fstop_ghz": 1.0,
                        "runtime_seconds": 1.0, "mesh_cells": 1000,
                    }
                ],
                "selection": {
                    "frequency_range_ghz": [0.1, 1.0],
                    "selected_intervals": [
                        {
                            "fstart_ghz": 0.1, "fstop_ghz": 1.0,
                            "runtime_seconds": 1.0, "mesh_cells": 1000,
                        }
                    ],
                },
            }
            _store_cached_interval_plan(cache_path, manifest)
            config.write_text(
                "variables: {arm_length: {default: 0.2}}\\n",
                encoding="utf-8",
            )
            loaded = _load_cached_interval_plan(
                cache_path, config_path=config,
                fingerprint=cache_path.stem, n_pts=41,
            )
            self.assertIsNone(loaded)

    def test_cache_write_is_atomic_and_creates_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_path = Path(temp) / "nested" / "plan.json"
            _store_cached_interval_plan(cache_path, {"plan_version": 1})
            self.assertTrue(cache_path.is_file())
            self.assertFalse(list(cache_path.parent.glob("*.tmp")))


    @patch("multiband_fdtd.characterization._interval_plan_fingerprint",
           return_value="c" * 64)
    @patch("multiband_fdtd.characterization.characterize_frequency_intervals")
    def test_second_request_reuses_plan_without_running_pilots(
        self, characterize, fingerprint
    ):
        from multiband_fdtd.characterization import BandCost

        def fake_characterize(config, output, bands, **kwargs):
            output = Path(output)
            output.mkdir(parents=True, exist_ok=False)
            records = [
                BandCost(
                    band.fstart_ghz, band.fstop_ghz, 1.0,
                    1000, None,
                )
                for band in bands
            ]
            digest = hashlib.sha256(Path(config).read_bytes()).hexdigest()
            details = [
                {
                    "fstart_ghz": item.fstart_ghz,
                    "fstop_ghz": item.fstop_ghz,
                    "runtime_seconds": item.runtime_seconds,
                    "mesh_cells": item.mesh_cells,
                    "n_pts": 41,
                    "output_dir": "pilot",
                }
                for item in records
            ]
            manifest = {
                "plan_version": 1,
                "config_path": str(Path(config).resolve()),
                "config_sha256": digest,
                "end_criteria_db": kwargs["end_criteria_db"],
                "max_timesteps": kwargs["max_timesteps"],
                "threads": kwargs["threads"],
                "max_pilot_mesh_cells": kwargs["max_pilot_mesh_cells"],
                "intervals": details,
                "skipped_intervals": [],
            }
            manifest_path = output / "characterization.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            return {
                "records": records,
                "manifest": manifest,
                "manifest_path": str(manifest_path),
                "output_dir": str(output),
            }

        characterize.side_effect = fake_characterize
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\\n", encoding="utf-8")
            cache = Path(temp) / "cache"
            common = {
                "frequency_range_ghz": (0.1, 1.1),
                "interval_widths_ghz": (0.5, 1.0),
                "boundary_step_ghz": 0.25,
                "n_pts": 41,
                "minimum_overlap_ghz": 0.1,
                "cache_dir": cache,
            }
            first = characterize_and_select_intervals(
                config, Path(temp) / "run1", **common
            )
            second = characterize_and_select_intervals(
                config, Path(temp) / "run2", **common
            )
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(len(second["selected_bands"]), len(first["selected_bands"]))
            self.assertTrue(Path(second["manifest_path"]).is_file())
            self.assertEqual(characterize.call_count, 1)
            fingerprint.assert_called()

class FrozenIntervalPlanTests(unittest.TestCase):
    def _write_plan(self, directory, intervals):
        config = Path(directory) / "design.yaml"
        config.write_text("variables: {}\n", encoding="utf-8")
        digest = hashlib.sha256(config.read_bytes()).hexdigest()
        plan_path = Path(directory) / "characterization.json"
        plan_path.write_text(json.dumps({
            "plan_version": 1,
            "config_sha256": digest,
            "candidate_generation": {"n_pts": 41},
            "selection": {
                "frequency_range_ghz": [0.1, 1.0],
                "selected_intervals": intervals,
            },
        }), encoding="utf-8")
        return config, plan_path

    def test_loads_frozen_boundaries_and_resamples_extraction_density(self):
        with tempfile.TemporaryDirectory() as temp:
            config, plan = self._write_plan(temp, [
                {"fstart_ghz": 0.1, "fstop_ghz": 0.6},
                {"fstart_ghz": 0.5, "fstop_ghz": 1.0},
            ])
            bands = load_interval_plan(plan, n_points=301, config_path=config)
            self.assertEqual(
                [(b.fstart_ghz, b.fstop_ghz) for b in bands],
                [(0.1, 0.6), (0.5, 1.0)],
            )
            self.assertEqual(
                [b.n_pts for b in bands],
                [168, 168],
            )

    def test_rejects_plan_for_different_design(self):
        with tempfile.TemporaryDirectory() as temp:
            config, plan = self._write_plan(temp, [
                {"fstart_ghz": 0.1, "fstop_ghz": 1.0},
            ])
            config.write_text("variables: {arm_length: {default: 0.05}}\n",
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different design"):
                load_interval_plan(plan, config_path=config)

    def test_rejects_frequency_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            config, plan = self._write_plan(temp, [
                {"fstart_ghz": 0.1, "fstop_ghz": 0.4},
                {"fstart_ghz": 0.5, "fstop_ghz": 1.0},
            ])
            with self.assertRaisesRegex(ValueError, "gap"):
                load_interval_plan(plan, config_path=config)

class CharacterizationTests(unittest.TestCase):
    @patch("multiband_fdtd.characterization._run_band")
    def test_records_reference_interval_measurements(self, run_band):
        class FakeResult:
            def __init__(self, runtime, cells, output):
                self.runtime_seconds = runtime
                self.cells = cells
                self.output_dir = str(output)

        run_band.side_effect = [
            FakeResult(1.25, 12000, "pilot_001"),
            FakeResult(2.50, 18000, "pilot_002"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\n", encoding="utf-8")
            output = Path(temp) / "measurements"
            result = characterize_frequency_intervals(
                config, output,
                [FrequencyBand(0.1, 0.6, 41), FrequencyBand(0.5, 1.0, 41)],
                threads=2,
            )
            self.assertEqual(len(result["records"]), 2)
            self.assertEqual(result["records"][0].runtime_seconds, 1.25)
            self.assertEqual(result["records"][1].mesh_cells, 18000)
            self.assertTrue(Path(result["manifest_path"]).is_file())
            saved = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(len(saved["intervals"]), 2)
            self.assertEqual(saved["threads"], 2)
            self.assertEqual(run_band.call_count, 2)

    @patch("multiband_fdtd.characterization._run_band")
    def test_skips_candidate_over_mesh_limit_and_records_reason(self, run_band):
        from multiband_fdtd.engine import PilotMeshLimitExceeded

        class FakeResult:
            runtime_seconds = 1.0
            cells = 1000
            output_dir = "pilot_002"

        run_band.side_effect = [
            PilotMeshLimitExceeded("Mesh contains 3,000,000 cells, exceeding the 2,000,000-cell pilot limit.", 3_000_000),
            FakeResult(),
        ]
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\n", encoding="utf-8")
            result = characterize_frequency_intervals(
                config, Path(temp) / "measurements",
                [FrequencyBand(0.1, 1.0, 41), FrequencyBand(0.5, 1.0, 41)],
            )
            self.assertEqual(len(result["records"]), 1)
            saved = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(len(saved["skipped_intervals"]), 1)
            self.assertEqual(saved["skipped_intervals"][0]["mesh_cells"], 3_000_000)
            self.assertIn("exceeding", saved["skipped_intervals"][0]["reason"])
            self.assertEqual(run_band.call_count, 2)

    @patch("multiband_fdtd.characterization._run_band")
    def test_honors_stop_request_between_pilot_runs(self, run_band):
        class FakeResult:
            runtime_seconds = 1.0
            cells = 1000
            output_dir = "pilot_001"

        run_band.return_value = FakeResult()
        requests = iter((False, True))
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(InterruptedError, "interrupted"):
                characterize_frequency_intervals(
                    config,
                    Path(temp) / "measurements",
                    [FrequencyBand(0.1, 0.6, 41), FrequencyBand(0.5, 1.0, 41)],
                    stop_requested=lambda: next(requests),
                )
        self.assertEqual(run_band.call_count, 1)

    def test_rejects_empty_candidate_list(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                characterize_frequency_intervals(config, Path(temp) / "out", [])



class AutomaticRunTests(unittest.TestCase):
    @patch("multiband_fdtd.engine.run_multiband")
    def test_builds_equal_width_plan_without_pilot_simulations(self, run_production):
        run_production.return_value = {
            "output_dir": "/tmp/automatic_run/simulation",
            "profile": "accurate",
            "freq": np.array([1.0]),
        }
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\\n", encoding="utf-8")
            output = Path(temp) / "automatic_run"
            result = run_multiband_auto(
                config, output, (0.1, 3.0), profile="accurate",
                end_criteria_db=-35.0, max_timesteps=700000,
                n_points=801, threads=4, smoothing_sigma_points=1.5,
                record_far_field=True,
            )

        bands = result["selected_bands"]
        self.assertEqual(len(bands), 6)
        self.assertEqual(sum(band.n_pts for band in bands), 801)
        self.assertTrue(all(band.fstop_ghz - band.fstart_ghz <= 0.5 + 1e-12
                            for band in bands))
        self.assertAlmostEqual(bands[0].fstart_ghz, 0.1)
        self.assertAlmostEqual(bands[-1].fstop_ghz, 3.0)
        self.assertTrue(all(
            abs(left.fstop_ghz - right.fstart_ghz) < 1e-12
            for left, right in zip(bands, bands[1:])
        ))
        kwargs = run_production.call_args.kwargs
        self.assertEqual(Path(kwargs["output_dir"]), output / "simulation")
        self.assertEqual(kwargs["bands"], bands)
        self.assertEqual(kwargs["profile"], "accurate")
        self.assertEqual(kwargs["end_criteria_db"], -35.0)
        self.assertEqual(kwargs["max_timesteps"], 700000)
        self.assertEqual(kwargs["n_points"], 801)
        self.assertEqual(kwargs["threads"], 4)
        self.assertEqual(kwargs["smoothing_sigma_points"], 1.5)
        self.assertTrue(kwargs["record_far_field"])
        self.assertTrue((output / "automatic_run.json").is_file())

    @patch("multiband_fdtd.engine.run_multiband")
    def test_uses_profile_defaults(self, run_production):
        run_production.return_value = {"output_dir": "simulation"}
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\\n", encoding="utf-8")
            run_multiband_auto(
                config, Path(temp) / "run", (0.1, 1.0),
                profile="optimization",
            )
        kwargs = run_production.call_args.kwargs
        self.assertEqual(kwargs["end_criteria_db"], -15.0)
        self.assertEqual(kwargs["max_timesteps"], 200000)
        self.assertEqual(kwargs["n_points"], 301)
        self.assertEqual(kwargs["profile"], "optimization")

    def test_rejects_invalid_profile_before_solver(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\\n", encoding="utf-8")
            with patch("multiband_fdtd.engine.run_multiband") as run_production:
                with self.assertRaises(ValueError):
                    run_multiband_auto(
                        config, Path(temp) / "run", (0.1, 1.0),
                        profile="unknown",
                    )
                run_production.assert_not_called()


class EqualWidthIntervalPlannerTests(unittest.TestCase):
    def test_ceil_band_count_and_even_point_distribution(self):
        from multiband_fdtd import build_equal_width_frequency_bands

        bands = build_equal_width_frequency_bands((0.1, 3.0), 301)
        self.assertEqual(len(bands), 6)
        self.assertEqual([band.n_pts for band in bands], [51, 50, 50, 50, 50, 50])
        self.assertEqual(sum(band.n_pts for band in bands), 301)
        self.assertTrue(all(band.fstop_ghz - band.fstart_ghz <= 0.5 + 1e-12
                            for band in bands))

    def test_rejects_insufficient_samples_for_interval_count(self):
        from multiband_fdtd import build_equal_width_frequency_bands

        with self.assertRaisesRegex(ValueError, "insufficient"):
            build_equal_width_frequency_bands((0.1, 3.0), 10)


if __name__ == "__main__":
    unittest.main()
