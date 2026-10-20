# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import numpy as np
import pytest

from prophet.models import CmdStanPyBackend


class TestCmdStanPyJsonInputs:
    write = staticmethod(CmdStanPyBackend._write_json_inputs)

    def test_values_round_trip_exactly(self, tmp_path):
        data = {"T": 3, "y": [0.1, 1e-17, -2.5, 1 / 3], "X": [[1.0, 2.0], [3.0, 4.0]], "tau": 0.05}
        inits = {"k": 0.3, "delta": [0.0, 1e300]}
        data_path, inits_path = self.write(data, inits, str(tmp_path))
        for path, obj in ((data_path, data), (inits_path, inits)):
            with open(path) as f:
                assert json.load(f) == obj

    def test_matches_stanio_bytes(self, tmp_path):
        stanio = pytest.importorskip("stanio")
        data = {"T": 3, "y": [0.1, 1e-17, -2.5], "X": [[1.0, 2.0], [3.0, 4.0]], "tau": 0.05}
        inits = {"k": 0.3, "delta": [0.0, 1e300]}
        data_path, inits_path = self.write(data, inits, str(tmp_path))
        for path, obj in ((data_path, data), (inits_path, inits)):
            ref = tmp_path / "ref.json"
            stanio.write_stan_json(str(ref), obj)
            with open(path) as f:
                assert f.read() == ref.read_text()

    def test_numpy_scalars_are_written(self, tmp_path):
        data_path, _ = self.write({"T": np.int64(2), "y": [np.float64(1.5), 2.0]}, {}, str(tmp_path))
        with open(data_path) as f:
            assert f.read() == '{"T": 2, "y": [1.5, 2.0]}'

    def test_falls_back_to_dicts_on_nan(self, tmp_path):
        data = {"y": [1.0, float("nan")]}
        inits = {"k": 0.0}
        assert self.write(data, inits, str(tmp_path)) == (data, inits)
