from __future__ import annotations

import pytest
import torch

from dggt.utils.tokenizer_checkpoint import load_scene_tokenizer_state_dict_strict


def _tokenizer() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.LayerNorm(4),
    )


def _filled_state(module: torch.nn.Module, value: float) -> dict[str, torch.Tensor]:
    return {
        key: torch.full_like(tensor, value)
        for key, tensor in module.state_dict().items()
    }


def test_loads_tokenizer_trainer_payload_strictly() -> None:
    tokenizer = _tokenizer()
    expected = _filled_state(tokenizer, 3.0)

    load_scene_tokenizer_state_dict_strict(
        tokenizer,
        {"scene_tokenizer": expected, "optimizer": {"ignored": True}},
        source="tokenizer.pt",
    )

    for key, value in tokenizer.state_dict().items():
        assert torch.equal(value, expected[key])


def test_loads_embedded_ddp_tokenizer_state_strictly() -> None:
    tokenizer = _tokenizer()
    expected = _filled_state(tokenizer, 5.0)
    full_state = {
        "module.aggregator.weight": torch.ones(1),
        **{f"module.scene_tokenizer.{key}": value for key, value in expected.items()},
    }

    load_scene_tokenizer_state_dict_strict(
        tokenizer,
        {"state_dict": full_state},
        source="dggt.pt",
    )

    for key, value in tokenizer.state_dict().items():
        assert torch.equal(value, expected[key])


def test_rejects_full_dggt_checkpoint_without_tokenizer_and_preserves_parameters() -> None:
    tokenizer = _tokenizer()
    before = {key: value.clone() for key, value in tokenizer.state_dict().items()}

    with pytest.raises(RuntimeError, match="randomly initialized tokenizer is forbidden"):
        load_scene_tokenizer_state_dict_strict(
            tokenizer,
            {"state_dict": {"aggregator.weight": torch.ones(1)}},
            source="legacy_dggt.pt",
        )

    for key, value in tokenizer.state_dict().items():
        assert torch.equal(value, before[key])


def test_rejects_partial_or_wrong_shape_tokenizer_before_loading() -> None:
    tokenizer = _tokenizer()
    before = {key: value.clone() for key, value in tokenizer.state_dict().items()}
    invalid = _filled_state(tokenizer, 7.0)
    invalid.pop("1.bias")
    invalid["0.weight"] = torch.ones(1, 1)

    with pytest.raises(RuntimeError, match="shape_mismatches=1"):
        load_scene_tokenizer_state_dict_strict(
            tokenizer,
            {"scene_tokenizer": invalid},
            source="partial_tokenizer.pt",
        )

    for key, value in tokenizer.state_dict().items():
        assert torch.equal(value, before[key])
