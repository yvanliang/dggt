from __future__ import annotations

import torch

from dggt.models.scene_flow import WanSceneFlow


def _tiny_model(**kwargs) -> WanSceneFlow:
    defaults = dict(
        patch_grid=(2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=27,
        out_channels=8,
        text_dim=12,
        ffn_dim=32,
        num_layers=4,
        base_model_depth=1,
        ddt_head_dim=16,
        ddt_head_heads=2,
        ddt_head_depth=2,
        rope_max_seq_len=8,
        camera_gen_dim=10,
    )
    defaults.update(kwargs)
    return WanSceneFlow(**defaults)


def _checkpoint_test_loss(model: WanSceneFlow, z: torch.Tensor) -> torch.Tensor:
    masks = torch.zeros(z.shape[:-1] + (1,), dtype=z.dtype)
    output = model(
        z,
        torch.tensor([0.5], dtype=z.dtype),
        torch.zeros_like(z),
        torch.zeros_like(z),
        masks,
        masks,
        torch.ones_like(masks),
        torch.zeros(1, 0, 12, dtype=z.dtype),
    )
    return output.square().mean()


def test_half_gradient_checkpointing_selects_alternating_blocks() -> None:
    model = _tiny_model()

    model.enable_gradient_checkpointing()
    assert model.checkpointed_block_indices(len(model.blocks)) == (0, 1, 2, 3)
    assert model.checkpointed_block_indices(len(model.ddt_head), block_group="ddt") == (0, 1)

    model.enable_half_gradient_checkpointing()
    assert model.gradient_checkpointing is True
    assert model.gradient_checkpointing_mode == "half"
    assert model.checkpointed_block_indices(len(model.blocks)) == (0, 2)
    assert model.checkpointed_block_indices(len(model.ddt_head), block_group="ddt") == (0,)

    model.enable_three_quarter_gradient_checkpointing()
    assert model.gradient_checkpointing_mode == "three_quarter"
    assert model.checkpointed_block_indices(len(model.blocks)) == (0, 1, 2)
    assert model.checkpointed_block_indices(len(model.ddt_head), block_group="ddt") == ()

    model.disable_gradient_checkpointing()
    assert model.checkpointed_block_indices(len(model.blocks)) == ()
    assert model.checkpointed_block_indices(len(model.ddt_head), block_group="ddt") == ()


def test_three_quarter_gradient_checkpointing_invokes_selected_blocks(monkeypatch) -> None:
    model = _tiny_model()
    model.enable_three_quarter_gradient_checkpointing()
    original_checkpoint = torch.utils.checkpoint.checkpoint
    checkpoint_calls = []

    def counted_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(function)
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counted_checkpoint)
    z = torch.randn(1, 2, 4, 8, requires_grad=True)

    _checkpoint_test_loss(model, z).backward()

    assert len(checkpoint_calls) == 3


def test_three_quarter_gradient_checkpointing_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(123)
    reference = _tiny_model()
    three_quarter = _tiny_model()
    three_quarter.load_state_dict(reference.state_dict(), strict=True)
    three_quarter.enable_three_quarter_gradient_checkpointing()
    z_reference = torch.randn(1, 2, 4, 8, requires_grad=True)
    z_three_quarter = z_reference.detach().clone().requires_grad_(True)

    reference_loss = _checkpoint_test_loss(reference, z_reference)
    three_quarter_loss = _checkpoint_test_loss(three_quarter, z_three_quarter)
    reference_loss.backward()
    three_quarter_loss.backward()

    assert torch.equal(reference_loss.detach(), three_quarter_loss.detach())
    assert torch.allclose(z_reference.grad, z_three_quarter.grad, rtol=1e-5, atol=1e-6)
    reference_grads = dict(reference.named_parameters())
    for name, parameter in three_quarter.named_parameters():
        reference_grad = reference_grads[name].grad
        if reference_grad is None or parameter.grad is None:
            assert reference_grad is None and parameter.grad is None, name
            continue
        assert torch.allclose(
            reference_grad,
            parameter.grad,
            rtol=1e-5,
            atol=1e-6,
        ), name
