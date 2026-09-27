"""CPU unit tests for the parameter-matched latent residual DDPM model."""

from __future__ import annotations

import unittest

import torch

from experiments.latent_residual_ddpm.ddpm_model import LatentResidualDDPM
from experiments.latent_residual_fm.fm_model import LatentResidualFM


EXPECTED_PARAMETER_COUNT = 34_946_606
BATCH_SIZE = 2
SEQUENCE_LENGTH = 7
RESIDUAL_DIM = 13
CONDITION_DIM = 1280
NUM_HIDDEN_STATES = 33


def _parameter_count(model: torch.nn.Module, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _dummy_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
]:
    x_t = torch.linspace(
        -1.0,
        1.0,
        steps=BATCH_SIZE * SEQUENCE_LENGTH * RESIDUAL_DIM,
    ).reshape(BATCH_SIZE, SEQUENCE_LENGTH, RESIDUAL_DIM)
    t = torch.tensor([0.001, 1.0], dtype=torch.float32)
    z_quant = torch.linspace(
        0.5,
        -0.5,
        steps=BATCH_SIZE * SEQUENCE_LENGTH * RESIDUAL_DIM,
    ).reshape(BATCH_SIZE, SEQUENCE_LENGTH, RESIDUAL_DIM)
    hidden_base = torch.linspace(
        -0.2,
        0.2,
        steps=BATCH_SIZE * SEQUENCE_LENGTH * CONDITION_DIM,
    ).reshape(BATCH_SIZE, SEQUENCE_LENGTH, CONDITION_DIM)
    hidden_states = [
        hidden_base + float(index) / 100.0
        for index in range(NUM_HIDDEN_STATES)
    ]
    res_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    return x_t, t, z_quant, hidden_states, res_mask


class LatentResidualDDPMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.fm = LatentResidualFM()
        torch.manual_seed(42)
        cls.ddpm = LatentResidualDDPM()

    def test_fm_and_ddpm_total_parameter_counts_are_equal(self) -> None:
        self.assertEqual(
            _parameter_count(self.fm, trainable_only=False),
            _parameter_count(self.ddpm, trainable_only=False),
        )

    def test_fm_and_ddpm_trainable_parameter_counts_are_equal(self) -> None:
        self.assertEqual(
            _parameter_count(self.fm, trainable_only=True),
            _parameter_count(self.ddpm, trainable_only=True),
        )

    def test_both_models_have_expected_parameter_count(self) -> None:
        self.assertEqual(
            _parameter_count(self.fm, trainable_only=False),
            EXPECTED_PARAMETER_COUNT,
        )
        self.assertEqual(
            _parameter_count(self.fm, trainable_only=True),
            EXPECTED_PARAMETER_COUNT,
        )
        self.assertEqual(
            _parameter_count(self.ddpm, trainable_only=False),
            EXPECTED_PARAMETER_COUNT,
        )
        self.assertEqual(
            _parameter_count(self.ddpm, trainable_only=True),
            EXPECTED_PARAMETER_COUNT,
        )

    def test_ordered_parameter_names_shapes_and_numel_match(self) -> None:
        fm_topology = [
            (name, tuple(parameter.shape), parameter.numel())
            for name, parameter in self.fm.named_parameters()
        ]
        ddpm_topology = [
            (name, tuple(parameter.shape), parameter.numel())
            for name, parameter in self.ddpm.named_parameters()
        ]
        self.assertEqual(fm_topology, ddpm_topology)

    def test_output_layer_initialization_matches_fm(self) -> None:
        torch.testing.assert_close(
            self.fm.output_projection.weight,
            self.ddpm.output_projection.weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            self.fm.output_projection.bias,
            self.ddpm.output_projection.bias,
            rtol=0.0,
            atol=0.0,
        )

    def test_layer_mixture_weights(self) -> None:
        weights = self.ddpm.layer_weights()
        self.assertEqual(weights.shape, (NUM_HIDDEN_STATES,))
        self.assertTrue(bool(torch.isfinite(weights).all()))
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
        torch.testing.assert_close(
            weights,
            torch.full_like(weights, 1.0 / NUM_HIDDEN_STATES),
        )
        self.assertTrue(self.ddpm.layer_logits.requires_grad)

    def test_dummy_forward_shape(self) -> None:
        inputs = _dummy_inputs()
        with torch.inference_mode():
            output = self.ddpm(*inputs)
        self.assertEqual(
            output.shape,
            (BATCH_SIZE, SEQUENCE_LENGTH, RESIDUAL_DIM),
        )

    def test_dummy_forward_is_finite(self) -> None:
        inputs = _dummy_inputs()
        with torch.inference_mode():
            output = self.ddpm(*inputs)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_padding_output_is_exactly_zero(self) -> None:
        inputs = _dummy_inputs()
        with torch.inference_mode():
            output = self.ddpm(*inputs)
        padding = ~inputs[-1]
        torch.testing.assert_close(
            output[padding],
            torch.zeros_like(output[padding]),
            rtol=0.0,
            atol=0.0,
        )

    def test_valid_output_is_invariant_to_padding_values(self) -> None:
        x_t, t, z_quant, hidden_states, res_mask = _dummy_inputs()
        changed_x_t = x_t.clone()
        changed_z_quant = z_quant.clone()
        changed_hidden_states = [state.clone() for state in hidden_states]
        padding = ~res_mask
        changed_x_t[padding] = 1_000_000.0
        changed_z_quant[padding] = -1_000_000.0
        for index, state in enumerate(changed_hidden_states):
            state[padding] = 1_000_000.0 + index

        with torch.inference_mode():
            original = self.ddpm(x_t, t, z_quant, hidden_states, res_mask)
            changed = self.ddpm(
                changed_x_t,
                t,
                changed_z_quant,
                changed_hidden_states,
                res_mask,
            )
        torch.testing.assert_close(original[res_mask], changed[res_mask])

    def test_backward_gradients_exist_and_are_finite(self) -> None:
        self.ddpm.zero_grad(set_to_none=True)
        x_t, t, z_quant, hidden_states, res_mask = _dummy_inputs()
        epsilon_pred = self.ddpm(x_t, t, z_quant, hidden_states, res_mask)
        target = torch.full_like(epsilon_pred, 0.125)
        loss = (
            (epsilon_pred - target).square() * res_mask[..., None]
        ).sum() / (res_mask.sum() * RESIDUAL_DIM)
        loss.backward()

        required_parameters = [
            "layer_logits",
            "z_quant_projection.weight",
            "condition_projection.weight",
            "time_mlp.0.weight",
            "input_projection.weight",
            "blocks.0.modulation.weight",
            "blocks.0.mlp.0.weight",
            "output_norm.weight",
            "output_projection.weight",
        ]
        named_parameters = dict(self.ddpm.named_parameters())
        for name in required_parameters:
            gradient = named_parameters[name].grad
            self.assertIsNotNone(gradient, msg=f"missing gradient for {name}")
            self.assertTrue(
                bool(torch.isfinite(gradient).all()),
                msg=f"non-finite gradient for {name}",
            )

    def test_wrong_residual_dimension_fails(self) -> None:
        _, t, _, hidden_states, res_mask = _dummy_inputs()
        wrong_x_t = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH, RESIDUAL_DIM - 1)
        wrong_z_quant = torch.zeros_like(wrong_x_t)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            self.ddpm(wrong_x_t, t, wrong_z_quant, hidden_states, res_mask)

    def test_wrong_z_quant_shape_fails(self) -> None:
        x_t, t, z_quant, hidden_states, res_mask = _dummy_inputs()
        with self.assertRaisesRegex(ValueError, "same shape"):
            self.ddpm(x_t, t, z_quant[:, :-1], hidden_states, res_mask)

    def test_wrong_hidden_state_count_fails(self) -> None:
        x_t, t, z_quant, hidden_states, res_mask = _dummy_inputs()
        with self.assertRaisesRegex(ValueError, "expected 33 Transformer outputs"):
            self.ddpm(x_t, t, z_quant, hidden_states[:-1], res_mask)

    def test_wrong_hidden_state_width_or_batch_length_fails(self) -> None:
        x_t, t, z_quant, hidden_states, res_mask = _dummy_inputs()
        hidden_states[0] = torch.zeros(
            BATCH_SIZE, SEQUENCE_LENGTH, CONDITION_DIM - 1
        )
        with self.assertRaisesRegex(ValueError, r"hidden_states\[0\] must have shape"):
            self.ddpm(x_t, t, z_quant, hidden_states, res_mask)

        _, _, _, hidden_states, _ = _dummy_inputs()
        hidden_states[0] = torch.zeros(
            BATCH_SIZE, SEQUENCE_LENGTH - 1, CONDITION_DIM
        )
        with self.assertRaisesRegex(ValueError, r"hidden_states\[0\] must have shape"):
            self.ddpm(x_t, t, z_quant, hidden_states, res_mask)

    def test_wrong_mask_or_time_shape_fails(self) -> None:
        x_t, t, z_quant, hidden_states, res_mask = _dummy_inputs()
        with self.assertRaisesRegex(ValueError, "res_mask must have shape"):
            self.ddpm(x_t, t, z_quant, hidden_states, res_mask[:, :-1])
        with self.assertRaisesRegex(ValueError, "t must have shape"):
            self.ddpm(x_t, t[:1], z_quant, hidden_states, res_mask)


if __name__ == "__main__":
    unittest.main()
