"""CPU unit tests for one Local Matched Residual DDPM training batch."""

from __future__ import annotations

from typing import Sequence
import unittest

import torch
from torch import nn

from experiments.latent_residual_ddpm.ddpm_diffusion import (
    DEFAULT_NUM_TIMESTEPS,
    RESIDUAL_DIM,
    DDPMSchedule,
    masked_epsilon_mse,
)
from experiments.latent_residual_ddpm.ddpm_model import LatentResidualDDPM
from experiments.latent_residual_ddpm.ddpm_training import ddpm_training_step


BATCH_SIZE = 2
SEQUENCE_LENGTH = 7
CONDITION_DIM = 1280
NUM_HIDDEN_STATES = 33


class ResiduewiseMock(nn.Module):
    """Small pointwise epsilon model with the production call signature."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_quant: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        del t
        mask = res_mask.to(dtype=x_t.dtype)[..., None]
        prediction = (
            self.scale * x_t
            + 0.1 * z_quant
            + 0.01 * hidden_states[0][..., :RESIDUAL_DIM]
        )
        return prediction * mask


class ZeroEpsilonMock(nn.Module):
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_quant: torch.Tensor,
        hidden_states: Sequence[torch.Tensor],
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        del t, z_quant, hidden_states, res_mask
        return torch.zeros_like(x_t)


def _dummy_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x0 = torch.linspace(
        -1.0,
        1.0,
        steps=BATCH_SIZE * SEQUENCE_LENGTH * RESIDUAL_DIM,
        dtype=torch.float32,
    ).reshape(BATCH_SIZE, SEQUENCE_LENGTH, RESIDUAL_DIM)
    z_quant = torch.linspace(
        0.5,
        -0.5,
        steps=BATCH_SIZE * SEQUENCE_LENGTH * RESIDUAL_DIM,
        dtype=torch.float32,
    ).reshape(BATCH_SIZE, SEQUENCE_LENGTH, RESIDUAL_DIM)
    hidden_base = torch.linspace(
        -0.2,
        0.2,
        steps=BATCH_SIZE * SEQUENCE_LENGTH * CONDITION_DIM,
        dtype=torch.float32,
    ).reshape(BATCH_SIZE, SEQUENCE_LENGTH, CONDITION_DIM)
    hidden_states = [
        hidden_base + float(index) / 100.0
        for index in range(NUM_HIDDEN_STATES)
    ]
    res_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    timesteps = torch.tensor([1, DEFAULT_NUM_TIMESTEPS], dtype=torch.long)
    noise = torch.linspace(
        -0.75,
        0.75,
        steps=BATCH_SIZE * SEQUENCE_LENGTH * RESIDUAL_DIM,
        dtype=torch.float32,
    ).reshape(BATCH_SIZE, SEQUENCE_LENGTH, RESIDUAL_DIM)
    noise = noise * res_mask[..., None]
    return x0, z_quant, hidden_states, res_mask, timesteps, noise


class DDPMTrainingStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = DDPMSchedule()
        torch.manual_seed(42)
        cls.actual_model = LatentResidualDDPM()

    def setUp(self) -> None:
        (
            self.x0,
            self.z_quant,
            self.hidden_states,
            self.res_mask,
            self.timesteps,
            self.noise,
        ) = _dummy_inputs()
        self.model = ResiduewiseMock()

    def _explicit_step(self) -> dict[str, torch.Tensor]:
        return ddpm_training_step(
            model=self.model,
            schedule=self.schedule,
            x0=self.x0,
            z_quant=self.z_quant,
            hidden_states=self.hidden_states,
            res_mask=self.res_mask,
            timesteps=self.timesteps,
            noise=self.noise,
        )

    def test_explicit_timestep_and_noise_step_runs(self) -> None:
        result = self._explicit_step()
        self.assertEqual(
            set(result),
            {"loss", "x_t", "epsilon", "epsilon_pred", "timesteps", "tau"},
        )

    def test_loss_is_finite_scalar(self) -> None:
        loss = self._explicit_step()["loss"]
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_x_t_and_epsilon_prediction_shapes(self) -> None:
        result = self._explicit_step()
        expected_shape = (BATCH_SIZE, SEQUENCE_LENGTH, RESIDUAL_DIM)
        self.assertEqual(result["x_t"].shape, expected_shape)
        self.assertEqual(result["epsilon_pred"].shape, expected_shape)

    def test_timestep_shape_and_dtype(self) -> None:
        returned = self._explicit_step()["timesteps"]
        self.assertEqual(returned.shape, (BATCH_SIZE,))
        self.assertEqual(returned.dtype, torch.long)
        torch.testing.assert_close(returned, self.timesteps)

    def test_tau_shape_and_endpoints(self) -> None:
        tau = self._explicit_step()["tau"]
        self.assertEqual(tau.shape, (BATCH_SIZE,))
        self.assertEqual(tau.dtype, self.x0.dtype)
        torch.testing.assert_close(
            tau,
            torch.tensor([1.0 / DEFAULT_NUM_TIMESTEPS, 1.0]),
        )

    def test_float32_dtype_is_preserved(self) -> None:
        result = self._explicit_step()
        self.assertEqual(result["x_t"].dtype, torch.float32)
        self.assertEqual(result["epsilon"].dtype, self.x0.dtype)
        self.assertEqual(result["epsilon_pred"].dtype, self.x0.dtype)

    def test_q_sample_formula_matches_independent_calculation(self) -> None:
        result = self._explicit_step()
        indices = self.timesteps - 1
        sqrt_alpha_bar = self.schedule.sqrt_alpha_bars[indices].to(
            dtype=self.x0.dtype
        ).view(BATCH_SIZE, 1, 1)
        sqrt_one_minus_alpha_bar = self.schedule.sqrt_one_minus_alpha_bars[
            indices
        ].to(dtype=self.x0.dtype).view(BATCH_SIZE, 1, 1)
        mask = self.res_mask.to(dtype=self.x0.dtype)[..., None]
        expected = (
            sqrt_alpha_bar * (self.x0 * mask)
            + sqrt_one_minus_alpha_bar * (self.noise * mask)
        ) * mask
        torch.testing.assert_close(result["x_t"], expected)

    def test_returned_epsilon_is_explicit_noise(self) -> None:
        epsilon = self._explicit_step()["epsilon"]
        torch.testing.assert_close(epsilon, self.noise, rtol=0.0, atol=0.0)

    def test_loss_matches_direct_masked_epsilon_mse(self) -> None:
        result = self._explicit_step()
        expected = masked_epsilon_mse(
            result["epsilon_pred"], result["epsilon"], self.res_mask
        )
        torch.testing.assert_close(result["loss"], expected, rtol=0.0, atol=0.0)

    def test_padding_outputs_are_zero(self) -> None:
        result = self._explicit_step()
        padding = ~self.res_mask
        torch.testing.assert_close(
            result["x_t"][padding],
            torch.zeros_like(result["x_t"][padding]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            result["epsilon"][padding],
            torch.zeros_like(result["epsilon"][padding]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            result["epsilon_pred"][padding],
            torch.zeros_like(result["epsilon_pred"][padding]),
            rtol=0.0,
            atol=0.0,
        )

    def test_padding_values_do_not_change_valid_output_or_loss(self) -> None:
        original = self._explicit_step()
        padding = ~self.res_mask
        changed_x0 = self.x0.clone()
        changed_z_quant = self.z_quant.clone()
        changed_noise = self.noise.clone()
        changed_hidden = [state.clone() for state in self.hidden_states]
        changed_x0[padding] = 1_000_000.0
        changed_z_quant[padding] = -1_000_000.0
        changed_noise[padding] = 500_000.0
        for index, state in enumerate(changed_hidden):
            state[padding] = 1_000_000.0 + index

        changed = ddpm_training_step(
            model=self.model,
            schedule=self.schedule,
            x0=changed_x0,
            z_quant=changed_z_quant,
            hidden_states=changed_hidden,
            res_mask=self.res_mask,
            timesteps=self.timesteps,
            noise=changed_noise,
        )
        torch.testing.assert_close(
            original["epsilon_pred"][self.res_mask],
            changed["epsilon_pred"][self.res_mask],
        )
        torch.testing.assert_close(original["loss"], changed["loss"])

    def test_perfect_zero_epsilon_prediction_has_zero_loss(self) -> None:
        result = ddpm_training_step(
            model=ZeroEpsilonMock(),
            schedule=self.schedule,
            x0=self.x0,
            z_quant=self.z_quant,
            hidden_states=self.hidden_states,
            res_mask=self.res_mask,
            timesteps=self.timesteps,
            noise=torch.zeros_like(self.x0),
        )
        self.assertEqual(float(result["loss"]), 0.0)

    def test_training_step_supports_backward(self) -> None:
        result = self._explicit_step()
        result["loss"].backward()
        self.assertIsNotNone(self.model.scale.grad)
        self.assertTrue(bool(torch.isfinite(self.model.scale.grad)))

    def test_actual_ddpm_parameter_gradients_are_finite(self) -> None:
        self.actual_model.zero_grad(set_to_none=True)
        length = 2
        x0 = torch.linspace(-0.5, 0.5, steps=length * RESIDUAL_DIM).reshape(
            1, length, RESIDUAL_DIM
        )
        z_quant = torch.linspace(0.25, -0.25, steps=length * RESIDUAL_DIM).reshape(
            1, length, RESIDUAL_DIM
        )
        hidden_base = torch.linspace(
            -0.1, 0.1, steps=length * CONDITION_DIM
        ).reshape(1, length, CONDITION_DIM)
        hidden_states = [
            hidden_base + float(index) / 1000.0
            for index in range(NUM_HIDDEN_STATES)
        ]
        res_mask = torch.ones(1, length, dtype=torch.bool)
        noise = torch.linspace(-1.0, 1.0, steps=length * RESIDUAL_DIM).reshape(
            1, length, RESIDUAL_DIM
        )
        result = ddpm_training_step(
            model=self.actual_model,
            schedule=self.schedule,
            x0=x0,
            z_quant=z_quant,
            hidden_states=hidden_states,
            res_mask=res_mask,
            timesteps=torch.tensor([500], dtype=torch.long),
            noise=noise,
        )
        result["loss"].backward()
        for name, parameter in self.actual_model.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=f"missing gradient for {name}")
            self.assertTrue(
                bool(torch.isfinite(parameter.grad).all()),
                msg=f"non-finite gradient for {name}",
            )

    def test_none_inputs_sample_valid_timesteps_and_noise(self) -> None:
        torch.manual_seed(42)
        result = ddpm_training_step(
            model=self.model,
            schedule=self.schedule,
            x0=self.x0,
            z_quant=self.z_quant,
            hidden_states=self.hidden_states,
            res_mask=self.res_mask,
        )
        self.assertEqual(result["timesteps"].dtype, torch.long)
        self.assertEqual(result["timesteps"].shape, (BATCH_SIZE,))
        self.assertTrue(bool((result["timesteps"] >= 1).all()))
        self.assertTrue(
            bool((result["timesteps"] <= DEFAULT_NUM_TIMESTEPS).all())
        )
        self.assertEqual(result["epsilon"].dtype, self.x0.dtype)
        self.assertTrue(bool(torch.isfinite(result["epsilon"]).all()))

    def test_invalid_timestep_bounds_fail_through_q_sample(self) -> None:
        for invalid in (0, DEFAULT_NUM_TIMESTEPS + 1):
            with self.subTest(timestep=invalid):
                with self.assertRaisesRegex(ValueError, "must lie in"):
                    ddpm_training_step(
                        model=self.model,
                        schedule=self.schedule,
                        x0=self.x0,
                        z_quant=self.z_quant,
                        hidden_states=self.hidden_states,
                        res_mask=self.res_mask,
                        timesteps=torch.tensor([invalid, 1]),
                        noise=self.noise,
                    )

    def test_shape_mismatches_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "noise must have the same shape"):
            ddpm_training_step(
                model=self.model,
                schedule=self.schedule,
                x0=self.x0,
                z_quant=self.z_quant,
                hidden_states=self.hidden_states,
                res_mask=self.res_mask,
                timesteps=self.timesteps,
                noise=self.noise[:, :-1],
            )
        with self.assertRaisesRegex(ValueError, "res_mask must have shape"):
            ddpm_training_step(
                model=self.model,
                schedule=self.schedule,
                x0=self.x0,
                z_quant=self.z_quant,
                hidden_states=self.hidden_states,
                res_mask=self.res_mask[:, :-1],
                timesteps=self.timesteps,
                noise=self.noise,
            )
        with self.assertRaisesRegex(ValueError, "timesteps must have shape"):
            ddpm_training_step(
                model=self.model,
                schedule=self.schedule,
                x0=self.x0,
                z_quant=self.z_quant,
                hidden_states=self.hidden_states,
                res_mask=self.res_mask,
                timesteps=self.timesteps[:1],
                noise=self.noise,
            )


if __name__ == "__main__":
    unittest.main()
