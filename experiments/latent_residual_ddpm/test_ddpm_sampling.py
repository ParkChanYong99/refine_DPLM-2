"""Unit tests for the full ancestral Local Matched Residual DDPM sampler."""

from __future__ import annotations

import unittest

import torch

from experiments.latent_residual_ddpm.ddpm_diffusion import DDPMSchedule
from experiments.latent_residual_ddpm.ddpm_sampling import (
    ddpm_reverse_step,
    sample_residual_ddpm,
)


class RecordingEpsilonModel(torch.nn.Module):
    """Small deterministic epsilon model with the production call signature."""

    def __init__(self, epsilon_value: float = 0.0) -> None:
        super().__init__()
        self.epsilon_value = epsilon_value
        self.received_tau: list[torch.Tensor] = []

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_quant: torch.Tensor,
        hidden_states: list[torch.Tensor],
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.received_tau.append(t.detach().clone())
        return torch.full_like(x_t, self.epsilon_value)


class PointwiseEpsilonModel(torch.nn.Module):
    """Residue-wise mock that makes masking of its x_t input observable."""

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_quant: torch.Tensor,
        hidden_states: list[torch.Tensor],
        res_mask: torch.Tensor,
    ) -> torch.Tensor:
        del t
        return 0.05 * x_t + 0.01 * z_quant + 0.02 * hidden_states[0][..., :13]


def make_inputs(
    batch: int = 2,
    length: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]:
    x_t = torch.linspace(-1.0, 1.0, batch * length * 13, dtype=torch.float32)
    x_t = x_t.view(batch, length, 13)
    z_quant = torch.linspace(0.5, -0.5, batch * length * 13, dtype=torch.float32)
    z_quant = z_quant.view(batch, length, 13)
    hidden_states = [
        torch.full((batch, length, 13), index / 100.0, dtype=torch.float32)
        for index in range(33)
    ]
    res_mask = torch.ones(batch, length, dtype=torch.float32)
    res_mask[:, -1] = 0.0
    return x_t, z_quant, hidden_states, res_mask


class ReverseStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = DDPMSchedule(num_timesteps=10)
        self.x_t, self.z_quant, self.hidden_states, self.res_mask = make_inputs()

    def run_step(
        self,
        timestep: int,
        *,
        model: torch.nn.Module | None = None,
        noise: torch.Tensor | None = None,
        x_t: torch.Tensor | None = None,
        z_quant: torch.Tensor | None = None,
        hidden_states: list[torch.Tensor] | None = None,
        res_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return ddpm_reverse_step(
            model=model or RecordingEpsilonModel(0.25),
            schedule=self.schedule,
            x_t=self.x_t if x_t is None else x_t,
            timestep=timestep,
            z_quant=self.z_quant if z_quant is None else z_quant,
            hidden_states=self.hidden_states if hidden_states is None else hidden_states,
            res_mask=self.res_mask if res_mask is None else res_mask,
            noise=noise,
        )

    def test_mathematical_endpoint_indices(self) -> None:
        endpoints = torch.tensor([1, self.schedule.num_timesteps], dtype=torch.long)
        indices = self.schedule.mathematical_timesteps_to_indices(endpoints)
        self.assertEqual(indices.tolist(), [0, self.schedule.num_timesteps - 1])

    def test_tau_endpoints(self) -> None:
        model = RecordingEpsilonModel()
        self.run_step(1, model=model, noise=torch.ones_like(self.x_t))
        self.run_step(self.schedule.num_timesteps, model=model, noise=torch.zeros_like(self.x_t))
        torch.testing.assert_close(
            model.received_tau[0],
            torch.full((self.x_t.shape[0],), 1.0 / self.schedule.num_timesteps),
        )
        torch.testing.assert_close(
            model.received_tau[1], torch.ones(self.x_t.shape[0])
        )

    def test_posterior_mean_matches_manual_equation(self) -> None:
        timestep = 7
        model = RecordingEpsilonModel(0.25)
        result = self.run_step(timestep, model=model, noise=torch.zeros_like(self.x_t))
        index = timestep - 1
        mask = self.res_mask[..., None]
        epsilon = torch.full_like(self.x_t, 0.25) * mask
        alpha = self.schedule.alphas[index].to(dtype=self.x_t.dtype)
        beta = self.schedule.betas[index].to(dtype=self.x_t.dtype)
        denominator = self.schedule.sqrt_one_minus_alpha_bars[index].to(
            dtype=self.x_t.dtype
        )
        expected = ((self.x_t * mask) - beta / denominator * epsilon) / torch.sqrt(alpha)
        expected = expected * mask
        torch.testing.assert_close(result["posterior_mean"], expected)

    def test_posterior_variance_uses_schedule_buffer(self) -> None:
        timestep = 6
        result = self.run_step(timestep, noise=torch.zeros_like(self.x_t))
        expected = self.schedule.posterior_variance[timestep - 1].to(self.x_t.dtype)
        torch.testing.assert_close(
            result["posterior_variance"],
            expected.expand(self.x_t.shape[0], 1, 1),
        )

    def test_t_greater_than_one_zero_noise_equals_mean(self) -> None:
        result = self.run_step(5, noise=torch.zeros_like(self.x_t))
        torch.testing.assert_close(result["x_prev"], result["posterior_mean"])

    def test_t_greater_than_one_nonzero_noise_matches_manual(self) -> None:
        noise = torch.full_like(self.x_t, 0.75)
        result = self.run_step(5, noise=noise)
        expected = result["posterior_mean"] + torch.sqrt(
            result["posterior_variance"]
        ) * noise * self.res_mask[..., None]
        expected = expected * self.res_mask[..., None]
        torch.testing.assert_close(result["x_prev"], expected)

    def test_t_one_ignores_noise_and_returns_mean(self) -> None:
        first = self.run_step(1, noise=torch.full_like(self.x_t, 1000.0))
        second = self.run_step(1, noise=torch.full((1,), float("nan")))
        torch.testing.assert_close(first["x_prev"], first["posterior_mean"])
        torch.testing.assert_close(second["x_prev"], second["posterior_mean"])
        torch.testing.assert_close(first["x_prev"], second["x_prev"])

    def test_padding_positions_remain_zero(self) -> None:
        result = self.run_step(8, noise=torch.ones_like(self.x_t))
        padding = self.res_mask == 0
        self.assertTrue(torch.equal(result["epsilon_pred"][padding], torch.zeros_like(result["epsilon_pred"][padding])))
        self.assertTrue(torch.equal(result["posterior_mean"][padding], torch.zeros_like(result["posterior_mean"][padding])))
        self.assertTrue(torch.equal(result["x_prev"][padding], torch.zeros_like(result["x_prev"][padding])))

    def test_padding_x_input_does_not_change_valid_output(self) -> None:
        changed_x = self.x_t.clone()
        changed_x[self.res_mask == 0] = 1.0e6
        noise = torch.full_like(self.x_t, 0.2)
        first = self.run_step(8, model=PointwiseEpsilonModel(), noise=noise)
        second = self.run_step(
            8, model=PointwiseEpsilonModel(), noise=noise, x_t=changed_x
        )
        valid = self.res_mask.bool()
        torch.testing.assert_close(first["x_prev"][valid], second["x_prev"][valid])

    def test_float32_and_finite_outputs(self) -> None:
        result = self.run_step(9, noise=torch.ones_like(self.x_t))
        for name in ("x_prev", "posterior_mean", "posterior_variance", "epsilon_pred", "tau"):
            self.assertEqual(result[name].dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(result[name]).all()))

    def test_invalid_timestep_endpoints_raise(self) -> None:
        with self.assertRaises(ValueError):
            self.run_step(0)
        with self.assertRaises(ValueError):
            self.run_step(self.schedule.num_timesteps + 1)

    def test_wrong_noise_shape_raises_for_stochastic_step(self) -> None:
        with self.assertRaises(ValueError):
            self.run_step(2, noise=torch.zeros(1, 1, 13))

    def test_wrong_mask_and_input_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            self.run_step(2, res_mask=torch.ones(2, 3))
        with self.assertRaises(ValueError):
            self.run_step(2, x_t=torch.zeros(2, 4, 12))
        with self.assertRaises(ValueError):
            self.run_step(2, z_quant=torch.zeros(2, 3, 13))


class FullChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x_t, self.z_quant, self.hidden_states, self.res_mask = make_inputs(
            batch=1, length=3
        )
        self.initial_noise = torch.linspace(
            -0.7, 0.7, self.z_quant.numel(), dtype=torch.float32
        ).view_as(self.z_quant)

    def sample(
        self,
        *,
        seed: int,
        schedule: DDPMSchedule | None = None,
        initial_noise: torch.Tensor | None = None,
        return_metadata: bool = False,
    ):
        generator = torch.Generator(device=self.z_quant.device)
        generator.manual_seed(seed)
        return sample_residual_ddpm(
            model=RecordingEpsilonModel(),
            schedule=schedule or DDPMSchedule(num_timesteps=8),
            z_quant=self.z_quant,
            hidden_states=self.hidden_states,
            res_mask=self.res_mask,
            initial_noise=self.initial_noise if initial_noise is None else initial_noise,
            generator=generator,
            return_metadata=return_metadata,
        )

    def test_primary_schedule_has_exactly_1000_nfe_in_reverse_order(self) -> None:
        model = RecordingEpsilonModel()
        output, metadata = sample_residual_ddpm(
            model=model,
            schedule=DDPMSchedule(),
            z_quant=self.z_quant,
            hidden_states=self.hidden_states,
            res_mask=self.res_mask,
            initial_noise=self.initial_noise,
            generator=torch.Generator().manual_seed(42),
            return_metadata=True,
        )
        self.assertEqual(metadata["nfe"], 1000)
        self.assertEqual(len(model.received_tau), 1000)
        self.assertEqual(metadata["timesteps"], tuple(range(1000, 0, -1)))
        self.assertEqual(tuple(output.shape), (1, 3, 13))

    def test_same_initial_and_generator_seed_is_exactly_reproducible(self) -> None:
        first = self.sample(seed=123)
        second = self.sample(seed=123)
        self.assertTrue(torch.equal(first, second))

    def test_different_generator_seed_changes_valid_output(self) -> None:
        first = self.sample(seed=123)
        second = self.sample(seed=124)
        valid = self.res_mask.bool()
        self.assertFalse(torch.equal(first[valid], second[valid]))

    def test_generated_initial_state_uses_generator_without_global_rng(self) -> None:
        global_state = torch.random.get_rng_state().clone()
        generated = self.sample(
            seed=321,
            initial_noise=None,
        )
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        self.assertEqual(tuple(generated.shape), tuple(self.z_quant.shape))

    def test_full_chain_padding_zero_finite_and_model_eval(self) -> None:
        model = RecordingEpsilonModel()
        model.train()
        generator = torch.Generator().manual_seed(7)
        output = sample_residual_ddpm(
            model=model,
            schedule=DDPMSchedule(num_timesteps=8),
            z_quant=self.z_quant,
            hidden_states=self.hidden_states,
            res_mask=self.res_mask,
            initial_noise=self.initial_noise,
            generator=generator,
        )
        self.assertFalse(model.training)
        self.assertEqual(tuple(output.shape), (1, 3, 13))
        self.assertTrue(torch.equal(output[self.res_mask == 0], torch.zeros_like(output[self.res_mask == 0])))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_bad_initial_noise_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.sample(seed=42, initial_noise=torch.zeros(1, 2, 13))


if __name__ == "__main__":
    unittest.main()
