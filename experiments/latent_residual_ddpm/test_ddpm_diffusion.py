"""CPU unit tests for the Local Matched Residual DDPM math core."""

from __future__ import annotations

import unittest

import torch

from experiments.latent_residual_ddpm.ddpm_diffusion import (
    DEFAULT_BETA_END,
    DEFAULT_BETA_START,
    DEFAULT_NUM_TIMESTEPS,
    RESIDUAL_DIM,
    DDPMSchedule,
    masked_epsilon_mse,
    q_sample,
)


class DDPMScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = DDPMSchedule()

    def test_default_schedule_shapes_and_endpoints(self) -> None:
        expected_names = {
            "betas",
            "alphas",
            "alpha_bars",
            "alpha_bars_prev",
            "posterior_variance",
            "sqrt_alpha_bars",
            "sqrt_one_minus_alpha_bars",
        }
        self.assertEqual(set(dict(self.schedule.named_buffers())), expected_names)
        for buffer in self.schedule.buffers():
            self.assertEqual(buffer.shape, (DEFAULT_NUM_TIMESTEPS,))
            self.assertEqual(buffer.dtype, torch.float64)
            self.assertTrue(bool(torch.isfinite(buffer).all()))
        torch.testing.assert_close(
            self.schedule.betas[0],
            torch.tensor(DEFAULT_BETA_START, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            self.schedule.betas[-1],
            torch.tensor(DEFAULT_BETA_END, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_schedule_algebra_monotonicity_and_posterior_t1(self) -> None:
        torch.testing.assert_close(
            self.schedule.alphas,
            1.0 - self.schedule.betas,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            self.schedule.alpha_bars,
            torch.cumprod(self.schedule.alphas, dim=0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(
            bool((self.schedule.alpha_bars[1:] < self.schedule.alpha_bars[:-1]).all())
        )
        self.assertEqual(float(self.schedule.alpha_bars_prev[0]), 1.0)
        self.assertEqual(float(self.schedule.posterior_variance[0]), 0.0)
        self.assertTrue(bool((self.schedule.posterior_variance >= 0.0).all()))

    def test_schedule_has_no_parameters_and_buffers_can_be_cast(self) -> None:
        self.assertEqual(list(self.schedule.parameters()), [])
        cast_schedule = self.schedule.to(dtype=torch.float32)
        for buffer in cast_schedule.buffers():
            self.assertEqual(buffer.dtype, torch.float32)

    def test_mathematical_timestep_to_python_index_mapping(self) -> None:
        timesteps = torch.tensor([1, 2, DEFAULT_NUM_TIMESTEPS], dtype=torch.long)
        expected = torch.tensor([0, 1, DEFAULT_NUM_TIMESTEPS - 1])
        torch.testing.assert_close(
            self.schedule.mathematical_timesteps_to_indices(timesteps), expected
        )

    def test_invalid_schedule_and_timestep_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            DDPMSchedule(num_timesteps=0)
        with self.assertRaises(ValueError):
            DDPMSchedule(beta_start=0.0)
        with self.assertRaises(ValueError):
            self.schedule.mathematical_timesteps_to_indices(torch.tensor([0]))
        with self.assertRaises(ValueError):
            self.schedule.mathematical_timesteps_to_indices(
                torch.tensor([DEFAULT_NUM_TIMESTEPS + 1])
            )
        with self.assertRaises(TypeError):
            self.schedule.mathematical_timesteps_to_indices(torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            self.schedule.mathematical_timesteps_to_indices(torch.tensor([[1]]))


class QSampleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.schedule = DDPMSchedule()
        self.x0 = torch.arange(2 * 7 * RESIDUAL_DIM, dtype=torch.float32).reshape(
            2, 7, RESIDUAL_DIM
        ) / 100.0
        self.noise = torch.full_like(self.x0, 0.25)

    def test_output_shapes_and_explicit_epsilon_target(self) -> None:
        output = q_sample(
            self.x0,
            torch.tensor([1, DEFAULT_NUM_TIMESTEPS]),
            self.noise,
            self.schedule,
        )
        self.assertEqual(output["x_t"].shape, self.x0.shape)
        self.assertEqual(output["epsilon"].shape, self.x0.shape)
        self.assertEqual(output["sqrt_alpha_bar"].shape, (2, 1, 1))
        self.assertEqual(output["sqrt_one_minus_alpha_bar"].shape, (2, 1, 1))
        torch.testing.assert_close(output["epsilon"], self.noise)

    def test_t1_uses_first_alpha_bar(self) -> None:
        output = q_sample(
            self.x0[:1], torch.tensor([1]), self.noise[:1], self.schedule
        )
        expected = (
            self.schedule.sqrt_alpha_bars[0].float() * self.x0[:1]
            + self.schedule.sqrt_one_minus_alpha_bars[0].float() * self.noise[:1]
        )
        torch.testing.assert_close(output["x_t"], expected)

    def test_tmax_uses_last_alpha_bar(self) -> None:
        output = q_sample(
            self.x0[:1],
            torch.tensor([DEFAULT_NUM_TIMESTEPS]),
            self.noise[:1],
            self.schedule,
        )
        expected = (
            self.schedule.sqrt_alpha_bars[-1].float() * self.x0[:1]
            + self.schedule.sqrt_one_minus_alpha_bars[-1].float() * self.noise[:1]
        )
        torch.testing.assert_close(output["x_t"], expected)

    def test_zero_noise_reduces_to_scaled_clean_sample(self) -> None:
        timesteps = torch.tensor([17, 999])
        output = q_sample(
            self.x0, timesteps, torch.zeros_like(self.x0), self.schedule
        )
        expected = output["sqrt_alpha_bar"] * self.x0
        torch.testing.assert_close(output["x_t"], expected)
        torch.testing.assert_close(output["epsilon"], torch.zeros_like(self.x0))

    def test_zero_x0_reduces_to_scaled_noise(self) -> None:
        timesteps = torch.tensor([17, 999])
        zero_x0 = torch.zeros_like(self.x0)
        output = q_sample(zero_x0, timesteps, self.noise, self.schedule)
        expected = output["sqrt_one_minus_alpha_bar"] * self.noise
        torch.testing.assert_close(output["x_t"], expected)

    def test_padding_is_zero_in_x_t_and_epsilon(self) -> None:
        mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.bool,
        )
        output = q_sample(
            self.x0,
            torch.tensor([10, 20]),
            self.noise,
            self.schedule,
            res_mask=mask,
        )
        padding = ~mask
        torch.testing.assert_close(
            output["x_t"][padding], torch.zeros_like(output["x_t"][padding])
        )
        torch.testing.assert_close(
            output["epsilon"][padding],
            torch.zeros_like(output["epsilon"][padding]),
        )

    def test_invalid_shapes_and_timesteps_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "x0 must have shape"):
            q_sample(
                torch.zeros(2, 7, 12),
                torch.tensor([1, 2]),
                torch.zeros(2, 7, 12),
                self.schedule,
            )
        with self.assertRaisesRegex(ValueError, "same shape"):
            q_sample(
                self.x0,
                torch.tensor([1, 2]),
                torch.zeros(2, 6, RESIDUAL_DIM),
                self.schedule,
            )
        with self.assertRaisesRegex(ValueError, "timesteps must have shape"):
            q_sample(self.x0, torch.tensor([1]), self.noise, self.schedule)
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            q_sample(self.x0, torch.tensor([1.0, 2.0]), self.noise, self.schedule)
        with self.assertRaisesRegex(ValueError, "res_mask must have shape"):
            q_sample(
                self.x0,
                torch.tensor([1, 2]),
                self.noise,
                self.schedule,
                res_mask=torch.ones(2, 6),
            )


class MaskedEpsilonMSETest(unittest.TestCase):
    def test_matches_hand_calculation(self) -> None:
        target = torch.zeros(1, 3, RESIDUAL_DIM)
        prediction = torch.zeros_like(target)
        prediction[0, 0, :] = 1.0
        prediction[0, 1, :] = 2.0
        prediction[0, 2, :] = 100.0
        mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        loss = masked_epsilon_mse(prediction, target, mask)
        expected = torch.tensor((1.0**2 + 2.0**2) / 2.0)
        torch.testing.assert_close(loss, expected)

    def test_padding_invariance(self) -> None:
        target = torch.zeros(1, 4, RESIDUAL_DIM)
        prediction_a = torch.ones_like(target)
        prediction_b = prediction_a.clone()
        prediction_b[:, 2:, :] = 1_000_000.0
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        loss_a = masked_epsilon_mse(prediction_a, target, mask)
        loss_b = masked_epsilon_mse(prediction_b, target, mask)
        torch.testing.assert_close(loss_a, loss_b, rtol=0.0, atol=0.0)

    def test_perfect_prediction_has_zero_loss(self) -> None:
        epsilon = torch.arange(
            2 * 5 * RESIDUAL_DIM, dtype=torch.float32
        ).reshape(2, 5, RESIDUAL_DIM)
        mask = torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
        )
        loss = masked_epsilon_mse(epsilon, epsilon.clone(), mask)
        self.assertEqual(float(loss), 0.0)

    def test_invalid_shapes_and_zero_denominator_fail(self) -> None:
        prediction = torch.zeros(1, 3, RESIDUAL_DIM)
        target = torch.zeros_like(prediction)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            masked_epsilon_mse(prediction, torch.zeros(1, 2, RESIDUAL_DIM), torch.ones(1, 3))
        with self.assertRaisesRegex(ValueError, "epsilon_pred must have shape"):
            masked_epsilon_mse(
                torch.zeros(1, 3, 12),
                torch.zeros(1, 3, 12),
                torch.ones(1, 3),
            )
        with self.assertRaisesRegex(ValueError, "denominator is zero"):
            masked_epsilon_mse(prediction, target, torch.zeros(1, 3))


if __name__ == "__main__":
    unittest.main()
