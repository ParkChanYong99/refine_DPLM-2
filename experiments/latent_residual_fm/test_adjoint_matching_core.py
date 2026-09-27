"""Synthetic tests for the reusable Adjoint Matching mathematical core."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from experiments.latent_residual_fm.adjoint_matching_core import (
    adjoint_matching_loss,
    adjoint_matching_residual,
    base_drift,
    control_from_velocities,
    deterministic_warm_start,
    lean_adjoint_step,
    memoryless_step,
    sigma_offset,
)


DTYPE = torch.float64
H = 0.025


def _time_on_state(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if t.ndim == 0:
        return t
    return t.reshape(*t.shape, *([1] * (x.ndim - t.ndim)))


def simple_velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return 0.5 * x + _time_on_state(t, x)


class TinyVelocity(nn.Module):
    def __init__(self, dimension: int = 3) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([0.25, -0.15, 0.4], dtype=DTYPE)[:dimension]
        )
        self.bias = nn.Parameter(
            torch.tensor([0.05, -0.02, 0.03], dtype=DTYPE)[:dimension]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias + 0.1 * _time_on_state(t, x)


def test_sigma_offset() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        t = torch.tensor(
            [0.0, H, 0.5, 1.0 - H], dtype=DTYPE, device=device
        )
        actual = sigma_offset(t, H)
        expected = torch.sqrt(2 * (1 - t + H) / (t + H))
        assert actual.dtype == t.dtype and actual.device == t.device
        assert bool(torch.isfinite(actual).all() and (actual > 0).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sigma_offset(torch.tensor([-0.01], dtype=DTYPE), H)


def test_deterministic_warm_start() -> None:
    x0 = torch.tensor([[-2.0, 0.5, 3.0]], dtype=DTYPE)
    expected = x0 + H * simple_velocity(
        x0, torch.tensor(0.0, dtype=DTYPE)
    )
    actual = deterministic_warm_start(x0, H, simple_velocity)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(
        actual,
        deterministic_warm_start(x0, H, simple_velocity),
        rtol=0,
        atol=0,
    )


def test_memoryless_step() -> None:
    x = torch.tensor([[0.2, -1.0, 2.5]], dtype=DTYPE)
    noise = torch.tensor([[0.5, 0.0, -0.25]], dtype=DTYPE)
    t = torch.tensor(0.25, dtype=DTYPE)
    sigma = torch.sqrt(2 * (1 - t + H) / (t + H))
    expected = (
        x
        + H * (2 * simple_velocity(x, t) - x / t)
        + H**0.5 * sigma * noise
    )
    actual = memoryless_step(x, t, H, simple_velocity, noise)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match="t > 0"):
        memoryless_step(x, 0.0, H, simple_velocity, noise)
    with pytest.raises(ValueError, match="t > 0"):
        base_drift(x, 0.0, simple_velocity)


def test_analytic_scalar_adjoint() -> None:
    diagonal = torch.tensor([0.2, -0.35, 0.7, 1.1], dtype=DTYPE)
    x = torch.tensor(
        [0.4, -1.2, 2.0, 0.1], dtype=DTYPE, requires_grad=True
    )
    adjoint = torch.tensor(
        [1.0, -0.5, 0.3, 2.0], dtype=DTYPE, requires_grad=True
    )

    def velocity(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return diagonal * state + time.square()

    for t_value in (1.0, 0.75, 0.5, 0.25, 0.025):
        actual = lean_adjoint_step(
            x, adjoint, t_value, H, velocity
        )
        expected = (
            adjoint.detach()
            + H * (2 * diagonal - 1 / t_value) * adjoint.detach()
        )
        torch.testing.assert_close(
            actual, expected, rtol=1e-13, atol=1e-13
        )
        assert not actual.requires_grad
    assert x.grad is None and adjoint.grad is None


def test_non_symmetric_vjp() -> None:
    matrix = torch.tensor(
        [
            [0.2, 1.1, -0.4],
            [-0.7, 0.3, 0.9],
            [0.5, -1.2, 0.1],
        ],
        dtype=DTYPE,
    )
    x = torch.tensor([0.3, -0.8, 1.4], dtype=DTYPE)
    adjoint = torch.tensor([1.2, -0.6, 0.25], dtype=DTYPE)
    t = 0.5

    class FrozenMatrixVelocity(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.matrix = nn.Parameter(matrix.clone(), requires_grad=False)

        def forward(
            self, state: torch.Tensor, time: torch.Tensor
        ) -> torch.Tensor:
            del time
            return self.matrix @ state

    velocity = FrozenMatrixVelocity()
    actual = lean_adjoint_step(x, adjoint, t, H, velocity)
    jacobian = 2 * matrix - torch.eye(3, dtype=DTYPE) / t
    expected = adjoint + H * (jacobian.T @ adjoint)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    wrong_jvp = adjoint + H * (jacobian @ adjoint)
    assert not torch.allclose(actual, wrong_jvp, rtol=1e-10, atol=1e-10)
    assert velocity.matrix.grad is None


def test_terminal_cost_sign() -> None:
    x = torch.tensor(
        [0.2, -0.7, 1.5], dtype=DTYPE, requires_grad=True
    )
    target = torch.tensor([-0.1, 0.4, 0.8], dtype=DTYPE)
    terminal_cost = 0.5 * (x - target).square().sum()
    actual = torch.autograd.grad(terminal_cost, x)[0]
    expected = x.detach() - target
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, -expected)


def test_am_regression_gradient_routing() -> None:
    torch.manual_seed(11)
    base = TinyVelocity().requires_grad_(False)
    finetune = copy.deepcopy(base).requires_grad_(True)
    for left, right in zip(finetune.parameters(), base.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    x = torch.randn(2, 4, 3, dtype=DTYPE, requires_grad=True)
    adjoint = torch.randn(
        2, 4, 3, dtype=DTYPE, requires_grad=True
    )
    loss = adjoint_matching_loss(
        x,
        adjoint,
        0.5,
        finetune,
        base,
        sigma_offset(torch.tensor(0.5, dtype=DTYPE), H),
    )
    assert bool(torch.isfinite(loss))
    loss.backward()
    gradients = [
        parameter.grad for parameter in finetune.parameters()
    ]
    assert all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gradients
    )
    grad_norm = torch.linalg.vector_norm(
        torch.cat([gradient.reshape(-1) for gradient in gradients])
    )
    assert float(grad_norm) > 0
    assert all(parameter.grad is None for parameter in base.parameters())
    assert x.grad is None and adjoint.grad is None


def test_zero_reward_fixed_point() -> None:
    torch.manual_seed(12)
    base = TinyVelocity().requires_grad_(False)
    finetune = copy.deepcopy(base).requires_grad_(True)
    x = torch.randn(2, 5, 3, dtype=DTYPE)
    adjoint = torch.zeros_like(x)
    loss = adjoint_matching_loss(
        x,
        adjoint,
        0.4,
        finetune,
        base,
        sigma_offset(torch.tensor(0.4, dtype=DTYPE), H),
    )
    torch.testing.assert_close(
        loss, torch.zeros_like(loss), rtol=0, atol=0
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in finetune.parameters()
    ]
    assert all(gradient is not None for gradient in gradients)
    grad_norm = torch.linalg.vector_norm(
        torch.cat([gradient.reshape(-1) for gradient in gradients])
    )
    torch.testing.assert_close(
        grad_norm, torch.zeros_like(grad_norm), rtol=0, atol=1e-14
    )


def test_perturbed_base_regularization() -> None:
    torch.manual_seed(13)
    base = TinyVelocity().requires_grad_(False)
    finetune = copy.deepcopy(base).requires_grad_(True)
    with torch.no_grad():
        finetune.weight.add_(
            torch.tensor([0.01, -0.02, 0.03], dtype=DTYPE)
        )
    x = torch.randn(2, 5, 3, dtype=DTYPE)
    loss = adjoint_matching_loss(
        x,
        torch.zeros_like(x),
        0.35,
        finetune,
        base,
        sigma_offset(torch.tensor(0.35, dtype=DTYPE), H),
    )
    assert bool(torch.isfinite(loss)) and float(loss) > 0


def test_mask_semantics() -> None:
    torch.manual_seed(14)
    x = torch.randn(2, 5, 3, dtype=DTYPE)
    noise = torch.randn_like(x)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=DTYPE
    )
    padded = ~mask.bool()
    dirty_x, dirty_noise = x.clone(), noise.clone()
    dirty_x[padded] = 1e8
    dirty_noise[padded] = -1e9
    clean_x, clean_noise = x.clone(), noise.clone()
    clean_x[padded] = 0
    clean_noise[padded] = 0
    dirty_step = memoryless_step(
        dirty_x, 0.3, H, simple_velocity, dirty_noise, mask
    )
    clean_step = memoryless_step(
        clean_x, 0.3, H, simple_velocity, clean_noise, mask
    )
    torch.testing.assert_close(dirty_step, clean_step, rtol=0, atol=0)
    assert bool((dirty_step[padded] == 0).all())

    adjoint = torch.randn_like(x)
    adjoint[padded] = 1e7
    propagated = lean_adjoint_step(
        dirty_x, adjoint, 0.3, H, simple_velocity, mask
    )
    assert bool((propagated[padded] == 0).all())

    base = TinyVelocity().requires_grad_(False)
    finetune = copy.deepcopy(base).requires_grad_(True)
    sigma = sigma_offset(torch.tensor(0.3, dtype=DTYPE), H)
    clean_loss = adjoint_matching_loss(
        clean_x,
        torch.zeros_like(x),
        0.3,
        finetune,
        base,
        sigma,
        mask,
    )
    dirty_loss = adjoint_matching_loss(
        dirty_x,
        adjoint * padded.unsqueeze(-1),
        0.3,
        finetune,
        base,
        sigma,
        mask,
    )
    torch.testing.assert_close(dirty_loss, clean_loss, rtol=0, atol=0)


def test_stop_gradient_semantics() -> None:
    torch.manual_seed(15)
    base = TinyVelocity().requires_grad_(True)
    finetune = copy.deepcopy(base).requires_grad_(True)
    x = torch.randn(2, 3, 3, dtype=DTYPE, requires_grad=True)
    adjoint = torch.randn(
        2, 3, 3, dtype=DTYPE, requires_grad=True
    )
    sigma = torch.tensor(1.3, dtype=DTYPE, requires_grad=True)
    residual = adjoint_matching_residual(
        x, adjoint, 0.6, finetune, base, sigma
    )
    loss = residual.square().mean()
    loss.backward()
    assert all(
        parameter.grad is not None for parameter in finetune.parameters()
    )
    assert all(parameter.grad is None for parameter in base.parameters())
    assert x.grad is None and adjoint.grad is None and sigma.grad is None

    vf = torch.tensor([1.0, 2.0], dtype=DTYPE, requires_grad=True)
    vb = torch.tensor([-1.0, 0.5], dtype=DTYPE, requires_grad=True)
    control = control_from_velocities(vf, vb, 2.0)
    torch.testing.assert_close(
        control, vf - vb.detach(), rtol=0, atol=0
    )
    control.sum().backward()
    assert vf.grad is not None and vb.grad is None


TESTS = (
    ("sigma schedule", test_sigma_offset),
    ("warm start", test_deterministic_warm_start),
    ("memoryless step", test_memoryless_step),
    ("analytic scalar adjoint", test_analytic_scalar_adjoint),
    ("non-symmetric VJP", test_non_symmetric_vjp),
    ("terminal sign", test_terminal_cost_sign),
    ("AM regression gradient routing", test_am_regression_gradient_routing),
    ("zero-reward fixed point", test_zero_reward_fixed_point),
    ("perturbed-base regularization", test_perturbed_base_regularization),
    ("mask semantics", test_mask_semantics),
    ("stop-gradient semantics", test_stop_gradient_semantics),
)


def main() -> int:
    print("ADJOINT_MATCHING_CORE_TEST")
    failures = 0
    for label, test in TESTS:
        try:
            test()
            print(f"{label}: PASS")
        except Exception as exc:
            failures += 1
            print(f"{label}: FAIL ({type(exc).__name__}: {exc})")
    marker = (
        "ADJOINT_MATCHING_CORE_PASS"
        if failures == 0
        else "ADJOINT_MATCHING_CORE_FAIL"
    )
    print(f"overall: {marker}")
    print(marker)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
