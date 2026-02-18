# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FP8 MoE backend selection ordering.

Verifies that DeepGEMM is preferred over FlashInfer CUTLASS on Hopper (SM 9.0)
GPUs, while FlashInfer CUTLASS retains priority on Blackwell and newer (SM >=
10.0). See: https://github.com/vllm-project/vllm/issues/34249
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)

# Block-scale FP8 quant keys (used by DeepSeek-style models)
_WEIGHT_KEY = kFp8Static128BlockSym
_ACTIVATION_KEY = kFp8Dynamic128Sym


def _make_supported_cls():
    """Return a mock kernel class that always reports itself as supported."""
    mock_cls = MagicMock()
    mock_cls.is_supported_config.return_value = (True, None)
    return mock_cls


def _make_unsupported_cls():
    """Return a mock kernel class that always reports itself as unsupported."""
    mock_cls = MagicMock()
    mock_cls.is_supported_config.return_value = (False, "mocked as unsupported")
    return mock_cls


def _mock_backend_to_kernel_cls(supported_backends):
    """
    Return a mock for ``backend_to_kernel_cls`` that reports only the
    backends listed in *supported_backends* as supported.
    """

    def _impl(backend):
        if backend in supported_backends:
            return _make_supported_cls()
        return _make_unsupported_cls()

    return _impl


@pytest.fixture()
def _moe_config():
    return make_dummy_moe_config()


def _run_select(
    moe_config,
    *,
    is_sm90: bool,
    supported_backends: set,
) -> Fp8MoeBackend:
    """
    Run ``select_fp8_moe_backend`` with the specified architecture and set of
    supported backends, returning only the selected :class:`Fp8MoeBackend`.
    """
    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.fp8.current_platform"
    ) as mock_platform:
        # Simulate CUDA device
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False

        # is_device_capability(90) controls whether we are on SM 9.0 (Hopper)
        mock_platform.is_device_capability.return_value = is_sm90

        with (
            patch(
                "vllm.model_executor.layers.fused_moe.oracle.fp8."
                "is_supported_config_trtllm_fp8",
                return_value=(False, "not supported in test"),
            ),
            patch(
                "vllm.model_executor.layers.fused_moe.oracle.fp8.backend_to_kernel_cls",
                side_effect=_mock_backend_to_kernel_cls(supported_backends),
            ),
        ):
            backend, _ = select_fp8_moe_backend(
                config=moe_config,
                weight_key=_WEIGHT_KEY,
                activation_key=_ACTIVATION_KEY,
            )
    return backend


class TestFp8BackendOrderingOnHopper:
    """
    On Hopper (SM 9.0), when both DeepGEMM and FlashInfer CUTLASS backends
    report themselves as supported, DeepGEMM must be selected first.
    """

    def test_deepgemm_preferred_when_both_supported(self, _moe_config):
        """Regression test for https://github.com/vllm-project/vllm/issues/34249.

        DeepGEMM must come before FlashInfer CUTLASS in the backend priority
        list on Hopper so that it is selected when both are available.
        """
        selected = _run_select(
            _moe_config,
            is_sm90=True,
            supported_backends={
                Fp8MoeBackend.DEEPGEMM,
                Fp8MoeBackend.FLASHINFER_CUTLASS,
            },
        )
        assert selected == Fp8MoeBackend.DEEPGEMM

    def test_flashinfer_cutlass_selected_when_deepgemm_unavailable(self, _moe_config):
        """FlashInfer CUTLASS is used as a fallback when DeepGEMM is not
        available on Hopper (e.g. deep_gemm package not installed)."""
        selected = _run_select(
            _moe_config,
            is_sm90=True,
            supported_backends={Fp8MoeBackend.FLASHINFER_CUTLASS},
        )
        assert selected == Fp8MoeBackend.FLASHINFER_CUTLASS

    def test_triton_selected_as_last_resort(self, _moe_config):
        """Triton is the final fallback when no specialised backend is
        available."""
        selected = _run_select(
            _moe_config,
            is_sm90=True,
            supported_backends={Fp8MoeBackend.TRITON},
        )
        assert selected == Fp8MoeBackend.TRITON


class TestFp8BackendOrderingOnBlackwell:
    """
    On Blackwell and newer (SM >= 10.0, i.e. ``is_device_capability(90)``
    returns False for non-Hopper), FlashInfer CUTLASS retains its higher
    priority over DeepGEMM.
    """

    def test_flashinfer_cutlass_preferred_when_both_supported(self, _moe_config):
        """FlashInfer CUTLASS must come before DeepGEMM in the priority list
        on non-Hopper architectures."""
        selected = _run_select(
            _moe_config,
            is_sm90=False,
            supported_backends={
                Fp8MoeBackend.DEEPGEMM,
                Fp8MoeBackend.FLASHINFER_CUTLASS,
            },
        )
        assert selected == Fp8MoeBackend.FLASHINFER_CUTLASS

    def test_deepgemm_selected_when_flashinfer_cutlass_unavailable(self, _moe_config):
        """DeepGEMM is the fallback when FlashInfer CUTLASS is not available
        on Blackwell."""
        selected = _run_select(
            _moe_config,
            is_sm90=False,
            supported_backends={Fp8MoeBackend.DEEPGEMM},
        )
        assert selected == Fp8MoeBackend.DEEPGEMM
