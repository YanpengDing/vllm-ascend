# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-Next cache specs and compressed-cache addressing."""

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.attention.indexer_kpool import select_indexer_block_size
from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.models.glm5next.kv_cache import (
    AscendIndexerKPoolStateSpec,
    format_indexer_kpool_slot_mapping,
)


def test_state_uses_sliding_pages_and_full_precision():
    register_ascend_kv_cache_specs()
    spec = AscendIndexerKPoolStateSpec(
        block_size=4,
        sliding_window=4,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.float32,
    )
    assert spec.page_size_bytes == 4096
    assert KVCacheSpecRegistry.get_manager_class(spec) is SlidingWindowManager
    assert spec.max_admission_blocks_per_request(16, 1024) > 1


@pytest.mark.parametrize(
    ("dtype", "block_size", "sliding_window"),
    [
        (torch.bfloat16, 4, 4),
        (torch.float32, 8, 4),
    ],
)
def test_invalid_state_layout_is_rejected(dtype, block_size, sliding_window):
    with pytest.raises(ValueError):
        AscendIndexerKPoolStateSpec(
            block_size=block_size,
            sliding_window=sliding_window,
            num_kv_heads=1,
            head_size=256,
            dtype=dtype,
        )


def test_completed_pool_slots_preserve_logical_block_padding():
    slots = torch.tensor([0, 14, 15, 16, 127, 128, 143, -1])
    positions = torch.tensor([0, 14, 15, 16, 127, 128, 143, 15])
    actual = format_indexer_kpool_slot_mapping(slots, positions, 128, 16)
    assert actual.tolist() == [-1, -1, 0, -1, 7, -1, 8, -1]


@pytest.mark.parametrize("ratio", [0, 1, 3])
def test_invalid_pool_geometry_is_rejected(ratio):
    with pytest.raises(ValueError):
        format_indexer_kpool_slot_mapping(
            torch.tensor([0]), torch.tensor([0]), 128, ratio
        )


@pytest.mark.parametrize(
    ("storage_block_size", "expected"),
    [(16, (16, 1)), (1024, (1024, 1)), (2048, (1024, 2)), (1536, (768, 2))],
)
def test_indexer_block_size_selection(storage_block_size, expected):
    assert select_indexer_block_size(storage_block_size) == expected


@pytest.mark.parametrize("storage_block_size", [0, 7, 17])
def test_invalid_indexer_block_size_is_rejected(storage_block_size):
    with pytest.raises(ValueError):
        select_indexer_block_size(storage_block_size)
