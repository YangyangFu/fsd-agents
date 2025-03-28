import pytest
import torch

from fsd.models.detectors.bevformer.modules.temporal_self_attention import TemporalSelfAttention


def test_temporal_self_attention():
    """Test TemporalSelfAttention module."""
    # Set up parameters
    batch_size = 3
    embed_dims = 256
    num_heads = 8
    num_points = 5
    num_bev_queue = 2
    num_levels = 1
    num_query = 100
    
    # Create module
    attn = TemporalSelfAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        num_bev_queue=num_bev_queue,
    )
    
    # Create inputs: bev query at current time and bev feature from prev step
    # bev range
    h, w = 10, 10
    num_query = h * w
    query = torch.rand(batch_size, num_query, embed_dims)
    bev_prev = torch.rand(batch_size, num_query, embed_dims)
    value = torch.concat([bev_prev, query], 0)
    
    # Create spatial shapes and level start index
    spatial_shapes = torch.tensor([[10, 10]])
    level_start_index = torch.tensor([0])
    
    # Create reference points for each query
    reference_points = torch.rand(batch_size*num_bev_queue, num_query, num_levels, 2)
    
    # Forward pass
    output = attn(
        query=query,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    
    # Check output shape
    assert output.shape == (batch_size, num_query, embed_dims)


if __name__ == '__main__':
    test_temporal_self_attention()
    print("All tests passed!")
