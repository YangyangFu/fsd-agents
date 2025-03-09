import pytest
import torch

from fsd.models.detectors.bevformer.modules.spatial_cross_attention import (
    SpatialCrossAttention,
    MultiScaleDeformableAttention3D
)


def test_multi_scale_deformable_attention_3d():
    """Test MultiScaleDeformableAttention3D module."""
    # Set up parameters
    batch_size = 2
    embed_dims = 256
    num_query = 100
    num_key = 400+100+25+4  # 4 feature levels
    num_heads = 8
    num_levels = 4
    num_points = 5
    
    # Create module
    attn = MultiScaleDeformableAttention3D(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
    )
    
    # Create inputs
    query = torch.rand(batch_size, num_query, embed_dims)
    value = torch.rand(batch_size, num_key, embed_dims)
    
    # For simplicity, assume we have 4 feature levels with same size
    h = [20, 10, 5, 2]
    w = [20, 10, 5, 2]
    
    # Create reference points for each query
    num_z_anchors = 3  # Number of Z anchors for each BEV query
    reference_points = torch.rand(batch_size, num_query, num_levels*num_z_anchors, 2)
    
    # Create spatial shapes and level start index
    spatial_shapes = torch.tensor([[h[0], w[0]], [h[1], w[1]], [h[2], w[2]], [h[3], w[3]]])
    level_start_index = torch.tensor([0, h[0]*w[0], h[0]*w[0]+h[1]*w[1], h[0]*w[0]+h[1]*w[1]+h[2]*w[2]])
    
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


def test_spatial_cross_attention():
    """Test SpatialCrossAttention module."""
    # Set up parameters
    batch_size = 2
    embed_dims = 256
    num_query = 100
    num_cams = 5
    num_heads = 8
    num_levels = 4
    num_points = 12
    num_z = 4
    
    # Create module
    attn = SpatialCrossAttention(
        embed_dims=embed_dims,
        num_cams=num_cams,
        pc_range=None,
        deformable_attention=dict(
            type='MultiScaleDeformableAttention3D',
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
        )
    )

    # Assume 4 feature levels with decreasing sizes
    h = [20, 10, 5, 2]
    w = [20, 10, 5, 2]
    num_key_per_cam = sum([h[0]*w[0], h[1]*w[1], h[2]*w[2], h[3]*w[3]])
    
    # Create inputs
    query = torch.rand(batch_size, num_query, embed_dims)
    key = torch.rand(num_cams, num_key_per_cam, batch_size, embed_dims)
    query_pos = torch.rand(batch_size, num_query, embed_dims)

    # Create spatial shapes and level start index
    spatial_shapes = torch.tensor([[h[0], w[0]], [h[1], w[1]], [h[2], w[2]], [h[3], w[3]]])
    level_start_index = torch.tensor([0, h[0]*w[0], h[0]*w[0]+h[1]*w[1], h[0]*w[0]+h[1]*w[1]+h[2]*w[2]])
    
    # Create reference points 
    reference_points_cam = torch.rand(num_cams, batch_size, num_query, num_z, 2)
    bev_mask = torch.rand(num_cams, batch_size, num_query, num_z)
    
    # Forward pass
    output = attn(
        query=query,
        key=key,
        value=key,
        query_pos=query_pos,
        spatial_shapes=spatial_shapes,
        reference_points=reference_points_cam,
        bev_mask=bev_mask,
        level_start_index=level_start_index,
    )
    
    # Check output shape
    assert output.shape == (batch_size, num_query, embed_dims)


if __name__ == '__main__':
    #test_multi_scale_deformable_attention_3d()
    test_spatial_cross_attention()
    print("All tests passed!")
