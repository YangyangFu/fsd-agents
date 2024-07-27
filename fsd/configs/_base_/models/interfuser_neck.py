from fsd.registry import MODELS

model = dict(
    type='InterFuserNeck',
    in_channels=2048,
    out_channels=256
)

model = MODELS.build(model)
print(model)