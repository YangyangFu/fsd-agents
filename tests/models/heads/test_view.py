import torch

# check the difference between tensor.view, tensor.reshape and tensor.permute, tensor.transpose

# view
# view is a method that returns a new tensor with the same data as the original tensor but with a different shape.

orignal_tensor = torch.Tensor(range(24)).view(2, 3, 4)
print(orignal_tensor)

print( 'viewing the tensor; --------------')
view_tensor = orignal_tensor.view(3, 2, 4)
print(view_tensor)

print( 'reshaping the tensor; --------------')
reshape_tensor = orignal_tensor.reshape(3, 2, 4)
print(reshape_tensor)

print( 'permuting the tensor; --------------')
permute_tensor = orignal_tensor.permute(1, 0, 2)
print(permute_tensor)