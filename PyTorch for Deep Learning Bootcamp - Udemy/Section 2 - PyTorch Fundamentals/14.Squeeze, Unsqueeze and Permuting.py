
import torch

#To do so you can use torch.squeeze() (I remember this as squeezing the tensor to only have dimensions over 1).

x = torch.ones(1,1,10)
print(x,x.shape)
#tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]) torch.Size([1, 1, 10])

print(x.shape,x.ndim)
#torch.Size([1, 1, 10]) 3

print(x.squeeze())
#tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

print(x.squeeze().shape,x.squeeze().ndim)
#torch.Size([10]) 1






x = torch.randn(2, 3, 5)
print(x.size())


torch.Size([2, 3, 5])


torch.permute(x, (2, 0, 1)).size()


torch.Size([5, 2, 3])



