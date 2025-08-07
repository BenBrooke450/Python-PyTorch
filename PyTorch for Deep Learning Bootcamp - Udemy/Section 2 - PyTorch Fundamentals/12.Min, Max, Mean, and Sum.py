
import torch

tensor_A = torch.tensor([[1,2],[4,5],[7,8]])

tensor_B = torch.tensor([[3,2],[6,5],[7,8]])


x = torch.arange(0,100,10)
print(x)
#tensor([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])


print(torch.min(x),x.min())
#tensor(0) tensor(0)


print(torch.max(x),x.max())
#tensor(90) tensor(90)




print(torch.mean(x.type(torch.float32)), x.type(torch.float32).mean())
#tensor(45.)tensor(45.)







print(x.argmin())
#tensor(0)


print(x.argmax())
#tensor(9)












