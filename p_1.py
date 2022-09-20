import torch
import numpy as np
# np_data=np.arange(9).reshape(3,3)
# torch_data=torch.from_numpy(np_data)
# tensor2arry=torch_data.numpy()
#
# print("\n numpy array:".title(),np_data)
# print("\n torch tensor:".title(),torch_data)
# print("\n tensor2arry:".title(),tensor2arry)
#
# #abs绝对值计算
# data=[-1,-12,1,20]
# tensor=torch.FloatTensor(data)
#
# print(f'''
# abs:
# numpy:{np.abs(data)}
# tensor:{torch.abs(tensor)}
# ''')
# #sin 三角函数sin计算
# print(f'''
# sin:
# numpy:{np.sin(data)}
# tensor:{torch.sin(tensor)}
# ''')
# #mean均值
# print(f'''
# mean:
# numpy:{np.mean(data)}
# tensor:{torch.mean(tensor)}
# ''')
# # matrix multiplication 矩阵点乘
# #matmul,mm
# print(f'''
# matmul,mm:
# numpy:{np.matmul(np_data,np_data)}
# tensor:{torch.mm(torch_data,torch_data)}
# ''')
# data=np.array(data)
# print(data)
#
#
# print(f'''
# dot:
# numpy:{data.dot(data)}
# tensor:{(tensor.dot(tensor))}
# ''')
from torch.autograd import  Variable
#VARIABLE
print("variable")
tensor=torch.FloatTensor([[1,2],[3,4]])
variable=Variable(tensor,requires_grad=True)
print(tensor)
print(variable)
t_out=torch.mean(tensor*tensor)
print(tensor*tensor*tensor)
v_out=torch.mean(variable*variable)
print(t_out)
print(v_out)
v_out.backward()
print(variable.grad)
