
"""
Write a Python function that computes the dot product of a matrix and a vector.
    The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible.
    A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector.
    For example, an n x m matrix requires a vector of length m.

"""


import torch

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:

    try:
        a = torch.tensor(a)
        b = torch.tensor(b).unsqueeze(1)
        matrix_dot = torch.mm(a,b)
        print(matrix_dot)
    except RuntimeError:
        return -1


print(matrix_dot_vector(a = [[1,2,3],[2,4,5]], b = [1, 2]))




def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:

    if len(a[0]) != len(b):
        return -1
    else:
        for x in zip(a,[[c]]:
            for a in x:



