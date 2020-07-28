"""
    Copyright 2020 by P Suarez-Serrato, Jose Ruíz and M Evangelista-Alvarado.
    Instituto de Matemáticas (UNAM-CU) México
    This is free software; you can redistribute it and/or
    modify it under the terms of the MIT License,
    https://en.wikipedia.org/wiki/MIT_License.

    This software has NO WARRANTY, not even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
from __future__ import unicode_literals

import sympy as sym
import numpy as np
import torch
import itertools


def matrix_of(two_tensor, dim):
    """
        This method converts a two tensor (in dictionary format) to
        matrix format as a symbol variable.

        Params:
            :two_tensor:
                Is a two tensor in dictonary format
            :dim:
                Is the matrix dimension, ie the matrix dimension is dim x dim
        Return:
            A symbolic matrix associated with a two tensor
        Example:
            >>> two_tensor = {(1, 2): 'x3', (3, 1): 'x2', (2, 3): 'x1'}
            >>> dim = 4
            >>> result = matrix_of(two_tensor, dim)
            >>> result
            >>> Matrix([
                [  0,  x3, -x2, 0],
                [-x3,   0,  x1, 0],
                [ x2, -x1,   0, 0],
                [  0,   0,   0, 0]])
    """
    # Create a zero matrix with appropriate dimension
    matrix = sym.zeros(dim)
    # Assign the value to correspond to the dictionary key (i,j) to each entry
    # of the matrix if the value exists
    for key in two_tensor:
        matrix[key[0] - 1, key[1] - 1] = two_tensor[key]
        matrix[key[1] - 1, key[0] - 1] = (-1) * matrix[key[0] - 1, key[1] - 1]
    # Sympy matrix
    return matrix


def vector_of(one_tensor, dim):
    """
        This method converts a one tensor (in dictionary format) to
        matrix format (vector-column) as a symbol variable.

        Params:
            :two_tensor:
                Is a one tensor in dictonary format
            :dim:
                Is the vector-column dimension, ie the vector dimension is dim x 1
        Return:
            A symbolic matrix associated with a one tensor
        Example:
            >>> one_tensor = {(1,): 'x1', (2,): 'x2', (3,): 'x3'}
            >>> dim = 4
            >>> result = vector_of(one_tensor, dim)
            >>> result
            >>> Matrix([
                [ x1],
                [ x2],
                [ x3],
                [  0]])
    """
    # Create a zero vector-column with appropriate dimension
    vector = sym.zeros(dim, 1)
    # Assign the value to correspond to the dictionary key (i,) to each entry
    # of the matrix (i,1) if the value exists
    for key in one_tensor:
        vector[key[0] - 1] = one_tensor[key]
    # Sympy matrix
    return vector


def dict_mesh_eval(dictionary, mesh, coords):
    """
        This method evaluates a symbolic dictionary on a mesh, it also preserves
        the order of the mesh, that is, it is possible to associate a point with
        its evaluation.

        Params:
            :dictionary:
                Is a n-tensor in dictonary format.
            :mesh:
                Is a set of points to evaluates.
            :coords:
                Is the coordinates that is writen the dictionary symbolic
        Return:
            A list by comprehension that contains all evaluations from the mesh
    """
    # Create symbolic lambda function
    lamb_dict = sym.lambdify(coords, dictionary)
    # Evaluated all the points from de mesh and returns a list by comprehension
    return [lamb_dict(*e) for e in mesh]


def list_mesh_eval(some_list, mesh, coords):
    """
        This method evaluates a symbolic list on a mesh, it also preserves
        the order of the mesh, that is, it is possible to associate a point with
        its evaluation.
        Params:
            :some_lits:
                Is a 1-tensor in list format.
            :mesh:
                Is a set of points to evaluates.
            :coords:
                Is the coordinates that is writen the dictionary symbolic
        Return:
            A list by comprehension that contains all evaluations from the mesh
    """
    # Create symbolic lambda function
    lamb_some_list = sym.lambdify(coords, some_list, 'numpy')
    # Evaluated all the points from de mesh and returns a list by comprehension
    return [lamb_some_list(*e) for e in mesh]


def num_one_form_to_vector(one_form, mesh, dim, coords):
    """
        This method evaluates a 1-tensor (in dictionary format) with all points
        from a mesh.
        Params:
            :one_form:
                Is a 1-tensor in dictonary format
            :mesh:
                Is a set of points to evaluates.
            :dim:
                Is the vector-column dimension, ie the vector dimension is dim x 1
        Return:
            A list that contains all evaluations from 1-vector with all point from a mesh
    """
    one_form = vector_of(one_form, dim)
    lamb_one_form = sym.lambdify(coords, one_form, 'numpy')
    results = [lamb_one_form(*e) for e in mesh]
    return results


def check_multivector_dim(multivector):
    """
        Check that all values keys of multivector are correctly
    """
    # TODO ADD when multivector has dimension 0
    multivector_keys = [len(key) for key in multivector.keys()]
    if len(set(multivector_keys)) == 1:
        return multivector_keys[0]
    return False


def dicts_to_matrices(dict_list, dim):
    """
        PENDIENTE
    """
    results = []
    for dict_element in dict_list:
        base_matrix = np.zeros((dim, 1)) if check_multivector_dim(dict_list[0]) == 1 else np.zeros((dim, dim))
        for key in dict_element.keys():
            if len(key) == 1:
                i, j = key[0]-1, 0
                base_matrix[i, j] = dict_element[key]
            else:
                i, j = key[0]-1, key[1]-1
                base_matrix[i, j] = dict_element[key]
                base_matrix[j, i] = (-1)*(dict_element[key])
        results.append(base_matrix)
    return np.array(results)


def create_tensor_base(dim, multivector_dim):
    """
        PENDIENTE
    """
    # Create all combinations from multivector_dim in dim
    combs = list(itertools.combinations(range(1, dim+1), multivector_dim))
    # Get all multi-index from a multivector base
    tensor_index = list()
    index_position_aux = list()
    if multivector_dim == 0:
        # return the same string that user input
        return torch.zeros(1, dim)
    if multivector_dim == dim:
        return torch.zeros(1, 1)
    if multivector_dim == 1:
        zero_tensor = torch.zeros((dim, 1))
    if multivector_dim == 2:
        tensor_index.append(dim)
        tensor_index.append(dim)
        zero_tensor = torch.zeros(tensor_index)
    if multivector_dim > 2:
        multi_index = [[index for index in list(comb[0:len(comb)-2])] for comb in combs]
        print(f'multi_index: {multi_index}')
        # Create index to create tensor in Pytorch
        # Caso i = 0
        tensor_list = list(set([index[0] for index in multi_index]))
        tensor_index.append(len(tensor_list))
        # Otros Casos
        i = 1
        previous_value = tensor_list[0]
        while i < len(multi_index[0]):
            aux = list(set([index[i] for index in multi_index if previous_value in index]))
            index_position_aux.append(aux)
            tensor_index.append(len(aux))
            previous_value = aux[0]
            i = i + 1
        tensor_index.append(dim)
        tensor_index.append(dim)
        zero_tensor = torch.zeros(tensor_index)

    return zero_tensor, index_position_aux


def add_tensor_values(multivector, tensor_base, index_base):
    """
        PENDIENTE
    """
    if multivector == {0: 0}:
        return tensor_base

    # TODO multivector of dimension 1

    for key in multivector.keys():
        # TODO Arreglar de mayor a menor
        matrix_index = list()
        for index in list(key[0:len(key)-2]):
            matrix_index.append(index)
        II = key[len(key)-2]
        JJ = key[len(key)-1]
        # print(f"matrix_index: {matrix_index}, I: {I}, J: {J}")

        if len(matrix_index) == 0:
            matrix_tensor = tensor_base
        else:
            matrix_tensor = tensor_base
            matrix_tensor = matrix_tensor[matrix_index[0]-1]
            for i in matrix_index[1:]:
                partial_index = matrix_index.index(i)
                final_index = index_base[partial_index - 1].index(i)
                matrix_tensor = matrix_tensor[final_index]

        matrix_tensor[II-1, JJ-1] = multivector[key]
        matrix_tensor[JJ-1, II-1] = (-1)*multivector[key]
    return tensor_base


# Create our Dimension Error Exception
class DimensionError(Exception):
    pass


def show_coordinates(coordinates):
    """
        This method shows the current coordinates from Numerical Poisson Geometry Class
    """
    if len(coordinates) > 2:
        return f'({coordinates[0]},...,{coordinates[-1]})'
    else:
        raise DimensionError("Dimension do not sufficient")


def validate_dimension(dimension):
    """
        This method check if the dimension variable is valid for the this class
    """
    if not isinstance(dimension, int):
        raise TypeError("Your variable is not a int type")

    if dimension < 2:
        raise DimensionError("Your dimension is not greater than two")
    else:
        return dimension
