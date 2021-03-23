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
from numpoisson.errors import DimensionError


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
    if not bool(dictionary):
        return np.array([])
    # Create symbolic lambda function
    lamb_dict = sym.lambdify(coords, dictionary)
    # Evaluated all the points from de mesh and returns a list by comprehension
    dict_eval = [lamb_dict(*e) for e in mesh]
    return np.array([{e: dictt[e] for e in dictt if dictt[e] != 0} for dictt in dict_eval])


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
    if not bool(some_list):
        return np.array([])
    # Create symbolic lambda function
    lamb_some_list = sym.lambdify(coords, some_list, 'numpy')
    # Evaluated all the points from de mesh
    return np.array([lamb_some_list(*e) for e in mesh])


def num_matrix_of(two_tensor, dim):
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
    matrix = np.zeros((dim, dim))
    # Assign the value to correspond to the dictionary key (i,j) to each entry
    # of the matrix if the value exists
    for e in two_tensor:
        matrix[e[0] - 1, e[1] - 1] = two_tensor[e]
        matrix[e[1] - 1, e[0] - 1] = (-1) * matrix[e[0] - 1, e[1] - 1]
    return matrix


def num_vector_of(one_tensor, dim):
    """
        This method converts a one tensor (in dictionary format) to
        matrix format (vector-column) as a symbol variable.

        Params:
            :two_tensor:
                Is a one tensor in dictonary format
            :dim:
                Is the vector-column dimension, ie the vector dimension is dim x 1
        Return:
            A numeric matrix associated with a one tensor
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
    vector = np.zeros((dim, 1))
    # Assign the value to correspond to the dictionary key (i,) to each entry
    # of the matrix (i,1) if the value exists
    for key in one_tensor:
        vector[key[0] - 1] = one_tensor[key]
    return vector


def zeros_array(degree, dim):
    """"""
    if degree == 0:
        return np.zeros([])
    if degree == 1:
        return np.zeros((dim, 1))
    deg = [dim] * degree
    return np.zeros(deg)


def validate_dimension(dim):
    """ This method check if the dimension variable is valid for the this class"""
    if not isinstance(dim, int):
        raise DimensionError(F"Your variable {dim} is not a int type")

    if dim < 1:
        raise DimensionError("Your dimension is not greater than two")
    else:
        return dim
