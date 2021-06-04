"""
    Copyright 2020 by P Suárez-Serrato, Jose C. Ruíz Pantaleón and M Evangelista-Alvarado.
    Instituto de Matemáticas (UNAM-CU) México
    This is free software; you can redistribute it and/or
    modify it under the terms of the MIT License,
    https://en.wikipedia.org/wiki/MIT_License.

    This software has NO WARRANTY, not even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
from __future__ import unicode_literals

import numpy as np
import sympy as sym
import permutation as perm
import torch
import tensorflow as tf
from scipy import linalg as splinalg
from scipy.sparse import csr_matrix, triu
from poisson.poisson import PoissonGeometry
from numpoisson.errors import (MultivectorError, FunctionError,
                               DiferentialFormError, CasimirError,
                               DimensionError)
from numpoisson.utils import (dict_mesh_eval, list_mesh_eval,
                              num_matrix_of, num_vector_of,
                              zeros_array, validate_dimension)


class NumPoissonGeometry:
    """ This class provides some useful tools for Poisson-Nijenhuis calculus on Poisson manifolds."""
    def __init__(self, dimension, variable='x'):
        # Obtains the dimension
        self.dim = validate_dimension(dimension)
        # Define what variables the class will work with
        self.variable = variable
        # Create the symbolics symbols
        self.coords = sym.symbols(f'{self.variable}1:{self.dim + 1}')
        # Intances Poisson Geometry package
        self.pg = PoissonGeometry(self.dim, self.variable)

    def num_bivector(self, bivector, mesh, torch_output=False, tf_output=False, dict_output=False):
        """ Evaluates a bivector field into at each point of the mesh.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :pt_out/tf_out:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of a bivector. This value
            can be converted to Tensor PyTorch or TensorFlow by setting their respective flag as True in
            the params.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg3 = NumPoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # Creates a simple mesh
            >>> mesh = np.array([[0., 0., 1.]])
            >>> # Evaluates the mesh into bivector
            >>> npg3.num_bivector(bivector, mesh)
            >>> [[[ 0.  1. -0.]
                  [-1.  0.  0.]
                  [ 0. -0.  0.]]
            >>> npg3.num_bivector(bivector, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[ 0.  1. -0.]
                  [-1.  0.  0.]
                  [ 0. -0.  0.]]], shape=(1, 3, 3), dtype=float64)
            >>> npg3.num_bivector(bivector, mesh, torch_output=True)
                tensor([[[ 0.,  1., -0.],
                         [-1.,  0.,  0.],
                         [ 0., -0.,  0.]]], dtype=torch.float64)
            >>> npg3.num_bivector(bivector, mesh, dict_output=True)
            >>> [{(1, 2): 1.0, (1, 3): -0.0, (2, 3): 0.0}]
        """
        len_keys = []
        for e in bivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F"repeated indexes {e} in {bivector}")
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F"invalid key {e} in {bivector}")
            len_keys.append(len(e))
        if len(set(len_keys)) > 1:
            raise MultivectorError('keys with different lengths')

        # Evaluates all point from the mesh in the bivector and save in a np array
        dict_list = dict_mesh_eval(bivector, mesh, self.coords)
        raw_result = [num_matrix_of(e, self.dim) for e in dict_list]
        np_result = np.array(raw_result)

        # return the result in a PyTorch tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            return dict_list
        # return the result in Numpy array
        return np_result

    def num_bivector_to_matrix(self, bivector, mesh, torch_output=False, tf_output=False):
        """ Evaluates a matrix of a 2-contravariant tensor field or bivector field into a mesh.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :pt_out/tf_out:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of a bivector. This value
            can be converted to Tensor PyTorch or TensorFlow by setting their respective flag as True in
            the params.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg3 = NumPoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}
            >>> # Creates a mesh
            >>> mesh = np.array([[0., 0., 1.]])
            >>> # Evaluates the mesh into bivector
            >>> npg3.num_bivector_to_matrix(bivector, mesh)
            >>> [[[ 0.  1.  0.]
                  [-1.  0.  0.]
                  [-0. -0.  0.]]]
            >>> npg3.num_bivector_to_matrix(bivector, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[ 0.  1.  0.]
                  [-1.  0.  0.]
                  [-0. -0.  0.]]], shape=(1, 3, 3), dtype=float64)
            >>> npg3.num_bivector_to_matrix(bivector, mesh, torch_output=True)
            >>> tensor([[[ 0.,  1.,  0.],
                         [-1.,  0.,  0.],
                         [-0., -0.,  0.]]], dtype=torch.float64)
        """
        len_keys = []
        for e in bivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F"repeated indexes {e} in {bivector}")
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F"invalid key {e} in {bivector}")
            len_keys.append(len(e))
        if len(set(len_keys)) > 1:
            raise MultivectorError('keys with different lengths')

        dict_list = dict_mesh_eval(bivector, mesh, self.coords)
        # Converts to bivector from dict format into matrix format
        raw_result = [num_matrix_of(e, self.dim) for e in dict_list]
        # Evaluates all point from the mesh in the bivector and save in a np array
        np_result = np.array(raw_result)
        # return the result in a PyTorch tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # TODO add dict_output flag
        # return the result in Numpy array
        return np_result

    def num_sharp_morphism(self, bivector, one_form, mesh, torch_output=False, tf_output=False, dict_output=False):
        """
            Evaluates the image of a differential 1-form under the vector bundle morphism 'sharp' P #: T * M -> TM
            defined by P # (alpha): = i_ (alpha) P in the points from a mesh given, where P is a Poisson bivector
            field on a manifold M, alpha a 1-form on M and i the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :one_form:
            Is a 1-form differential in a dictionary format with tuple type 'keys' and string type 'values'.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :pt_out/tf_out:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of sharp morphism. This value
            can be converted to Tensor PyTorch or TensorFlow by setting their respective flag as True in
            the params.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg3 = NumPoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> # For one form x1*dx1 + x2*dx2 + x3*dx3.
            >>> one_form = {(1,): 'x1', (2,): 'x2', (3,): 'x3'}
            >>> # Creates a mesh
            >>> mesh = np.array([[0., 0., 1.]])
            >>> # Evaluates the mesh into bivector
            >>> npg3.num_sharp_morphism(bivector, one_form, mesh)
            >>> [[[-0.]
                  [-0.]
                  [-0.]]]
            >>> npg3.num_sharp_morphism(bivector, one_form, mesh, torch_output=True)
            >>> tensor([[[-0.],
                         [-0.],
                         [-0.]]], dtype=torch.float64)
            >>> npg3.num_sharp_morphism(bivector, one_form, mesh, tf_output=True)
            >>> tf.Tensor([[[-0.]
                [-0.]
                [-0.]]], shape=(1, 3, 1), dtype=float64)
            >>> npg3.num_sharp_morphism(bivector, one_form, mesh, dict_output=True)
            >>> array([{}], dtype=object)
        """
        for e in one_form:
            if len(e) > 1:
                raise DiferentialFormError(F"invalid key {e} in {one_form}")
            if e[0] <= 0:
                raise DiferentialFormError(F"invalid key {e} in {one_form}")

        # Converts to one-form from dict format into vector-column format (matrix of dim x 1)
        dict_list = dict_mesh_eval(one_form, mesh, self.coords)
        one_form_num_vec = [num_vector_of(e, self.dim) for e in dict_list]
        # Converts to bivector from dict format into matrix format
        bivector_num_mat = self.num_bivector_to_matrix(bivector, mesh)
        raw_result = map(lambda e1, e2: (-1) * np.dot(e1, e2), bivector_num_mat, one_form_num_vec)
        # Making the product of bivector matrix with vector-column and saving the result in a Numpy array
        np_result = np.array(tuple(raw_result))

        # return the result in a PyTorch tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            dict_list = []
            for e in range(len(mesh)):
                dictionary = dict(enumerate(np_result[e].flatten(), 1))
                remove_zeros = {(e,): dictionary[e] for e in dictionary if dictionary[e] != 0}
                if not bool(remove_zeros):
                    remove_zeros = {}
                dict_list.append(remove_zeros)
            return np.array(dict_list)
        # return the result in Numpy array
        return np_result

    def num_hamiltonian_vf(self, bivector, function, mesh, torch_output=False, tf_output=False, dict_output=False):
        """
            Evaluates the Hamiltonian vector field of a function relative to a Poisson bivector field in
            the points from a mesh given. The Hamiltonian vector field is calculated as follows: X_h = P#(dh),
            where d is the exterior derivative of h and P#: T*M -> TM is the vector bundle morphism defined
            by P#(alpha) := i_(alpha)P, with i is the interior product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :ham_function:
            Is a function scalar h: M --> R that is a string type.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :torch_output/tf_output:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of a Hamiltonian vector field.
            This value can be converted to Tensor PyTorch or TensorFlow by setting their respective flag as True
            in the params

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg3 = NumPoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> # For hamiltonian_function h(x1,x2,x3) = x1 + x2 + x3.
            >>> ham_function = '1/(x1 - x2) + 1/(x1 - x3) + 1/(x2 - x3)'
            >>> # Creates a mesh
            >>> mesh = np.array([[1., 2., 3.]])
            >>> # Evaluates the mesh into bivector
            >>> npg3.num_hamiltonian_vf(bivector, ham_function, mesh)
            >>> [[[ 2.5]
                  [-5. ]
                  [ 2.5]]]
            >>> npg3.num_hamiltonian_vf(bivector, ham_function, mesh, torch_output=True)
            >>> tensor([[[ 2.5000],
                         [-5.0000],
                         [ 2.5000]]], dtype=torch.float64)
            >>> npg3.num_hamiltonian_vf(bivector, ham_function, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[ 2.5]
                  [-5. ]
                  [ 2.5]]], shape=(1, 3, 1), dtype=float64)
            >>> npg3.num_hamiltonian_vf(bivector, ham_function, mesh, dict_output=True)
            >>> array([{(1,): 2.5, (2,): -5.0, (3,): 2.5}], dtype=object)
        """
        # Converts the hamiltonian_function from type string to symbolic expression
        ff = sym.sympify(function)
        # Calculates the differential matrix of Hamiltonian function
        d_ff = sym.Matrix(sym.derive_by_array(ff, self.coords))
        d_ff = {(i + 1,): d_ff[i] for i in range(self.dim) if sym.simplify(d_ff[i]) != 0}
        return self.num_sharp_morphism(
            bivector, d_ff, mesh,
            tf_output=tf_output, torch_output=torch_output, dict_output=dict_output
        )

    def num_poisson_bracket(self, bivector, function_1, function_2,
                            mesh, torch_output=False, tf_output=False):
        """
            Calculates the evaluation of Poisson bracket {f,g} = π(df,dg) = ⟨dg,π#(df)⟩ of two functions f and g in
            a Poisson manifold (M,P) in all point from given a mesh. Where d is the exterior derivatives and
            P#: T*M -> TM is the vector bundle morphism defined by P#(alpha) := i_(alpha)P, with i the interior
            product of alpha and P.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :function_1/function_2:
            Is a function scalar f: M --> R that is a string type.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :torch_output/tf_output:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of Poisson bracket.
            This value can be converted to Tensor PyTorch or TensorFlow by setting their respective flag as True
            in the params.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg3 = NumPoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> # For f(x1,x2,x3) = x1 + x2 + x3.
            >>> function_1 = 'x1 + x2 + x3'
            >>> # For g(x1,x2,x3) = '2*x1 + 3*x2 + 4*x3'.
            >>> function_2 = '2*x1 + 3*x2 + 4*x3'
            >>> # Creates a mesh
            >>> mesh = np.array([[5., 10., 0.]])
            >>> # Evaluates the mesh into {f,g}
            >>> npg3.num_poisson_bracket(bivector, function_1, function_2, mesh)
            >>> [-15.]
            >>> npg3.num_poisson_bracket(bivector, function_1, function_2, mesh, torch_output=True)
            >>> tensor([-15.], dtype=torch.float64)
            >>> npg3.num_poisson_bracket(bivector, function_1, function_2, mesh, tf_output=True)
            >>> tf.Tensor([-15.], shape=(1,), dtype=float64)
        """
        # Convert from string to sympy value the function_2
        gg = sym.sympify(function_2)
        # Calculates the differential matrix of function_2
        d_gg = sym.derive_by_array(gg, self.coords)
        # Evaluates the differential matrix of function_2 in each point from a mesh and converts to Numpy array
        dgg_num_vec = list_mesh_eval(d_gg, mesh, self.coords)
        # Evaluates the Hamiltonian vector field with function_1 in each point from a mesh and converts to Numpy
        ham_ff_num_vec = self.num_hamiltonian_vf(bivector, function_1, mesh)
        raw_result = map(lambda e1, e2: np.dot(e1, e2)[0], dgg_num_vec, ham_ff_num_vec)
        np_result = np.array(tuple(raw_result))
        # return the result in a PyTorch tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # TODO add dict_output flag
        # return the result in Numpy array
        return np_result

    def num_curl_operator(self, multivector, function, mesh, torch_output=False, tf_output=False, dict_output=False):
        """
            Evaluates the divergence of multivector field in all points given of a mesh.

        Parameters
        ==========
        :multivector:
            Is a multivector filed in a dictionary format with integer type 'keys' and string type 'values'.
        :function:
            Is a nowhere vanishing function in a string type. If the function is constant you can input the
            number type.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :dict_output:
            Is a boolean flag to indicates if the result is given in a bivector in dictionary format, its
            default value is False.
        :torch_output/tf_output:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of the divergence of multivertor
            field in matricial format. This value can be converted to Tensor PyTorch, TensorFlow or dictionary form
            by setting their respective flag as True in the params.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 4
            >>> npg4 = NumPoissonGeometry(4)
            >>> # For bivector 2*x4*Dx1^Dx3 + 2*x3*Dx1^Dx4 - 2*x4*Dx2^Dx3 + 2*x3*Dx2^Dx4 + (x1-x2)*Dx3^Dx4
            >>> bivector = {(1, 3): '2*x4', (1, 4): '2*x3', (2, 3): '-2*x4', (2, 4): '2*x3', (3, 4): 'x1 - x2')}
            >>> mesh = np.array([0., 0., 0. ])
            >>> # Evaluates the mesh into {f,g}
            >>> npg4.num_curl_operator(bivector, function, mesh)
            >>> [[[ 0.]
                  [ 0.]
                  [ 0.]
                  [ 0.]]]]
            >>> npg4.num_curl_operator(bivector, function, mesh, torch_output=True)
            >>> tensor([[[ 0.],
                         [ 0.],
                         [ 0.],
                         [ 0.]]], dtype=torch.float64)
            >>> npg4.num_curl_operator(bivector, mesh, function, tf_output=True)
            >>> tf.Tensor(
                [[[ 0.],
                  [ 0.],
                  [ 0.],
                  [ 0.]]], shape=(1, 3, 1), dtype=float64)
            >>> npg3.num_curl_operator(bivector, 1, mesh, dict_output=True)
            >>> array([{}], dtype=object)
        """
        if sym.simplify(sym.sympify(function)) == 0:
            raise FunctionError(F'Fuction {function} == 0')

        if not bool(multivector):
            np_result = np.array([])
            if torch_output:
                return torch.from_numpy(np_result)
            if tf_output:
                return tf.convert_to_tensor(np_result)
            if dict_output:
                return np.array({})
            return np_result

        if isinstance(multivector, str):
            np_result = np.array([])
            if torch_output:
                return torch.from_numpy(np_result)
            if tf_output:
                return tf.convert_to_tensor(np_result)
            if dict_output:
                return np.array({})
            return np_result

        len_keys = []
        for e in multivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F"repeated indexes {e} in {multivector}")
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F"invalid key {e} in {multivector}")
            len_keys.append(len(e))
        if len(set(len_keys)) > 1:
            raise MultivectorError('keys with different lengths')

        curl_operator = self.pg.curl_operator(multivector, function)
        if not bool(curl_operator):
            np_result = np.array([])
            if torch_output:
                return torch.from_numpy(np_result)
            if tf_output:
                return tf.convert_to_tensor(np_result)
            if dict_output:
                return np.array({})
            return np_result
        curl_operator = {tuple(map(lambda x: x-1, list(e))): curl_operator[e] for e in curl_operator}
        deg_curl = len(next(iter(curl_operator)))
        permut_curl = []
        for e in curl_operator:
            permut_curl.append({x.permute(e): (x.sign) * curl_operator[e] for x in perm.Permutation.group(deg_curl)})
        for e in permut_curl:
            curl_operator.update(e)
        dict_eval = dict_mesh_eval(curl_operator, mesh, self.pg.coords)

        np_result = []
        zero_tensor = zeros_array(deg_curl, self.pg.dim)
        for dictt in dict_eval:
            copy_zero_tensor = np.copy(zero_tensor)
            for e2 in dictt:
                copy_zero_tensor[e2] = dictt[e2]
            np_result.append(copy_zero_tensor)
        np_result = np.array(np_result)

        # return the result in a PyTorch tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            dict_eval = [{tuple(map(lambda x: x+1, list(e))): dictt[e] for e in dictt} for dictt in dict_eval]
            return np.array(dict_eval)
        # return the result in Numpy array
        return np_result

    def num_coboundary_operator(self, bivector, multivector, mesh,
                                torch_output=False, tf_output=False, dict_output=False):
        """
            Evalueates the Schouten-Nijenhuis bracket between a given (Poisson) bivector field and a (arbitrary)
            multivector field in all points given of a mesh.
            The Lichnerowicz-Poisson operator is defined as
                [P,A](df1,...,df(a+1)) = sum_(i=1)^(a+1) (-1)**(i)*{fi,A(df1,...,î,...,df(a+1))}_P
                                     + sum(1<=i<j<=a+1) (-1)**(i+j)*A(d{fi,fj}_P,..î..^j..,df(a+1))
            where P = Pij*Dxi^Dxj (i < j), A = A^J*Dxj_1^Dxj_2^...^Dxj_a.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :multivector:
            Is a multivector filed in a dictionary format with integer type 'keys' and string type 'values'.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :dict_output:
            Is a boolean flag to indicates if the result is given in a bivector in dictionary format, its
            default value is False.
        :torch_output/tf_output:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of the Schouten-Nijenhuis bracket.
            This value can be converted to Tensor PyTorch, TensorFlow or dictionary for by setting their respective
            flag as True in the params.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg3 = NumPoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> # Defines a one form W
            >>> W = {(1,): 'x1 * exp(-1/(x1**2 + x2**2 - x3**2)**2) / (x1**2 + x2**2)',
                     (2,): 'x2 * exp(-1/(x1**2 + x2**2 - x3**2)**2) / (x1**2 + x2**2)',
                     (3,): 'exp(-1 / (x1**2 + x2**2 - x3**2)**2)'}
            >>> mesh = np.array([[0., 0., 1.]])
            >>> # Evaluates the mesh into {f,g}
            >>> npg3.num_coboundary_operator(bivector, W, mesh)
            >>> [[[ 0.        ,  0.36787945,  0.        ],
                  [-0.36787945,  0.        ,  0.        ],
                  [ 0.        ,  0.        ,  0.        ]]]
            >>> npg3.num_coboundary_operator(bivector, W, mesh, dict_output=True)
            >>> [{(1, 2): 0.36787944117144233}]
            >>> npg3.num_coboundary_operator(bivector, W, mesh, torch_output=True)
            >>> tensor([[[ 0.0000,  0.3679,  0.0000],
                         [-0.3679,  0.0000,  0.0000],
                         [ 0.0000,  0.0000,  0.0000]]], dtype=torch.float64)
            >>> npg3.num_coboundary_operator(bivector, W, mesh, tf_output=True)
            >>> tf.Tensor: shape=(1, 3, 3), dtype=float64, numpy=
                array([[[ 0.        ,  0.36787945,  0.        ],
                        [-0.36787945,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ]]]
        """
        if not bool(bivector) or not bool(multivector):
            np_result = np.array([])
            if torch_output:
                return torch.from_numpy(np_result)
            if tf_output:
                return tf.convert_to_tensor(np_result)
            if dict_output:
                return np.array({})
            return np_result

        if isinstance(multivector, str):
            # [P,f] = -X_f, for any function f.
            return self.num_hamiltonian_vf(
                bivector, f'(-1) * ({multivector})', mesh,
                torch_output=torch_output, tf_output=tf_output, dict_output=dict_output
            )

        len_keys = []
        for e in bivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F'repeated indexes {e} in {multivector}')
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F'invalid key {e} in {multivector}')
            len_keys.append(len(e))
        if len(set(len_keys)) > 1:
            raise MultivectorError('keys with different lengths')

        len_keys = []
        for e in multivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F'repeated indexes {e} in {multivector}')
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F'invalid key {e} in {multivector}')
            len_keys.append(len(e))
        if len(set(len_keys)) > 1:
            raise MultivectorError('keys with different lengths')

        # Degree of multivector
        deg_mltv = len(next(iter(multivector)))

        if deg_mltv + 1 > self.pg.dim:
            np_result = np.array([])
            if torch_output:
                return torch.from_numpy(np_result)
            if tf_output:
                return tf.convert_to_tensor(np_result)
            if dict_output:
                return np.array({})
            return np_result

        image_mltv = self.pg.coboundary_operator(bivector, multivector)
        if not bool(image_mltv):
            np_result = np.array([])
            if torch_output:
                return torch.from_numpy(np_result)
            if tf_output:
                return tf.convert_to_tensor(np_result)
            if dict_output:
                return np.array({})
            return np_result
        image_mltv = {tuple(map(lambda x: x-1, list(e))): image_mltv[e] for e in image_mltv}
        permut_image = []
        for e in image_mltv:
            permut_image.append({x.permute(e): (x.sign) * image_mltv[e] for x in perm.Permutation.group(deg_mltv + 1)})
        for e in permut_image:
            image_mltv.update(e)
        dict_eval = dict_mesh_eval(image_mltv, mesh, self.pg.coords)

        np_result = []
        zero_tensor = zeros_array(deg_mltv + 1, self.pg.dim)
        for dictt in dict_eval:
            copy_zero_tensor = np.copy(zero_tensor)
            for e2 in dictt:
                copy_zero_tensor[e2] = dictt[e2]
            np_result.append(copy_zero_tensor)
        np_result = np.array(np_result)

        # return the result in a TensorFlow tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a PyTorch tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            dict_eval = [{tuple(map(lambda x: x+1, list(e))): dictt[e] for e in dictt} for dictt in dict_eval]
            return np.array(dict_eval)
        # return the result in Numpy array
        return np_result

    def num_one_forms_bracket(self, bivector, one_form_1, one_form_2,
                              mesh, torch_output=False, tf_output=False, dict_output=False):
        """
            Evaluates the Lie bracket of two differential 1-forms induced by a given Poisson bivector field in all
            points given of a mesh.

            The calculus is as {alpha,beta}_P := i_P#(alpha)(d_beta) - i_P#(beta)(d_alpha) + d_P(alpha,beta)
            for 1-forms alpha and beta, where d_alpha and d_beta are the exterior derivative of alpha and beta,
            respectively, i_ the interior product of vector fields on differential forms, P#: T*M -> TM the vector
            bundle morphism defined by P#(alpha) := i_(alpha)P, with i the interior product of alpha and P. Note that,
            by definition {df,dg}_P = d_{f,g}_P, for ant functions f,g on M.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        one_form_1/one_form_2:
            Is a 1-form differential in a dictionary format with tuple type 'keys' and string type 'values'.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :torch_output/tf_output:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of {one_form_1, one_form_2}_π
            This value can be converted to Tensor PyTorch, TensorFlow or dictionary for by setting their respective
            flag as True in the params.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg3 = NumPoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> # For one form alpha
            >>> one_form_1 = {(1,): '2', (2,): '1', (3,): '2'}
            >>> # For one form beta
            >>> one_form_2 = {(1,): '1', (2,): '1', (3,): '1'}
            >>> # Defines a simple mesh
            >>> mesh = np.array([[1., 1., 1.]])
            >>> # Evaluates the mesh into {one_form_1, one_form_2}_π
            >>> npg3.num_one_forms_bracket(bivector, one_form_1, one_form_2, mesh,)
            >>> [[[-1.]
                  [ 0.]
                  [ 1.]]]
            >>> npg3.num_one_forms_bracket(bivector, one_form_1, one_form_2, mesh, torch_output=True)
            >>> tensor([[[-1.],
                         [ 0.],
                         [ 1.]]], dtype=torch.float64)
            >>> npg3.num_one_forms_bracket(bivector, one_form_1, one_form_2, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[-1.]
                  [ 0.]
                  [ 1.]]], shape=(1, 3, 1), dtype=float64)
            >>> npg3.num_one_forms_bracket(bivector, one_form_1, one_form_2, mesh, dict_output=True)
            >>> array([{(1,): -1.0, (3,): 1.0}], dtype=object)
        """
        for e in bivector:
            if len(set(e)) < len(e):
                raise MultivectorError(F"repeated indexes {e} in {bivector}")
            if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                raise MultivectorError(F"invalid key {e} in {bivector}")

        if self.pg.is_in_kernel(bivector, one_form_1) and self.pg.is_in_kernel(bivector, one_form_2):
            np_result = np.array([])
            if torch_output:
                return torch.from_numpy(np_result)
            if tf_output:
                return tf.convert_to_tensor(np_result)
            if dict_output:
                return np.array({})
            return np_result

        if self.pg.is_in_kernel(bivector, one_form_1):
            form_1_vector = sym.zeros(self.dim + 1, 1)

            for e in bivector:
                if len(set(e)) < len(e):
                    raise MultivectorError(F"repeated indexes {e} in {bivector}")
                if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                    raise MultivectorError(F"invalid key {e} in {bivector}")
            for e in one_form_1:
                if e[0] <= 0:
                    raise DiferentialFormError(F"invalid key {e} in {one_form_1}")
                form_1_vector[int(*e)] = one_form_1[e]
            for e in one_form_2:
                if e[0] <= 0:
                    raise DiferentialFormError(F"invalid key {e} in {one_form_2}")

            form_1_vector = form_1_vector[1:, :]
            jac_form_1 = form_1_vector.jacobian(self.pg.coords)
            jac_form_1 = sym.lambdify(self.pg.coords, jac_form_1)
            jac_form_1_eval = [jac_form_1(*e) for e in mesh]
            sharp_form_2_eval = self.num_sharp_morphism(bivector, one_form_2, mesh)
            raw_result = map(lambda e1, e2: np.dot(e1.T - e1, e2), jac_form_1_eval, sharp_form_2_eval)
            np_result = np.array(tuple(raw_result))

            # return the result in a PyTorch tensor if the flag is True
            if torch_output:
                return torch.from_numpy(np_result)
            # return the result in a TensorFlow tensor if the flag is True
            if tf_output:
                return tf.convert_to_tensor(np_result)
            # return the result in dictionary type
            if dict_output:
                dicts = [{(i + 1,): e[i][0] for i in range(self.pg.dim) if e[i][0] != 0} for e in np_result]
                return np.array(dicts)
            # return the result in Numpy array
            return np_result

        if self.pg.is_in_kernel(bivector, one_form_2):
            form_2_vector = sym.zeros(self.pg.dim + 1, 1)

            for e in bivector:
                if len(set(e)) < len(e):
                    raise MultivectorError(F"repeated indexes {e} in {bivector}")
                if len(tuple(filter(lambda x: (x <= 0), e))) > 0:
                    raise MultivectorError(F"invalid key {e} in {bivector}")
            for e in one_form_1:
                if e[0] <= 0:
                    raise DiferentialFormError(F"invalid key {e} in {one_form_1}")
            for e in one_form_2:
                if e[0] <= 0:
                    raise DiferentialFormError(F"invalid key {e} in {one_form_2}")
                form_2_vector[int(*e)] = one_form_2[e]

            form_2_vector = form_2_vector[1:, :]
            jac_form_2 = form_2_vector.jacobian(self.pg.coords)
            jac_form_2 = sym.lambdify(self.pg.coords, jac_form_2)
            jac_form_2_eval = [jac_form_2(*e) for e in mesh]
            sharp_form_1_eval = self.num_sharp_morphism(bivector, one_form_1, mesh)
            raw_result = map(lambda e1, e2: np.dot(e1 - e1.T, e2), jac_form_2_eval, sharp_form_1_eval)
            np_result = np.array(tuple(raw_result))

            # return the result in a PyTorch tensor if the flag is True
            if torch_output:
                return torch.from_numpy(np_result)
            # return the result in a TensorFlow tensor if the flag is True
            if tf_output:
                return tf.convert_to_tensor(np_result)
            # return the result in dictionary type
            if dict_output:
                dicts = [{(i + 1,): e[i][0] for i in range(self.pg.dim) if e[i][0] != 0} for e in np_result]
                return np.array(dicts)
            # return the result in Numpy array
            return np_result

        form_1_vector = sym.zeros(self.pg.dim + 1, 1)
        form_2_vector = sym.zeros(self.pg.dim + 1, 1)

        for e in one_form_1:
            if e[0] <= 0:
                raise DiferentialFormError(F"invalid key {e} in {one_form_1}")
            form_1_vector[int(*e)] = one_form_1[e]
        for e in one_form_2:
            if e[0] <= 0:
                raise DiferentialFormError(F"invalid key {e} in {one_form_2}")
            form_2_vector[int(*e)] = one_form_2[e]

        form_1_vector = form_1_vector[1:, :]
        form_2_vector = form_2_vector[1:, :]
        jac_form_1 = form_1_vector.jacobian(self.pg.coords)
        jac_form_2 = form_2_vector.jacobian(self.pg.coords)
        jac_form_1 = sym.lambdify(self.pg.coords, jac_form_1)
        jac_form_2 = sym.lambdify(self.pg.coords, jac_form_2)
        jac_form_1_eval = [jac_form_1(*e) for e in mesh]
        jac_form_2_eval = [jac_form_2(*e) for e in mesh]
        sharp_form_1_eval = self.num_sharp_morphism(bivector, one_form_1, mesh)
        sharp_form_2_eval = self.num_sharp_morphism(bivector, one_form_2, mesh)
        raw_result_1 = map(lambda e1, e2: np.dot(e1 - e1.T, e2), jac_form_2_eval, sharp_form_1_eval)  # T1
        raw_result_2 = map(lambda e1, e2: np.dot(e1.T - e1, e2), jac_form_1_eval, sharp_form_2_eval)  # T2

        sharp_form_1 = self.pg.sharp_morphism(bivector, one_form_1)
        sharp_1 = sym.zeros(self.pg.dim + 1, 1)
        for e in sharp_form_1:
            sharp_1[int(*e)] = sharp_form_1[e]
        sharp_1 = sharp_1[1:, :]
        pair_form_2_sharp_1 = (form_2_vector.T * sharp_1)[0]
        dd_pair_form_2_sharp_1 = sym.Matrix(sym.derive_by_array(pair_form_2_sharp_1, self.pg.coords))
        dd_pair_form_2_sharp_1 = sym.lambdify(self.pg.coords, dd_pair_form_2_sharp_1)
        dd_pair_f2_s1_eval = [dd_pair_form_2_sharp_1(*e) for e in mesh]  # T3

        raw_result = map(lambda e1, e2, e3: e1 + e2 + e3, raw_result_1, raw_result_2, dd_pair_f2_s1_eval)
        np_result = np.array(tuple(raw_result))

        # return the result in a PyTorch tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            dicts = [{(i + 1,): e[i][0] for i in range(self.pg.dim) if e[i][0] != 0} for e in np_result]
            return np.array(dicts)
        # return the result in Numpy array
        return np_result

    def num_linear_normal_form_R3(self, linear_bivector, mesh, torch_output=False, tf_output=False, dict_output=False):
        """
            Evaluates a normal form for Lie-Poisson bivector fields on R^3 (modulo linear isomorphisms) in all
            points given of a mesh.
        Parameters
        ==========
        :linear_bivector:
            Is a Lie-Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :torch_output/tf_output:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of lineal normal form from a
            bivector. This value can be converted to Tensor PyTorch, TensorFlow or dictionary for by setting
            their respective flag as True in the params.The result is a normal form in a dictionary format with
            integer type 'keys' and symbol type 'values'.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg3 = NumPoissonGeometry(3)
            >>> # For bivector (3*x3 - x1)*Dx1^Dx2 - (x1 + 2*x2)*Dx1^Dx3 + (x1 + x2 - x3)*Dx2^Dx3
            >>> bivector = {(1, 2): '3*x3 - x1', (1, 3): '-(x1 + 2*x2)', (2, 3): 'x1 + x2 - x3'}
            >>> # Defines a simple mesh
            >>> mesh = np.array([[1., 1., 1.]])
            >>> # Evaluates the mesh
            >>> npg3.num_linear_normal_form_R3(bivector, mesh)
            >>> [[[ 0. -1. -1.]
                  [ 1.  0.  1.]
                  [ 1. -1.  0.]]]
            >>> npg3.num_linear_normal_form_R3(bivector, mesh, torch_output=True)
            >>> tensor([[[ 0., -1., -1.],
                         [ 1.,  0.,  1.],
                         [ 1., -1.,  0.]]], dtype=torch.float64)
            >>> npg3.num_linear_normal_form_R3(bivector, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[ 0. -1. -1.]
                  [ 1.  0.  1.]
                  [ 1. -1.  0.]]], shape=(1, 3, 3), dtype=float64)
            >>> npg3.num_linear_normal_form_R3(bivector, mesh, dict_output=True)
            >>> array([{(1, 2): -1.0, (1, 3): -1.0, (2, 3): 1.0}], dtype=object)
        """
        if self.dim != 3:
            raise DimensionError(F'The dimension {self.dim} != 3')
        # Get the lineal normal form from a bivector
        lin_normal_form = self.pg.linear_normal_form_R3(linear_bivector)
        np_result = self.num_bivector_to_matrix(lin_normal_form, mesh)

        # return the result in a PyTorch tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            return dict_mesh_eval(lin_normal_form, mesh, self.pg.coords)
        # return the result in Numpy array
        return np_result

    def num_gauge_transformation(self, bivector, two_form,
                                 mesh, torch_output=False, tf_output=False,
                                 dict_output=False):
        """
            This method evaluates the Gauge transformation of a Poisson bivector field in all points given of a mesh.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :two_form:
            Is a closed differetial form in a dictionary format with tuple type 'keys' and string type 'values'.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :torch_output/tf_output:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of lineal normal form from a
            bivector. This value can be converted to Tensor PyTorch, TensorFlow or dictionary for by setting their
            respective flag as True in the params.The result is a normal form in a dictionary format with integer
            type 'keys' and symbol type 'values'.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg = NumPoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> P_so3 = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> Lambda = {(1, 2): 'x1 - x2', (1, 3): 'x1 - x3', (2, 3): 'x2 - x3'}
            >>> # Calculate a simple mesh
            >>> mesh = np.array([[1., 1., 1.]])
            >>> # Evaluates the mesh
            >>> npg.num_gauge_transformation(P_so3, Lambda, mesh)
            >>> [[[ 0.  1. -1.]
                  [-1.  0.  1.]
                  [ 1. -1.  0.]]]
            >>> npg.num_gauge_transformation(P_so3, Lambda, mesh, torch_output=True)
            >>> tensor([[[ 0.,  1., -1.],
                         [-1.,  0.,  1.],
                         [ 1., -1.,  0.]]], dtype=torch.float64)
            >>> npg.num_gauge_transformation(P_so3, Lambda, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[ 0.  1. -1.]
                  [-1.  0.  1.]
                  [ 1. -1.  0.]]], shape=(1, 3, 3), dtype=float64)
            >>> npg.num_gauge_transformation(P_so3, Lambda, mesh, dict_output=True)
            >>> array([{(1, 2): 1.0, (1, 3): -1.0, (2, 3): 1.0}], dtype=object)
        """
        # Evaluates the bivectors into the mesh
        bivector_num_mat = self.num_bivector_to_matrix(bivector, mesh)
        two_form_num_mat = self.num_bivector_to_matrix(two_form, mesh)
        # Calculate the product of bivector matrices minus the identity matrix
        I_matrix = np.identity(self.dim)
        I_min_form_bivector = map(lambda e1, e2: I_matrix - np.dot(e1, e2), two_form_num_mat, bivector_num_mat)
        inv_I_min_form_bivector = [splinalg.inv(e) for e in I_min_form_bivector if splinalg.det(e) != 0]
        raw_result = map(lambda e1, e2: np.dot(e1, e2), bivector_num_mat, inv_I_min_form_bivector)
        np_result = np.array(tuple(raw_result))
        # return the result in a PyTorch tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            dicts = [dict(triu(csr_matrix(e), k=1).todok()) for e in np_result]
            return np.array([{(e[0]+1, e[1]+1): dictt[e] for e in dictt} for dictt in dicts])
        # return the result in Numpy array
        return np_result

    def num_flaschka_ratiu_bivector(self, casimirs, mesh, symplectic_form=False,
                                    torch_output=False, tf_output=False, dict_output=False):
        """
            Calculate a Poisson bivector from Flaschka-Ratui formula where all Casimir function is in "casimir"
            variable. This Poisson bivector is the following form:
                i_π Ω := dK_1^...^dK_m-2
            where K_1, ..., K_m-2 are casimir functions and (M, Ω) is a diferentiable manifold with volument form Ω
            and dim(M)=m.

        Parameters
        ==========
        :casimirs_list:
            Is a list of Casimir functions where each element is a string type.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :torch_output/tf_output:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of lineal normal form from a
            bivector. This value can be converted to Tensor PyTorch, TensorFlow or dictionary for by setting their
            respective flag as True in the params.The result is a normal form in a dictionary format with integer
            type 'keys' and symbol type 'values'.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 4
            >>> npg4 = NumPoissonGeometry(4)
            >>> # Defines Casimir functions
            >>> casimirs = ['x1**2 +x2**2 +x3**2', 'x4']
            >>> # Calculate a simple mesh
            >>> mesh = np.array([[1., 1., 1., 1.]])
            >>> # Evaluates the mesh into
            >>> npg4.num_flaschka_ratiu_bivector(casimirs, mesh)
            >>> [[[ 0. -2.  2.  0.]
                  [ 2.  0. -2.  0.]
                  [-2.  2.  0.  0.]
                  [ 0.  0.  0.  0.]]]
            >>> npg4.num_flaschka_ratiu_bivector(casimirs, mesh, torch_output=True)
            >>> tensor([[[ 0., -2.,  2.,  0.],
                         [ 2.,  0., -2.,  0.],
                         [-2.,  2.,  0.,  0.],
                         [ 0.,  0.,  0.,  0.]]], dtype=torch.float64)
            >>> npg4.num_flaschka_ratiu_bivector(casimirs, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[ 0. -2.  2.  0.]
                  [ 2.  0. -2.  0.]
                  [-2.  2.  0.  0.]
                  [ 0.  0.  0.  0.]]], shape=(1, 4, 4), dtype=float64)
            >>> npg4.num_flaschka_ratiu_bivector(bivector, mesh, dict_output=True)
            >>> array([{(1, 2): -2.0, (1, 3): 2.0, (2, 3): -2.0}], dtype=object)
        """
        if sym.simplify(self.pg.dim - 2) <= 0:
            raise DimensionError(F'The dimension {self.pg.dim} < 3')
        if len(casimirs) != self.pg.dim - 2:
            raise CasimirError(F"The length to Casimir's functions is distinct to {self.pg.dim - 2}")

        if symplectic_form:
            FR_bivector = self.pg.flaschka_ratiu_bivector(casimirs, symplectic_form=True)
            np_result_FR = self.num_bivector_to_matrix(FR_bivector[0], mesh)
            np_result_2f = self.num_bivector_to_matrix(FR_bivector[1], mesh)

            # return the result in a PyTorch tensor if the flag is True
            if torch_output:
                return torch.from_numpy(np_result_FR), torch.from_numpy(np_result_2f)
            # return the result in a TensorFlow tensor if the flag is True
            if tf_output:
                return tf.convert_to_tensor(np_result_FR), tf.convert_to_tensor(np_result_2f)
            # return the result in dictionary type
            if dict_output:
                return dict_mesh_eval(FR_bivector[0], mesh, self.pg.coords), dict_mesh_eval(FR_bivector[1], mesh, self.pg.coords)  # noqa: E501
            # return the result in Numpy array
            return np_result_FR, np_result_2f

        FR_bivector = self.pg.flaschka_ratiu_bivector(casimirs)
        np_result = self.num_bivector_to_matrix(FR_bivector, mesh)

        # return the result in a PyTorch tensor if the flag is True
        if torch_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            return dict_mesh_eval(FR_bivector, mesh, self.pg.coords)
        # return the result in Numpy array
        return np_result

    def num_modular_vf(self, bivector, function, mesh, torch_output=False, tf_output=False, dict_output=False):
        """
            Calculates the modular vector field Z of a given Poisson bivector field P relative to the volume form
            f*Omega_0 defined as Z(g):= div(X_g) where Omega_0 = dx1^...^dx('dim'), f a non zero function and div(X_g)
            is the divergence respect to Omega of the Hamiltonian vector field
            X_g of f relative to P.
            Clearly, the modular vector field is Omega-dependent. If h*Omega is another volume form, with h a nowhere
            vanishing function on M, then
                Z' = Z - (1/h)*X_h,
            is the modular vector field of P relative to h*Omega.

        Parameters
        ==========
        :bivector:
            Is a Poisson bivector in a dictionary format with tuple type 'keys' and string type 'values'.
        :function:
            Is a function scalar h: M --> R that is a non zero and is a string type.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :torch_output/tf_output:
            Is a boolean flag to indicates if the result is given in a tensor from PyTorch/TensorFlow, its
            default value is False.

        Returns
        =======
            The default result is a NumPy array that contains all the evaluations of the modular vector field Z of a
            given Poisson bivector field P relative to the volume.
            This value can be converted to Tensor PyTorch, TensorFlow or dictionary for by setting their respective
            flag as True in the params.The result is a normal form in a dictionary format with integer type 'keys'
            and symbol type 'values'.

        Example
        ========
            >>> import numpy as np
            >>> from poisson import NumPoissonGeometry
            >>> # Instance the class to dimension 3
            >>> npg3 = NumericalPoissonGeometry(3)
            >>> # Defines a simple bivector
            >>> bivector = {(1, 2): '1/4*x3*(x1**4 + x2**4 + x3**4)', (1, 3): '-1/4*x2*(x1**4 + x2**4 + x3**4)', (2, 3): '1/4*x1*(x1**4 + x2**4 + x3**4)'} # noqa
            >>> mesh = np.array([[1., 1., 1.,]])
            >>> # Evaluates the mesh into
            >>> npg3.num_modular_vf(bivector, 1, mesh)
            >>> [[[ 0.]
                  [ 0.]
                  [ 0.]]]
            >>> npg3.num_modular_vf(bivector, 1, mesh, torch_output=True)
            >>> tf.Tensor(
                [[[ 0.]
                  [ 0.]
                  [ 0.]]], shape=(1, 3, 1), dtype=float64)
            >>> npg3.num_modular_vf(bivector, 1, mesh, tf_output=True)
            >>> tensor([[[ 0.],
                         [ 0.],
                         [ 0.]]], dtype=torch.float64)
            >>> npg3.num_modular_vf(bivector, 1, mesh, dict_output=True)
            >>> array([{}], dtype=object)
        """
        bivector = sym.sympify(bivector)
        bivector = {e: (-1) * bivector[e] for e in bivector}
        return self.num_curl_operator(bivector, function,
                                      mesh, torch_output=torch_output,
                                      tf_output=tf_output, dict_output=dict_output)
