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

import sympy as sym
import numpy as np
import tensorflow as tf
import torch
from scipy import linalg as splinalg
from poisson.poisson import PoissonGeometry
from numpoisson.utils import (matrix_of, vector_of, dict_mesh_eval, list_mesh_eval,
                           num_one_form_to_vector, show_coordinates, validate_dimension,
                           dicts_to_matrices, check_multivector_dim, create_tensor_base,
                           add_tensor_values)


# Check mensajes en todo el
class NumPoissonGeometry:
    """ This class provides some useful tools for Poisson-Nijenhuis calculus on Poisson manifolds."""

    def __init__(self, dimension, variable='x'):
        # Obtains the dimension
        self.dim = validate_dimension(dimension)
        # Define what variables the class will work with
        self.variable = variable
        # Create the symbolics symbols
        self.coords = sym.symbols(f'{self.variable}1:{self.dim + 1}')
        # Show the coordinates with that will the class works
        self.coordinates = show_coordinates(self.coords)
        # Intances Poisson Geometry package
        self.pg = PoissonGeometry(self.dim, self.variable)

    def num_bivector(self, bivector, mesh, pt_output=False, tf_output=False, dict_output=False):
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
            >>> bivector ={(1, 2): ‘x3’, (1, 3): ‘-x2’, (2, 3): ‘x1’}
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
            >>> npg3.num_bivector(bivector, mesh, pt_output=True)
                tensor([[[ 0.,  1., -0.],
                         [-1.,  0.,  0.],
                         [ 0., -0.,  0.]]], dtype=torch.float64)
            >>> npg3.num_bivector(bivector, mesh, dict_output=True)
            >>> [{(1, 2): 1.0, (1, 3): -0.0, (2, 3): 0.0}]
        """
        # Evaluates all point from the mesh in the bivector
        dict_list = np.array(dict_mesh_eval(bivector, mesh, self.coords))
        # save in a np array
        np_result = dicts_to_matrices(dict_list, self.dim)
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            return dict_list
        # return the result in Numpy array
        return np_result

    def num_bivector_to_matrix(self, bivector, mesh, pt_output=False, tf_output=False, dict_output=False):
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
            >>> npg3.num_bivector_to_matrix(bivector, mesh, pt_output=True)
            >>> tensor([[[ 0.,  1.,  0.],
                         [-1.,  0.,  0.],
                         [-0., -0.,  0.]]], dtype=torch.float64)
        """
        # Converts to bivector from dict format into matrix format
        bivector_matrix = matrix_of(bivector, self.dim)
        # Construct the lambdify function with the vector in a matrix format
        lamb_bivector_matrix = sym.lambdify(self.coords, bivector_matrix, 'numpy')
        # Evaluates all point from the mesh in the bivector and save in a np array
        np_result = np.array([lamb_bivector_matrix(*e) for e in mesh])
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # TODO add dict_output flag
        # return the result in Numpy array
        return np_result

    def num_sharp_morphism(self, bivector, one_form, mesh, tf_output=False, pt_output=False, dict_output=False):
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
            >>> npg3.num_sharp_morphism(bivector, one_form, mesh, pt_output=True)
            >>> tensor([[[-0.],
                         [-0.],
                         [-0.]]], dtype=torch.float64)
            >>> npg3.num_sharp_morphism(bivector, mesh, one_form, tf_output=True)
            >>> tf.Tensor([[[-0.]
                [-0.]
                [-0.]]], shape=(1, 3, 1), dtype=float64)
        """
        # Converts to bivector from dict format into matrix format
        bivector_num_mat = self.num_bivector_to_matrix(bivector, mesh)
        # Converts to one-form from dict format into vector-column format (matrix of dim x 1)
        one_form_num_vec = num_one_form_to_vector(one_form, mesh, self.dim, self.coords)
        # Making the product of bivector matrix with vector-column and saving the result in a Numpy array
        raw_result = map(lambda e1, e2: (-1) * np.dot(e1, e2), bivector_num_mat, one_form_num_vec)
        np_result = np.array(list(raw_result))
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # TODO add dict_output flag
        # return the result in Numpy array
        return np_result

    def num_hamiltonian_vf(self, bivector, ham_function, mesh, tf_output=False, pt_output=False, dict_output=False):
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
        :pt_output/tf_output:
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
            >>> npg3.num_hamiltonian_vf(bivector, ham_function, mesh, pt_output=True)
            >>> tensor([[[ 2.5000],
                         [-5.0000],
                         [ 2.5000]]], dtype=torch.float64)
            >>> npg3.num_hamiltonian_vf(bivector, mesh, ham_function, tf_output=True)
            >>> tf.Tensor(
                [[[ 2.5]
                  [-5. ]
                  [ 2.5]]], shape=(1, 3, 1), dtype=float64)
        """
        # Converts the hamiltonian_function from type string to symbolic expression
        ff = sym.sympify(ham_function)
        # Calculates the differential matrix of Hamiltonian function
        d_ff = sym.Matrix(sym.derive_by_array(ff, self.coords))
        # Evaluates the differential matrix of Hamiltonian function in each point from a mesh
        dff_num_vec = list_mesh_eval(d_ff, mesh, self.coords)
        # Evaluates all points from the mesh in the bivector matrix
        bivector_num_mat = self.num_bivector_to_matrix(bivector, mesh)
        # Calculate the product from bivector with the differential of Hamiltonian function to
        # each point from mesh and save the result in Numpy array
        raw_result = map(lambda e1, e2: (-1) * np.dot(e1, e2), bivector_num_mat, dff_num_vec)
        np_result = np.array(list(raw_result))
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # TODO add dict_output flag
        # return the result in Numpy array
        return np_result

    def num_poisson_bracket(self, bivector, function_1, function_2,
                            mesh, tf_output=False, pt_output=False, dict_output=False):
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
        :pt_output/tf_output:
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
            >>> npg3.num_poisson_bracket(bivector, function_1, function_2, mesh, pt_output=True)
            >>> tensor([-15.], dtype=torch.float64)
            >>> npg3.num_poisson_bracket(bivector, mesh, function_1, function_2, tf_output=True)
            >>> tf.Tensor([-15.], shape=(1,), dtype=float64)
        """

        # Convert from string to sympy value the function_2
        gg = sym.sympify(function_2)
        # Calculates the differential matrix of function_2
        d_gg = sym.Matrix(sym.derive_by_array(gg, self.coords))
        # Evaluates the differential matrix of function_2 in each point from a mesh and converts to Numpy array
        dgg_num_vec = np.array(tuple(list_mesh_eval(d_gg, mesh, self.coords)))
        # Evaluates the Hamiltonian vector field with function_1 in each point from a mesh and converts to Numpy
        ham_ff_num_vec = np.array(tuple(self.num_hamiltonian_vf(bivector, function_1, mesh)))
        # Calculate the product from differential matrix of function_2 with Hamiltonian vector field with function_1
        # to each point from mesh and save the result in Numpy array
        raw_result = map(lambda e1, e2: np.dot(e1.T, e2)[0][0], dgg_num_vec, ham_ff_num_vec)
        np_result = np.array(list(raw_result))
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # TODO add dict_output flag
        # return the result in Numpy array
        return np_result

    def num_curl_operator(self, multivector, function, mesh,
                          tf_output=False, pt_output=False, dict_output=False):
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
        :pt_output/tf_output:
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
            >>> npg4.num_curl_operator(bivector, function, mesh, dict_output=True)
            >>> [{0: 0}]
            >>> npg4.num_curl_operator(bivector, function, mesh, pt_output=True)
            >>> tensor([[[ 0.],
                         [ 0.],
                         [ 0.],
                         [ 0.]]], dtype=torch.float64)
            >>> npg4.num_curl_operator(bivector, mesh, function, tf_output=True)
            >>> tf.Tensor(
                [[[ 2.5]
                  [-5. ]
                  [ 2.5]]], shape=(1, 3, 1), dtype=float64)
        """
        curl_operator = self.pg.curl_operator(multivector, function)
        dict_eval = dict_mesh_eval(curl_operator, mesh, self.pg.coords)
        multivector_dim = check_multivector_dim(multivector) - 1
        # This block creates an evalutes numerically an multivector
        np_array = []
        for item in dict_eval:
            tensor_base, index_base = create_tensor_base(self.pg.dim, multivector_dim)
            result = add_tensor_values(item, tensor_base, index_base)
            np_array.append(result.tolist())
        # Add all evaluation in np array
        np_result = np.array(np_array)
        # return the result in dictionary type
        if dict_output:
            return dict_eval
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # return the result in Numpy array
        return np_result

    def num_coboundary_operator(self, bivector, multivector, mesh,
                                tf_output=False, pt_output=False, dict_output=False):
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
        :pt_output/tf_output:
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
            >>> npg3.num_coboundary_operator(bivector, multivector, mesh)
            >>> [[[ 0.        ,  0.36787945,  0.        ],
                  [-0.36787945,  0.        ,  0.        ],
                  [ 0.        ,  0.        ,  0.        ]]]
            >>> npg3.num_coboundary_operator(bivector, multivector, mesh, dict_output=True)
            >>> [{(1, 2): 0.36787944117144233}]
            >>> npg3.num_coboundary_operator(bivector, multivector, mesh, pt_output=True)
            >>> tensor([[[ 0.0000,  0.3679,  0.0000],
                         [-0.3679,  0.0000,  0.0000],
                         [ 0.0000,  0.0000,  0.0000]]], dtype=torch.float64)
            >>> npg3.num_coboundary_operator(bivector, mesh, multivector, tf_output=True)
            >>> tf.Tensor: shape=(1, 3, 3), dtype=float64, numpy=
                array([[[ 0.        ,  0.36787945,  0.        ],
                        [-0.36787945,  0.        ,  0.        ],
                        [ 0.        ,  0.        ,  0.        ]]]
        """
        image_bivector = self.pg.lichnerowicz_poisson_operator(bivector, multivector)
        dict_eval = dict_mesh_eval(image_bivector, mesh, self.pg.coords)
        multivector_dim = check_multivector_dim(bivector) + check_multivector_dim(multivector) - 1
        # This block creates an evalutes numerically an multivector
        np_array = []
        for item in dict_eval:
            tensor_base, index_base = create_tensor_base(self.pg.dim, multivector_dim)
            result = add_tensor_values(item, tensor_base, index_base)
            np_array.append(result.tolist())
        # Add all evaluation in np array
        np_result = np.array(np_array)
        # return the result in dictionary type
        if dict_output:
            return dict_eval
        # return the result in a TensorFlow tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a PyTorch tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # return the result in Numpy array
        return np_result

    def num_one_form_bracket(self, bivector, one_form_1, one_form_2,
                             mesh, tf_output=False, pt_output=False, dict_output=False):
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
        :pt_output/tf_output:
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
            >>> npg3.num_one_form_bracket(bivector, one_form_1, one_form_2, mesh,)
            >>> [[[-1.]
                  [ 0.]
                  [ 1.]]]
            >>> npg3.num_one_form_bracket(bivector, one_form_1, one_form_2, mesh, pt_output=True)
            >>> tensor([[[-1.],
                         [ 0.],
                         [ 1.]]], dtype=torch.float64)
            >>> npg3.num_one_form_bracket(bivector, one_form_1, one_form_2, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[-1.]
                  [ 0.]
                  [ 1.]]], shape=(1, 3, 1), dtype=float64)
        """
        """ This block calculates i_P#(alpha)(d_beta) """
        # Converts to one-form from dict format into matrix format
        beta_vect = vector_of(one_form_2, self.dim)
        # Calculates the differential matrix of one_form_2 parameter
        D_beta = sym.Matrix([sym.derive_by_array(e, self.coords) for e in beta_vect])
        # lambdify the differential matrix of one_form_2 parameter
        lamb_D_beta = sym.lambdify(self.coords, D_beta)
        # Evaluates the differential matrix of one_form_2 parameter in each point from a mesh
        D_beta_eval = [lamb_D_beta(*e) for e in mesh]
        # Evaluate the sharp morphism from bivector and one_form_1 parameters in each point from a mesh
        sharp_alpha_eval = self.num_sharp_morphism(bivector, one_form_1, mesh)
        # Result to i_P#(alpha)(d_beta)
        ii_sharp_alpha_D_beta_eval = map(lambda e1, e2: np.dot(e1 - e1.T, e2), D_beta_eval, sharp_alpha_eval)

        """ This block calculates i_P#(beta)(d_alpha) """
        # Converts to one-form from dict format into matrix format
        alpha_vect = vector_of(one_form_1, self.dim)
        # Calculates the differential matrix of one_form_1 parameter
        D_alpha = sym.Matrix([sym.derive_by_array(e, self.coords) for e in alpha_vect])
        # lambdify the differential matrix of one_form_1 parameter
        lamb_D_alpha = sym.lambdify(self.coords, D_alpha)
        # Evaluates the differential matrix of one_form_1 parameter in each point from a mesh
        D_alpha_eval = [lamb_D_alpha(*e) for e in mesh]
        # Evaluate the sharp morphism from bivector and one_form_1 parameters in each point from a mesh
        sharp_beta_eval = self.num_sharp_morphism(bivector, one_form_2, mesh)
        # Result to i_P#(beta)(d_alpha)
        ii_sharp_beta_D_alpha_eval = map(lambda e1, e2: np.dot(e1 - e1.T, e2), D_alpha_eval, sharp_beta_eval)

        """ This block calculate d_P(alpha,beta) that is the same to d(<beta,P#(alpha)>) """
        sharp_alpha_vect = vector_of(self.pg.sharp_morphism(bivector, one_form_1), self.dim)
        pairing_sharp_alpha_beta = (sym.transpose(sharp_alpha_vect) * beta_vect)[0]
        grad_pairing = sym.Matrix(sym.derive_by_array(pairing_sharp_alpha_beta, self.coords))
        # lambdify the grad_pairing variable
        lamb_grad_pairing = sym.lambdify(self.coords, grad_pairing)
        lamb_grad_pairing_eval = [lamb_grad_pairing(*e) for e in mesh]

        # Calculate {alpha, beta}_π to each point from mesh and save the result in Numpy array
        raw_result = map(lambda e1, e2, e3: e1 - e2 + e3, ii_sharp_alpha_D_beta_eval, ii_sharp_beta_D_alpha_eval, lamb_grad_pairing_eval)  # noqa
        np_result = np.array(list(raw_result))
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # return the result in Numpy array
        # TODO add dict_output flag
        return np_result

    # Asks about this function.
    def num_linear_normal_form_R3(self, linear_bivector, mesh, tf_output=False, pt_output=False, dict_output=False):
        """
            Evaluates a normal form for Lie-Poisson bivector fields on R^3 (modulo linear isomorphisms) in all
            points given of a mesh.
        Parameters
        ==========
        :linear_bivector:
            Is a Lie-Poisson bivector in a dictionary format with integer type 'keys' and string type 'values'.
        :mesh:
            Is a numpy array where each value is a list of float values that representa a point in R^{dim}.
        :pt_output/tf_output:
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
            >>> npg3.num_linear_normal_form_R3(bivector, mesh, pt_output=True)
            >>> tensor([[[ 0., -1., -1.],
                         [ 1.,  0.,  1.],
                         [ 1., -1.,  0.]]], dtype=torch.float64)
            >>> npg3.num_linear_normal_form_R3(bivector, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[ 0. -1. -1.]
                  [ 1.  0.  1.]
                  [ 1. -1.  0.]]], shape=(1, 3, 3), dtype=float64)
            >>> npg3.num_linear_normal_form_R3(bivector, mesh, dict_output=True)
        """
        # Get the lineal normal form from a bivector
        lin_normal_form = self.pg.linear_normal_form_R3(linear_bivector)
        # Evaluates the linear normal form from a bivector in each point from mesh and save the result in Numpy array
        dict_list = dict_mesh_eval(lin_normal_form, mesh, self.pg.coords)
        np_result = dicts_to_matrices(dict_list, self.pg.dim)
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            return dict_list
        # return the result in Numpy array
        return np_result

    def num_gauge_transformation(self, bivector, two_form, mesh, tf_output=False, pt_output=False, dict_output=False):
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
        :pt_output/tf_output:
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
            >>> npg3 = NumPoissonGeometry(3)
            >>> # For bivector x3*Dx1^Dx2 - x2*Dx1^Dx3 + x1*Dx2^Dx3
            >>> bivector = {(1, 2): 'x3', (1, 3): '-x2', (2, 3): 'x1'}
            >>> two_form = {(1, 2): 'x1 - x2', (1, 3): 'x1 - x3', (2, 3): 'x2 - x3'}
            >>> # Calculate a simple mesh
            >>> mesh = np.array([[1., 1., 1.]])
            >>> # Evaluates the mesh
            >>> npg3.num_linear_normal_form_R3(bivector, mesh)
            >>> [[[ 0.  1. -1.]
                  [-1.  0.  1.]
                  [ 1. -1.  0.]]]
            >>> npg3.num_linear_normal_form_R3(bivector, mesh, pt_output=True)
            >>> tensor([[[ 0.,  1., -1.],
                         [-1.,  0.,  1.],
                         [ 1., -1.,  0.]]], dtype=torch.float64)
            >>> npg3.num_linear_normal_form_R3(bivector, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[ 0.  1. -1.]
                  [-1.  0.  1.]
                  [ 1. -1.  0.]]], shape=(1, 3, 3), dtype=float64)
        """
        # Evaluates the bivectors into the mesh
        bivector_num_mat = self.num_bivector_to_matrix(bivector, mesh)
        two_form_num_mat = self.num_bivector_to_matrix(two_form, mesh)
        # Calculate the product of bivector matrices minus the identity matrix
        id_matrix = np.identity(self.dim)
        id_min_product_form_bivector = map(lambda e1, e2: id_matrix - np.dot(e1, e2), two_form_num_mat, bivector_num_mat)  # noqa
        # Check this line
        inv_id_min_product_form_bivector = [splinalg.inv(e) for e in id_min_product_form_bivector if splinalg.det(e) != 0]  # noqa
        raw_result = map(lambda e1, e2: np.dot(e1, e2), bivector_num_mat, inv_id_min_product_form_bivector)
        np_result = np.array(list(raw_result))
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            return tf.convert_to_tensor(np_result)
        # TODO add dict_output flag
        # return the result in Numpy array
        return np_result

    def num_flaschka_ratiu_bivector(self, casimirs_list, mesh, tf_output=False, pt_output=False, dict_output=False):
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
        :pt_output/tf_output:
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
            >>> npg4.num_flaschka_ratiu_bivector(bivector, mesh)
            >>> [[[ 0. -2.  2.  0.]
                  [ 2.  0. -2.  0.]
                  [-2.  2.  0.  0.]
                  [ 0.  0.  0.  0.]]]
            >>> npg4.num_flaschka_ratiu_bivector(bivector, mesh, pt_output=True)
            >>> tensor([[[ 0., -2.,  2.,  0.],
                         [ 2.,  0., -2.,  0.],
                         [-2.,  2.,  0.,  0.],
                         [ 0.,  0.,  0.,  0.]]], dtype=torch.float64)
            >>> npg4.num_flaschka_ratiu_bivector(bivector, mesh, tf_output=True)
            >>> tf.Tensor(
                [[[ 0. -2.  2.  0.]
                  [ 2.  0. -2.  0.]
                  [-2.  2.  0.  0.]
                  [ 0.  0.  0.  0.]]], shape=(1, 4, 4), dtype=float64)
            >>> npg4.num_flaschka_ratiu_bivector(bivector, mesh, dict_output=True)
            >>> [{(1, 2): -2.0, (1, 3): 2.0, (2, 3): -2.0}]
        """
        flaschka_ratiu_bivector = self.pg.flaschka_ratiu_bivector(casimirs_list)
        # Evaluates the linear normal form from a bivector in each point from mesh and save the result in Numpy array
        dict_list = dict_mesh_eval(flaschka_ratiu_bivector, mesh, self.pg.coords)
        np_result = dicts_to_matrices(dict_list, self.pg.dim)
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # return the result in dictionary type
        if dict_output:
            return dict_list
        # return the result in Numpy array
        return np_result

    def num_modular_vf(self, bivector, function, mesh, pt_output=False, tf_output=False, dict_output=False):
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
        :pt_output/tf_output:
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
            >>> bivector = {e: f'1/2*(x1**2 + x2**2 + x3**2 + x4**2)' for e in tuple(itls.combinations(range(1, dim + 1), 2))}  # noqa
            >>> mesh = np.array([[1., 1., 1.,]])
            >>> # Evaluates the mesh into
            >>> npg3.num_modular_vf(bivector, 1, mesh)
            >>> [[[-2.]
                  [ 0.]
                  [ 2.]]]
            >>> npg3.num_modular_vf(bivector, 1, mesh, pt_output=True)
            >>> tf.Tensor(
                [[[-2.]
                  [ 0.]
                  [ 2.]]], shape=(1, 3, 1), dtype=float64)
            >>> npg3.num_modular_vf(bivector, 1, mesh, tf_output=True)
            >>> tensor([[[-2.],
                         [ 0.],
                         [ 2.]]], dtype=torch.float64)
            >>> npg3.num_modular_vf(bivector, 1, mesh, dict_output=True)
            >>> [{(1,): -2.0, (2,): 0.0, (3,): 2.0}]
        """
        # Evaluates the modluar vector field in each point from mesh and save the result in Numpy array
        modular_vf = self.pg.modular_vf(bivector, function)
        dict_list = dict_mesh_eval(modular_vf, mesh, self.pg.coords)
        np_result = dicts_to_matrices(dict_list, self.pg.dim)
        # return the result in a TensorFlow tensor if the flag is True
        if tf_output:
            # TODO apply lambdify method to tensorflow
            return tf.convert_to_tensor(np_result)
        # return the result in a PyTorch tensor if the flag is True
        if pt_output:
            return torch.from_numpy(np_result)
        # return the result in dictionary type
        if dict_output:
            return dict_list
        # return the result in Numpy array
        return np_result
