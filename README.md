[![Python](https://img.shields.io/pypi/pyversions/numericalpoissongeometry.svg?style=plastic)]()
[![PyPI](https://badge.fury.io/py/numericalpoissongeometry.svg)](https://pypi.org/project/numericalpoissongeometry/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/git/git-scm.com/blob/master/MIT-LICENSE.txt)
![GitHub last commit](https://img.shields.io/github/last-commit/appliedgeometry/numericalpoissongeometry)
<!--[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://colab.research.google.com/drive/1XYcaJQ29XwkblXQOYumT1s8_00bHUEKZ) -->

---
# Numerical Poisson Geometry
A Python module for (local) Poisson-Nijenhuis calculus on Poisson manifolds, with the following functions:

| **num_bivector_field**        | **num_bivector_to_matrix**    | **num_poisson_bracket**           |
| :---------------------------: | :---------------------------: | :------------------------------:  |
| **num_hamiltonian_vf**        | **num_sharp_morphism**        | **num_coboundary_operator**       |
| **num_modular_vf**            | **num_curl_operator**         | **num_one_forms_bracket**         |
| **num_gauge_transformation**  | **num_linear_normal_form_R3** | **num_flaschka_ratiu_bivector**   |


This repository accompanies our paper ['On Computational Poisson Geometry II: Numerical Methods'](https://arxiv.org/abs/2010.09785).

<!-- For more information you can read the [wiki](https://github.com/mevangelista-alvarado/poisson_geometry/wiki) this project. or the our [documentation]()-->

## Motivation

This project includes numerical methods that implementation parts of:

* Miguel Evangelista-Alvarado, José C. Ruíz Pantaleón & P. Suárez-Serrato, <br/>
 [On Computational Poisson Geometry I: Symbolic Foundations](https://arxiv.org/pdf/1912.01746.pdf), <br/> 
   arXiv:1912.01746 [math.DG] (2019)


## 🚀
<!--- #### Testing: --->
<!-- Solo borrar esto
 * __Run our tutorial on Colab__ [English](https://colab.research.google.com/drive/1XYcaJQ29XwkblXQOYumT1s8_00bHUEKZ) / [Castellano](https://colab.research.google.com/drive/1F9I2TcrhSz0zRZSuALEWldxgw-AL6pOK)
   
 * __Run on your local machine__
   * Clone this repository on your local machine.
   * Open a terminal with the path where you clone this repository.
   * Create a virtual environment,(see this [link](https://gist.github.com/mevangelista-alvarado/8ee2fd663e7446e543fc04eacce0f303))
   * Install the requirements:
      ```
      (venv_name) C:Users/dekstop/poisson$ pip install poissongeometry
      ```
   * Open python terminal to start:
      ```
      (venv_name) C:Users/dekstop/poisson$ python
      ```
   * Import PoissonGeometry class
      ```
      >>> from poisson.poisson import PoissonGeometry
      ```	 
<!--  * Testing the class.
	   For example we want convert a bivector to a matrix  <a href="https://www.codecogs.com/eqnedit.php?latex=$$\pi=x_{3}\frac{\partial}{\partial&space;x_{1}}\wedge\frac{\partial}{\partial&space;x_{2}}&space;-&space;x_{2}\frac{\partial}{\partial&space;x_{1}}\wedge\frac{\partial}{\partial&space;x_{3}}&space;&plus;&space;x_{1}\frac{\partial}{\partial&space;x_{2}}\wedge\frac{\partial}{\partial&space;x_{3}}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\pi=x_{3}\frac{\partial}{\partial&space;x_{1}}\wedge\frac{\partial}{\partial&space;x_{2}}&space;-&space;x_{2}\frac{\partial}{\partial&space;x_{1}}\wedge\frac{\partial}{\partial&space;x_{3}}&space;&plus;&space;x_{1}\frac{\partial}{\partial&space;x_{2}}\wedge\frac{\partial}{\partial&space;x_{3}}$$" title="$$\pi=x_{3}\frac{\partial}{\partial x_{1}}\wedge\frac{\partial}{\partial x_{2}} - x_{2}\frac{\partial}{\partial x_{1}}\wedge\frac{\partial}{\partial x_{3}} + x_{1}\frac{\partial}{\partial x_{2}}\wedge\frac{\partial}{\partial x_{3}}$$" /></a>
	   then <a href="https://www.codecogs.com/eqnedit.php?latex=\pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi" title="\pi" /></a> is equivalent to ```{(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}``` in this class.
	   ```
	   >>> from poisson import PoissonGeometry
	   >>> # We instantiate the Poisson class for dimension 3
	   >>> pg = PoissonGeometry(3)
	   >>> pg.bivector_to_matrix({(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'})
	   Matrix([
	   [  0,  x3, -x2],
	   [-x3,   0,  x1],
	   [ x2, -x1,   0]])
	   ```
		
		This function has an option for output is in latex format string, for this, we change the flag latex_format to True (its default value is False) as shown below.
		
		```
		 >>> print(pg.bivector_to_matrix({(1,2): 'x3', (1,3): '-x2', (2,3): 'x1'}, latex_format=True))
		   \left[\begin{array}{ccc}0 & x_{3} & - x_{2}\\- x_{3} & 0 & x_{1}\\x_{2} & - x_{1} & 0\end{array}\right]
		```
		<!--For more information to how use this class you can read the [documentation]() or the our [wiki](https://github.com/mevangelista-alvarado/poisson_geometry/wiki)-->

<!--## TO DO
Calculate Poisson Cohomology with linear coefficients.-->

## Bugs & Contributions
Our issue tracker is at https://github.com/appliedgeometry/NumericalPoissonGeometry/issues. Please report any bugs that you find. Or, even better, if you are interested in our project you can fork the repository on GitHub and create a pull request.

## Licence 📄
MIT licence

## Authors ✒️
This work is developed and maintained by:
 * **José C. Ruíz Pantaleón** - [@jcrpanta](https://github.com/jcrpanta)
 * **Pablo Suárez Serrato** - [@psuarezserrato](https://github.com/psuarezserrato)
 * **Miguel Evangelista-Alvarado** - [@mevangelista-alvarado](https://github.com/mevangelista-alvarado)

## Thanks for citing our work if you use it! 🤓 ##
@misc{evangelistaalvarado2020computational,<br/>
      title={On Computational Poisson Geometry II: Numerical Methods}, <br/>
      author={M. Evangelista-Alvarado and J. C. Ruíz-Pantaleón and P. Suárez-Serrato},<br/>
      year={2020},<br/>
      eprint={2010.09785},<br/>
      archivePrefix={arXiv},<br/>
      primaryClass={math.DG}<br/>
}

## Acknowledgments
This work was partially supported by the grants CONACyT, “Programa para un Avance Global e Integrado de la Matemática Mexicana” CONACyT-FORDECYT 26566 and "Aprendizaje Geométrico Profundo" UNAM-DGAPA-PAPIIT-IN104819. JCRP wishes to also thank CONACyT for a postdoctoral fellowship held during the production of this work.

---

<p align="center">
  <img src="https://www.matem.unam.mx/++theme++im-theme-blue/images/unam-escudo-azul.png">
  <img src="https://www.matem.unam.mx/++theme++im-theme-blue/images/logo_imunam.png">
</p>
