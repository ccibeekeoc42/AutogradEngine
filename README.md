# AutogradEngine

In this repo, we build an auto gradient engine capable of backpropagation (just like TensorFlow or PyTorch). Please Enjoy!

### Software, Tools, and prerequisits

1. Access to Google Colab or some Jupyter Notebook.
2. Basic python programing.
3. Basic arithmetic & calculus knowledge

### Intro: What is an AutoGrad Engine?

An autograd engine (as used in TensorFlow and PyTorch) is a class used to calculate the derivatives (Jacobian-Vector product) at every differentiable point in the system.

It does this by keeping a record graph of all operations performed on a gradient-enabled object and creating an acyclic graph called the dynamic computational graph.
