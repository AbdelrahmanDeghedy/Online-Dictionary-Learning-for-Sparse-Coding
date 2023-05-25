# Online-Dictionary-Learning-for-Sparse-Coding

## Introduction

This repository contains the implementation of the online learning algorithms for sparse coding proposed by Julien Mairal in his paper titled _"Online Learning for Matrix Factorization and Sparse Coding"_.

Ssparse coding is a fundamental technique in machine learning and signal processing, with numerous applications in recommendation systems, image processing, and data analysis. While traditional approaches often rely on _batch learning_ algorithms, _online learning_ methods provide a more efficient and scalable solution, enabling _real-time_ updates and adaptivity to changing data streams.

Julien Mairal's paper presents novel online learning algorithms that tackle the challenges of matrix factorization and sparse coding in an online setting. These algorithms not only provide accurate representations of high-dimensional data but also incorporate efficient online updates, making them suitable for large-scale applications.

## Results

### 1. Plotting the Objective Function VS Time _(using the online method, batch size of 4, batch size of 10)_

- We used 100 iterations on this example.

![](./results/Objective_function_vs._time%2C_using_random_columns_of_the_dataset_as_initialization_for_D.png)

### 2. Effect of Increasing the Dictionary Size on the Reconstructed Signal

We study the effect of varying _K_ (number of columns in the dictionary), by plotting the Difference Matrix _(between the original dataset X, and reconstructed dataset x_hat)_

- This heat map illustrates the absolute difference between the original dataset values, and the entries of the reconstructed matrix.
- The darker the color, the more closer the reconstructed entries to the original ones.
- We just used 5 iterations on this example.

### K = 20

1. First Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration1%2C%20with%20K%20%3D%2020).png>)

2. Final (Fifth) Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration5%2C%20with%20K%20%3D%2020).png>)

### K = 40

1. First Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration1%2C%20with%20K%20%3D%2040).png>)

2. Final (Fifth) Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration5%2C%20with%20K%20%3D%2040).png>)

### K = 60

1. First Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration1%2C%20with%20K%20%3D%2060).png>)

2. Final (Fifth) Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration5%2C%20with%20K%20%3D%2060).png>)

### K = 80

1. First Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration1%2C%20with%20K%20%3D%2080).png>)

2. Final (Fifth) Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration5%2C%20with%20K%20%3D%2080).png>)

### K = 100

1. First Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration1%2C%20with%20K%20%3D%20100).png>)

2. Final (Fifth) Iteration

![](<./img/Matrix_Reconstruction_Difference_(Iteration5%2C%20with%20K%20%3D%20100).png>)

> **Conclusion**: The larger the value of K _(hence the larger the dictionary)_, the faster the convergence of the reconstructed signal to the original one.

### 3. Sparsity

The following two heat maps illustrates the values of both the dictionary $D$ and one of the the sparse coding coefficients columns $\alpha$. It is evident the sparsity in each of them.

![](<./img/Dictionary_(Iteration100%2C_with_K_%3D_40)%2C_with_D_initialized_randomly.png>)

![](<./img/Alpha_(Iteration100%2C%20with%20K%20%3D%2040).png>)

## Slides

The slides for our presentation about this paper, as a project for the **Optimization EEC 494** class, are available [here](./Slides/Online%20Dictionary%20Learning%20for%20Sparse%20Coding%20Slides.pdf)

## Thanks

Special thanks to the amazing people that contributed to this project:

1. Pensee Safwat
2. Dyaa Mohamed
3. Abdelrahman Ali
4. Fatma Ezzat
5. Me :)

## Contribution

Thank you for your interest in contributing to this research paper code repository! Your contributions can greatly enhance the quality and impact of the project. To contribute, please follow the guidelines outlined below.
Bug Reports and Issues

If you encounter any bugs, have questions, or want to report an issue related to the code in this repository, please create a new issue on the GitHub repository page. When creating an issue, provide a clear and concise description of the problem you encountered, along with steps to reproduce it if applicable. This will help us understand and address the issue promptly.
Feature Requests

If you have a feature request or an idea for improving the code, please submit a new issue on the GitHub repository. Clearly describe the proposed feature or enhancement, providing as much detail as possible. We appreciate well-documented feature requests that include the rationale behind the suggestion and any potential implementation approaches.
