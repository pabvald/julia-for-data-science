# Julia-for-data-science course 

 - [Datasets](#datasets)
 - [Data](#data)
 - [Pre-processing](#preprocessin)
 - [Linear Algebra](#linear-algebra) 
 - [Statistics](#statistics)
 - [Dimensionality Reduction](#dimensionality-reduction)
 - [Clustering](#clustering)
 - [Decision trees](#decision-trees)
 - [SVM](#svm) 
 - [Linear models](#linear-modelsa)
 - [Graphs](#graphs)
 - [Numerical Optimization](#numerical-optimization)
 - [Neural Networks](#neural-networks)
 - [From other languages](#from-other-languages)
 - [Visualization](#visualization)

**This respository is based in the [tutorial](https://github.com/JuliaAcademy/DataScience) created for [JuliaAcademy](https://juliaacademy.com/) and taught by [Huda Nassar](https://github.com/nassarhuda).**

Below are shown the most important Julia packages for different topics, most of them used in the course.

## Datasets 
  - [RDatasets](https://github.com/JuliaStats/RDatasets.jl): collection of datasets included in the R language.
  - [VegaDatasets](https://github.com/queryverse/VegaDatasets.jl): collection of datasets used in Vega and Vega-Lite examples.
  
## Preprocessing
#TODO
## Data

General file reading/writing libraries:

  - [DelimitedFiles](): included in the Standard Librarty. Allows to read and write complicated files. It should be used only when the file is really complicated.
  - [CSV](https://juliadata.github.io/CSV.jl/stable/): allows to work with *.csv* files. It is faster than DelimitedFiles and converts the read file in a DataFrame.
  - [XLSX](https://felipenoris.github.io/XLSX.jl/stable/): is a Julia package to read and write Excel spreadsheet files. It allows to read whole sheets and particular ranges of cells.

DataFrames in Julia: 
  - [DataFrames.jl](https://juliadata.github.io/DataFrames.jl/stable/).
  
Libraries to read/write files with specific formats: 
   - [JSON.jl](https://github.com/JuliaIO/JSON.jl): Tprovides for parsing and printing JSON in pure Julia.
   - [JLD.jl](https://github.com/JuliaIO/JLD.jl/blob/master/doc/jld.md): the JLD module reads and writes "Julia data files" (*.jld files) using HDF5.The key characteristic is that objects of many types can be written, and upon later reading they maintain the proper type. 
   - [NPZ.jl](https://github.com/fhs/NPZ.jl): The NPZ package provides support for reading and writing Numpy .npy and .npz files in Julia.
   - [RData.jl](https://github.com/JuliaData/RData.jl):Read R data files (.rda, .RData) and optionally convert the contents into Julia equivalents. Can read any R data archive, although not all R types could be converted into Julia.
   - [MAT.jl](https://github.com/JuliaIO/MAT.jl): This library can read MATLAB .mat files, both in the older v5/v6/v7 format, as well as the newer v7.3 format.
   
   
   
## Linear Algebra

  - [LinearAlgebra.jl](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/): in addition to (and as part of) its support for multi-dimensional arrays, Julia provides native implementations of many common and useful linear algebra operations.
  
  - [SparseArrays.jl](https://docs.julialang.org/en/v1/stdlib/SparseArrays/): Julia has support for sparse vectors and sparse matrices.
  
  

## Statistics

  - [Statistics.jl](https://docs.julialang.org/en/v1/stdlib/Statistics/): The Statistics standard library module contains basic statistics functionality (std,var, cor, cov, mean, median, middle, quantile).
  
  - [StatsBase.jl](https://juliastats.org/StatsBase.jl/stable/): a Julia package that provides basic support for statistics. Particularly, it implements a variety of statistics-related functions, such as scalar statistics, high-order moment computation, counting, ranking, covariances, sampling, and empirical density estimation.
  
  - [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl): kernel density estimators for Julia.
  - [Distributions.jl](https://juliastats.org/Distributions.jl/latest/):provides everything needeed to work with probability distributions.
  - [HypothesisTesting.jl](https://juliastats.org/HypothesisTests.jl/stable/): implements several hypothesis test in Julia.
  - [MlBase.jl](https://mlbasejl.readthedocs.io/en/latest/): s a Julia package that provides useful tools for machine learning applications (confusion matrix, ROC curves). 
  
See also [Visualization](#visualization).


## Dimensionality Reduction 
  - [MultivariateStats](https://multivariatestatsjl.readthedocs.io/en/stable/index.html):  is a Julia package for multivariate statistical analysis. It provides a rich set of useful analysis techniques, such as PCA, CCA, LDA, PLS, etc.
  - [TSne](https://github.com/lejon/TSne.jl): Julia implementation of L.J.P. van der Maaten and G.E. Hintons t-SNE visualisation technique.
  - [UMAP](https://github.com/dillondaudert/UMAP.jl): A pure Julia implementation of the Uniform Manifold Approximation and Projection dimension reduction algorithm.  
   - [ScikitLearn.descomposition](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition):  includes matrix decomposition algorithms, including among others PCA, NMF or ICA.
  
  
## Clustering

  - [Clustering.jl](https://juliastats.org/Clustering.jl/stable/): a julia package for data clustering. It covers two aspets of data clustering: algorithms(k-means, k-medoids, dbscan, hierarchical clustering) and validation(silhouettes, v-measure).
  - [Distances.jl](https://github.com/JuliaStats/Distances.jl): a Julia package for evaluating distances(metrics) between vectors.
  - [ScikitLearn.cluster](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster): module gathers popular unsupervised clustering algorithms.


## Nearest neighbors
  - [NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl): a package written in Julia to perform high performance nearest neighbor searches in arbitrarily high dimensions.  
  

## SVM
  - [ScikitLearn.svm](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
  - [LIBSVM.jl](https://github.com/JuliaML/LIBSVM.jl)
  

## Decision trees
  - [DecisionTree.jl](https://juliahub.com/docs/DecisionTree/pEDeB/0.10.8/): Julia implementation of Decision Tree (CART) and Random Forest algorithms. Available via AutoMLPipeline.jl, CombineMLP.jl, MLJ.jl and ScikitLearn.jl.



## Linear models

  - [GLM.jl](https://juliastats.org/GLM.jl/stable/): linear and generalized linear models in Julia.
  - [LsqFit.jl](https://julianlsolvers.github.io/LsqFit.jl/latest/): basic least-squares fitting in pure Julia under an MIT license.
  - [ScikitLearn.linear_models](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model):module implements a variety of linear models.
  - [ANOVA.JL](https://github.com/marcpabst/ANOVA.jl): calculate ANOVA tables for linear models.


## Graphs
#TODO


## Numerical Optimization 
#TODO


## Neural Networks 

Julia offers different possibilities to work with neural networks:
   - [Flux.jl](https://fluxml.ai/Flux.jl/stable/): the Julia Machine Learning Library. 
   - [Knet.jl](https://denizyuret.github.io/Knet.jl/stable/): Koç University deep learning framework.
   - [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/stable/): Julia Machine Learning framework by Alan Turing institute.
   - [MXNet.jl](https://mxnet.apache.org/versions/1.6/api/julia/docs/api/): Apache MXNet Julia package.
   - [TensorFlow.jl](https://malmaud.github.io/TensorFlow.jl/stable/): a Julia wrapper for TensorFlow.
   - [ScikitLearn](https://cstjean.github.io/ScikitLearn.jl/dev/): Julia implementation of the scikit-learn Python library.


## From other languages 
   - [PyCall](https://github.com/JuliaPy/PyCall.jl): allows to import any Python package as well as our own Python code. 
   - [RCall](http://juliainterop.github.io/RCall.jl/stable/): facilitates communication between the R and Julia languages and allows the user to call R packages from within Julia.
   - [Calling C and Fortran Code](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/) (in Julia documentation).


## Visualization 
  - [Plots.jl](http://docs.juliaplots.org/latest/).
  - [StatsPlots.jl](https://github.com/JuliaPlots/StatsPlots.jl): statistical plotting recipes for Plots.jl
  - [Makie](https://makie.juliaplots.org/stable/): high level plotting library with a focus on interactivity and speed.

