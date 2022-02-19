
Files that can be run are "k-fold-ensemble.py", "k-fold-decision-tree.py", k-fold-knn.py", "k-fold-random-forest.py" and "k-means-clustering.py"

So for example "python3 ensemble.py"

Each of the files has some global parameters that can be adapted accordingly, these are located in the top of the file. Note that runtime may be long, especially for the ensemble. 

The programs also assume that there is a folder "Genes" containing "data.csv" and "labels.csv". (The location of the folder can be adapted using the global "PATH" parameter.)