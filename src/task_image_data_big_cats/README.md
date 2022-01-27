# Image Classification and Clustering using Classical Pattern Recognition methods

### Feature Extraction and Dimension reduction
1. For feature generation, run [image_features_generation.ipynb](image_features_generation.ipynb). This notebook saves the generated dimension reduced features as npy files.
2. For visualizing the features, run [image_feature_visualizer.ipynb](image_feature_visualizer.ipynb).

### Image Classification
The following are the steps to run the code notebooks in the given order.  All the results will be logged in csv files.
1. For cross validation experiments, run [image_classification_cross_validation_experiments.ipynb](image_classification_cross_validation_experiments.ipynb). This runs grid search cross validation experiments for different classifiers using different features. It **can take a few days** to completely run all the experiments because of the larger space of grid search space.
2. For test experiments, run [image_classification_test_experiments.ipynb](image_classification_test_experiments.ipynb). This runs test experiments for all the classifiers. This may take half an hour to completely run all the experiments.
3. For ensemble experiments, run [image_classification_ensemble_experiments.ipynb](image_classification_ensemble_experiments.ipynb). This may take an hour to completely run all the experiments.
4. For generating plots for various classification experiments, run [generate_plots.ipynb](generate_plots.ipynb).

### Image Clustering
* For running image clustering experiments, run [run_image_clustering_experiments.ipynb](run_image_clustering_experiments.ipynb).
