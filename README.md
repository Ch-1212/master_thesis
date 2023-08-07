# master-thesis Barth

This folder contains the code written to detect operating states in cold water temperature data

The data is used is from a Berlin daycare center for children from the 08.04.2022 to the 16.05.2023.
It lays in the data folder. The data preprocessing file is used to preprocess the data and make it usuable for the following process.

Then, three different methods are applied on the data. We have feature extraction + k-means clustering, time series k-means clustering and SAX plus time series k-means clustering. For each method, we conducted some parameter tests, did a training process and then, tested the resulting models on the testing data.

The results can be found in data/results/ and created graphics in data/graphics/

The three methods are compared using the Compare_methods files.