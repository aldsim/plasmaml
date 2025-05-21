# Plasma ALD Dataset

## Introduction

This repository contains the supporting information for the publication
``Surrogate models to optimize plasma assisted atomic layer deposition in
high aspect ratio features`` to be published in Physics of Plasmas.

In this work, we have generated a dataset containing simulated thickness
profiles for thin film growth inside a high aspect ratio feature by
plasma-enhanced atomic layer deposition.

The goal of the task is to predict the saturation dose time based on these
inputs.


## Dataset structure

The dataset is broken down into training and testing sets. Each set comprises the following files:

- ``data``: contains two thickness profiles stored sequentially in a 1D array
- ``beta``: contains the ALD reaction probabilities
- ``betarec``: contains the surface recombination probability for the dominant plasma species
- ``labels``: contains the predicted multiplier required to achieve full saturation




