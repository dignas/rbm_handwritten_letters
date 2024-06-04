# Restricted Boltzmann machine. Theory and application for handwritten letters recognition

This project is a part of a master's thesis on restricted Boltzmann machines (RBMs).

## Data cleanup

Theory for this part can be found in the master's thesis in section 2.3.

In order to run the code:

```
python3.12 code/imgs_to_01.py -d dataset -o dataset_clean
```

Cleaned data is already in the folder `dataset_clean`.

## The experiment

Described in section 2.6.

In order to run the code:

```
python3.12 code/rbm/rbm_logistic_comparison.py -d dataset_clean -o save_rbm -D cyryllic
```

Trained models are already in the folder `save_rbm`. [Pickle](https://docs.python.org/3/library/pickle.html) format is used.

The complete output can be found in `output.txt`.

The classification report is additionally saved in `output_classification.txt`.
