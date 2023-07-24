# Customized Uncertainty Quantification of Parking Duration Predictions for EV Smart Charging

This repository contains the Python implementation to recreate the graphs and error metrics introduced for 
customised uncertainty quantification of parking duration predictions for EV smart charging. This approach is presented 
in the following paper:
> Kaleb Phipps, Karl Schwenk, Benjamin Briegel, Ralf Mikut, and Veit Hagenmeyer. 2023.
> Customized Uncertainty Quantification of Parking Duration Predictions for EV Smart Charging. In The IEEE Internet of
> Things Journal.

## Repository Structure

This repository is structured in a few key folders:

- `error_metrics`: This folder contains the code used to create the error metrics we report in our paper.
- `example_data`: This folder contains example data which you can use to test the metrics and plots present in this repository.
- `results_plots`: This folder contains code to reproduce the plots presented in our paper.


## Installation and Execution

To test our metrics you should set up a virtual environment with Python 3.10 using e.g. venv (`python3.10 -m venv venv`) 
or Anaconda (`conda create -n env_name python=3.10`). You can then install the dependencies via `pip install -r requirements.txt`.

Now you should be able to open the `Example_Evaluation_Notebook.ipynb` and using the example data recreate the error
metrics and plots from our paper.

**Please Note:** Do not expect the results to be identical to our paper, since we have only provided a small subset of
example data. The example data is specifically provided so that, if you create your own predictions, you know which format
the data should be in.

## Funding

This project is supported by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI and by the
Helmholtz Association under the Program “Energy System Design”.

## License

This code is licensed under the [MIT License](LICENSE).
