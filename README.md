# Recognition Using Deep Networks

PyTorch coursework on handwritten digit recognition and Greek-letter transfer learning.

- [MNIST example](pytorchexample.py)
- [Transfer-learning code](Transferlear.py)
- [Original run notes](readmep5.txt)
- [Project report](Project-5%20Report_Final.pdf)

The repository includes sample handwritten and Greek-letter images and saved model states under `results/` and `Transferlearn_results/`. Install PyTorch and torchvision and check the dataset and checkpoint paths in the selected script before running it.

## Fashion-MNIST experiment

`t4.py` is a separate Fashion-MNIST training experiment. Install compatible PyTorch and torchvision builds plus Matplotlib, then run `python t4.py` from the repository root. The script downloads the dataset to `./data` and reports model evaluation in the console. This workflow is distinct from the handwritten-digit and Greek-letter experiments.
