Fashion Categorizer Deep Neural Network
=======================================

This project is a custom implementation of a Deep Neural Network built from scratch using NumPy. It was created to apply and deepen the understanding of concepts learned in Coursera's *Neural Networks and Deep Learning* course.

The model classifies images from the Fashion MNIST dataset into one of 10 categories (e.g., T-shirt, Trouser, Pullover, etc.) and achieves a **Test Accuracy of 87.45%**.

Dataset
-------

The project uses the **Fashion MNIST** dataset.

-   **Source:** [Kaggle - Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

-   **Content:** 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale picture of a clothing item.

Project Structure
-----------------

Plaintext

```
fashion_categorizer/
├── data/                   # Directory for dataset files (not included in repo)
│   ├── fashion-mnist_train.csv
│   └── fashion-mnist_test.csv
├── neural_network.py       # Core DNN class and helper functions (ReLU, Softmax, etc.)
├── main.py                 # Main script to train the model and evaluate performance
├── main.ipynb              # Jupyter Notebook version of the training pipeline
├── test.ipynb              # Unit tests for individual network components
├── training_curve.png      # Generated plot of the learning curve
└── .gitignore

```

Prerequisites
-------------

To run this project, you need Python installed along with the following libraries:

-   **NumPy:** For matrix operations and linear algebra.

-   **Pandas:** For loading the CSV datasets.

-   **Matplotlib:** For plotting the cost/learning curve.

You can install these dependencies using pip:

Bash

```
pip install numpy pandas matplotlib jupyter

```

Setup & Installation
--------------------

1.  **Clone the repository:**

    Bash

    ```
    git clone https://github.com/jiabinc0602-collab/fashion_categorizer
    cd fashion_categorizer

    ```

2.  **Download the Data:**

    -   Go to the [Kaggle Fashion MNIST page](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

    -   Download `fashion-mnist_train.csv` and `fashion-mnist_test.csv`.

    -   Create a folder named `data` in the project root and place the CSV files inside it.

    *Your directory should look like this:*

    Plaintext

    ```
    fashion_categorizer/data/fashion-mnist_train.csv
    fashion_categorizer/data/fashion-mnist_test.csv

    ```

Usage
-----

### Option 1: Running the Python Script

To train the model and see the results immediately, run the main script:

Bash

```
python main.py

```

**What this does:**

1.  Loads the data.

2.  Normalizes pixel values (scales them to 0-1).

3.  Trains a 4-layer Neural Network (784 -> 128 -> 64 -> 10).

4.  Prints the cost every 100 iterations.

5.  Saves the learning curve graph to `training_curve.png`.

6.  Evaluates the model on the test set and prints the accuracy.

### Option 2: Jupyter Notebooks

-   **`main.ipynb`**: Contains the full training flow in an interactive notebook format.

-   **`test.ipynb`**: Contains unit tests to verify that functions like `one_hot`, `softmax`, and `relu` are working correctly before training.

To launch them:

Bash

```
jupyter notebook

```

Model Architecture
------------------

The implementation uses a fully connected Deep Neural Network with the following architecture:

-   **Input Layer:** 784 units (corresponding to 28x28 pixels)

-   **Hidden Layer 1:** 128 units (ReLU activation)

-   **Hidden Layer 2:** 64 units (ReLU activation)

-   **Output Layer:** 10 units (Softmax activation for multi-class classification)

**Hyperparameters:**

-   Learning Rate: 0.1

-   Epochs: 2000

Results
-------

-   **Test Accuracy:** 87.45%

-   **Learning Curve:** The model minimizes cost effectively over 2000 iterations.