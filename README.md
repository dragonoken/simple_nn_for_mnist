## Requirements
python-mnist

PyTorch (You can download this via https://pytorch.org/)

numpy

matplotlib

## Current Status (Sep 14th 2018)
I made my first working model in Jupyter Notebook.
This model takes 784 inputs (28 * 28, each ranging from 0 to 1), has 1 hidden layer with 100 hidden units, and output 10 probability values each corresponding to the probability of the input being a specific number.

The model was trained on 60000 training examples of hand-written digits as 28 by 28 pixel images for about 10,000 iterations in total.

5 examples with the least loss value (most correct):
![5 most correct](https://github.com/dragonoken/simple_nn_for_mnist/blob/master/most_correct_5.png)

5 examples with the most loss value (most incorrect):
![5 most incorrect](https://github.com/dragonoken/simple_nn_for_mnist/blob/master/most_incorrect_5.png)
Those are hard for me, too....

It was my first time actually using a machine learning framework other than scikit-learn (I have tensorflow, but I haven't used it myself).
The problem of autograd not working was the biggest problem...
