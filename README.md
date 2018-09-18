## Requirements
python-mnist

PyTorch (You can download this via https://pytorch.org/)

numpy

matplotlib

## Current Status (Sep 17th 2018)
I tried to implement an algorithm (from the lectures in fast.ai) for finding the best value for the learning rate.

This algorithm initially sets the learning rate to a small value, then, as it trains the parameters, it increases the learning rate exponentially while keeping track of past learning rates and corresponding loss values. When the loss value starts to "explode," the algorithm stops training --it detects the explosion by comparing the current loss value and another loss value from few steps back.

The algorithm, as a function, returns the entire lists of learning rates and corresponding loss values.

Here, I implemented a simple plotting fuction to visualize the learning rate curve.
![learning rate plot](https://github.com/dragonoken/simple_nn_for_mnist/blob/master/lr_plot.png)

...which suggests that the model seems to be improving fast when the learning rate is around 0.1 or less.

However, when I actually tried the learning rate of 0.1 for 100 iterations, it hardly improved a bit. In fact, in the same amount of iteration, learning rate of 1 gave me a huge improvement in training accuracy from about 10% to almost 85%!

It's very likely that I have implemented this in a proper way... or at least I hope that's the case.

I'll try to look into the actual code and methods to fix this thing.

And also, I'm going to try and implement the "Gradient Descent with Reset", make use of the validation set I made to fit the hyperparameters, and change the overall structure of the neural network (adding more hidden layers! my graphics card is not working hard enough!).

## Sep 14th 2018
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
