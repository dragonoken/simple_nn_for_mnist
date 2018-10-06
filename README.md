# Requirements
python-mnist

PyTorch (You can download this via https://pytorch.org/)

numpy

matplotlib

# To-Do List

* Create a dataset class that contains (or rather loads) MNIST dataset with pytorch
* Modularize the classifier using pytorch base classes and neural network functions
* Make use of sophisticated optimization algorithms provided
* Implement CLR from ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186)
(CLR with cosine annealing is already provided, but I would like to implement the original ones in the paper.)

# Update History

### Current Status (Oct 4th 2018)

For the past few weeks, I've read through some papers and articles about several useful and effective techniques for machine learning. Most of them were very interesting and insightful, but they were quite sophisticated and probably too much of an overkill for this project. However, I did got some rather simple techniques I could use for this small project for better performance, more hands-on experience, and experiments!

One of them was a method to find a good learning rate. I've already implemented this, but it seemed like increasing the learning rate exponentially hardly gives a good result when finding the right learning rate, at least for this particular dataset and model. I read the original paper ["Cyclical Learning Rates for Training Neural Networks"](https://arxiv.org/abs/1506.01186) and then changed the function so that it tests out many learning rate values as it increases the learning rate linearly and not exponentially. Also, I changed the resulting plot from {loss} vs. {learning rate} to {accuracy} vs. {learning rate}. The result of this modification gave me a plot that was easy to interpret (since the accuracy ranges from 0 to 1, unlike the loss which is rather unbounded) and matched my intuition from over 10000 iterations of training about the optimal value.

Result from modified learning rate finding function

![modified learning rate plot](https://github.com/dragonoken/simple_nn_for_mnist/blob/master/plots/modified_lr_plot.png)

Also, I found some other ways of initializing weights in a way that gives good results in less number of iterations. Specifically, I implemented He et al initialization method and Xavier initialization method. They are now part of my 'reset' function, so I can re-initialize my weights and biases in one of 4 options: uniform, standard normal, He, and Xavier. This function now takes an optional key argument for specifying which method to use. After trying out each of them, I found a quite significant improvement in learning.

![train losses with different initialization methods](https://github.com/dragonoken/simple_nn_for_mnist/blob/master/plots/train_losses.png)

Oh, I almost forgot to mention it, but I also implemented Stochastic Gradient Descent! But it appears that the size of the data is small enough to just do Batch Gradient Descent as I've been doing. Not sure if it will give some boost in speed when using CPU, not GPU, though.

---

### Sep 17th 2018

I tried to implement an algorithm (from the lectures in fast.ai) for finding the best value for the learning rate.

This algorithm initially sets the learning rate to a small value, then, as it trains the parameters, it increases the learning rate exponentially while keeping track of past learning rates and corresponding loss values. When the loss value starts to "explode," the algorithm stops training --it detects the explosion by comparing the current loss value and another loss value from few steps back.

The algorithm, as a function, returns the entire lists of learning rates and corresponding loss values.

Here, I implemented a simple plotting fuction to visualize the learning rate curve.
![learning rate plot](https://github.com/dragonoken/simple_nn_for_mnist/blob/master/plots/lr_plot.png)

...which suggests that the model seems to be improving fast when the learning rate is around 0.1 or less.

However, when I actually tried the learning rate of 0.1 for 100 iterations, it hardly improved a bit. In fact, in the same amount of iteration, learning rate of 1 gave me a huge improvement in training accuracy from about 10% to almost 85%!

It's very likely that I have implemented this in a proper way... or at least I hope that's the case.

I'll try to look into the actual code and methods to fix this thing.

And also, I'm going to try and implement the "Gradient Descent with Reset", make use of the validation set I made to fit the hyperparameters, and change the overall structure of the neural network (adding more hidden layers! my graphics card is not working hard enough!).

---

### Sep 14th 2018

I made my first working model in Jupyter Notebook.
This model takes 784 inputs (28 * 28, each ranging from 0 to 1), has 1 hidden layer with 100 hidden units, and output 10 probability values each corresponding to the probability of the input being a specific number.

The model was trained on 60000 training examples of hand-written digits as 28 by 28 pixel images for about 10,000 iterations in total.

5 examples with the least loss value (most correct):
![5 most correct](https://github.com/dragonoken/simple_nn_for_mnist/blob/master/plots/most_correct_5.png)

5 examples with the most loss value (most incorrect):
![5 most incorrect](https://github.com/dragonoken/simple_nn_for_mnist/blob/master/plots/most_incorrect_5.png)
Those are hard for me, too....

It was my first time actually using a machine learning framework other than scikit-learn (I have tensorflow, but I haven't used it myself).
The problem of autograd not working was the biggest problem...
