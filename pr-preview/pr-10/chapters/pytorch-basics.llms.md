# Tensors and Automatic Differentiation

Code

- [Show All Code](javascript:void(0))
- [Hide All Code](javascript:void(0))

Published

Last modified: 2026-10-25

Show code

``` downlit
# The Python chunks on this page import these; reticulate provisions them.
# Attaching reticulate gives the inline R its `py` object.
library(reticulate)
source("../tools/use-torch.R")
use_torch(c("numpy", "pandas", "matplotlib"))
```

These notes follow the order of topics in Brian Hutchinson’s CSCI 481/581, used with his permission; his module on the course’s tools is the starting point for this page. The exercises are new.

# 1 Welcome

## 1.1 Plan for this lecture

All code in these notes, and in the homework, uses PyTorch [\[1\]](#ref-paszke2019pytorch). This lecture covers the two parts of PyTorch that everything else is built on, on the handwritten digits from the [course overview](../chapters/course-overview.llms.md):

1.  [tensors](#def-tensor), the arrays that hold a network’s inputs, [parameters](https://morrison-lab.github.io/lds/chapters/big-ideas.html#def-parameter) and outputs;
2.  [automatic differentiation](#def-automatic-differentiation), which computes the derivative of a loss from the code that computes the loss.

# 2 Tensors

## 2.1 The digits as a tensor

> **NOTE:**
>
> **Exercise 1 (The digits as a tensor)** Load the digits from [`data/optdigits-test.csv`](data/optdigits-test.csv), the local copy of the [UCI data](https://archive.ics.uci.edu/dataset/80) [\[2\]](#ref-alpaydin1998optdigits) that the course overview uses. Put their pixels into one PyTorch [tensor](#def-tensor) that holds each image as an 8 by 8 grid, and their labels into another. What is the first tensor’s [shape](#def-shape)?

> **NOTE:**
>
> *Solution 1*.
>
> Show code
>
> ``` python
> import numpy as np
> import pandas as pd
> import torch
> import matplotlib.pyplot as plt
>
> digits = pd.read_csv("data/optdigits-test.csv", header=None)
> pixels = digits.iloc[:, :64].to_numpy()
> images = torch.tensor(pixels.reshape(-1, 8, 8))
> labels = torch.tensor(digits[64].to_numpy())
> n_images = images.shape[0]
> print(images.shape, labels.shape)
> ```
>
>     torch.Size([1797, 8, 8]) torch.Size([1797])
>
> Show code
>
> ``` python
> print(images[0])
> ```
>
>     tensor([[ 0,  0,  5, 13,  9,  1,  0,  0],
>             [ 0,  0, 13, 15, 10, 15,  5,  0],
>             [ 0,  3, 15,  2,  0, 11,  8,  0],
>             [ 0,  4, 12,  0,  0,  8,  8,  0],
>             [ 0,  5,  8,  0,  0,  9,  8,  0],
>             [ 0,  4, 11,  0,  1, 12,  7,  0],
>             [ 0,  2, 14,  5, 10, 12,  0,  0],
>             [ 0,  0,  6, 13, 10,  0,  0,  0]])
>
> `torch.tensor` copies a NumPy array into a [tensor](#def-tensor). The images make a tensor of [shape](#def-shape) (1797, 8, 8): a stack of 1797 grids, each with 8 rows and 8 columns. The labels make a tensor of shape (1797,), one number per image.

> **NOTE:**
>
> **Definition 1 (Tensor)** A **tensor** is an array of numbers arranged on a regular grid with any number of dimensions [\[3, Sec. 2.1\]](#ref-goodfellow2016deep). Every entry is picked out by one index per dimension.
>
> A tensor with no dimensions is a scalar, one with one dimension is a vector, and one with two is a matrix. PyTorch says “dimensions” (`ndim`, and `dim=` in its functions), where NumPy, which also says `ndim`, calls them “axes” (`axis=`); other sources say “modes”, or call the number of dimensions the tensor’s “order” or “rank”. That “rank” is not the rank of a matrix. Physics and mathematics use “tensor” for a multilinear map, which an array of numbers represents once a basis is chosen; [deep learning](../chapters/course-overview.llms.md#def-deep-learning) means only the array.

> **NOTE:**
>
> **Example 1 (The digits at three sizes)**  
>
> Show code
>
> ``` python
> print(images.ndim, images[0].ndim, images[0, 3].ndim, images[0, 3, 4].ndim)
> ```
>
>     3 2 1 0
>
> Show code
>
> ``` python
> print(images[0, 3])
> ```
>
>     tensor([ 0,  4, 12,  0,  0,  8,  8,  0])
>
> Show code
>
> ``` python
> print(images[0, 3, 4])
> ```
>
>     tensor(0)
>
> `images` from [Solution 1](#sol-load-tensor) has three dimensions: image, row and column. Indexing it once, as `images[0]`, gives the first image, a matrix; indexing twice, as `images[0, 3]`, gives that image’s row 3, a vector of 8 pixels; and indexing three times, as `images[0, 3, 4]`, gives one pixel, a scalar, with rows and columns counted from 0.

> **NOTE:**
>
> **Definition 2 (Shape)** A [tensor](#def-tensor)’s **shape** lists the size of each of its dimensions, in order. PyTorch reports it as `.shape`, or `.size()`; NumPy also calls it `.shape`.

> **NOTE:**
>
> **Example 2 (The shapes of the digits)**  
>
> Show code
>
> ``` python
> print(images.shape, images[0].shape, images[0, 3].shape, images[0, 3, 4].shape)
> ```
>
>     torch.Size([1797, 8, 8]) torch.Size([8, 8]) torch.Size([8]) torch.Size([])
>
> `images` has [shape](#def-shape) (1797, 8, 8): 1797 images, each with 8 rows and 8 columns. Indexing it drops the dimensions that the index fixes: one image has shape (8, 8), one of its rows has shape (8,), and one pixel has the empty shape ().

## 2.2 Data types

> **NOTE:**
>
> **Exercise 2 (The average pixel)** Find the average of the 64 pixels of the first image with the [tensor](#def-tensor) method `.mean()`. What goes wrong, and how can it be fixed?

> **NOTE:**
>
> *Solution 2*.
>
> Show code
>
> ``` python
> import textwrap
>
> print(images.dtype)
> ```
>
>     torch.int64
>
> Show code
>
> ``` python
> try:
>     images[0].mean()
> except RuntimeError as err:
>     print(textwrap.fill(str(err), 60))
> ```
>
>     mean(): could not infer output dtype. Input dtype must be
>     either a floating point or complex dtype. Got: Long
>
> Show code
>
> ``` python
> first_mean = images[0].float().mean().item()
> print(images[0].float().dtype, first_mean)
> ```
>
>     torch.float32 4.59375
>
> The pixels were read as whole numbers, so the [tensor](#def-tensor) stores 64-bit integers, `torch.int64`, which PyTorch also calls `Long`. `.mean()` refuses them, since an average of integers is not in general an integer. Converting the tensor with `.float()` to 32-bit floating-point numbers fixes it: the first image’s average pixel is 4.59, on the file’s scale of 0 to 16.

> **NOTE:**
>
> **Definition 3 (Data type)** A [tensor](#def-tensor)’s **data type** (in PyTorch, its `dtype`) is the kind of number that every one of its entries stores, such as 64-bit integers (`torch.int64`) or 32-bit floating-point numbers (`torch.float32`, PyTorch’s default for real numbers). All the entries of one tensor share one data type. PyTorch’s [tensor attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html) page lists them all.

> **NOTE:**
>
> **Example 3 (Two data types for one image)**  
>
> Show code
>
> ``` python
> pixel_int = images[0, 3, 4]
> pixel_float = pixel_int.float()
> print(pixel_int, pixel_int.dtype)
> ```
>
>     tensor(0) torch.int64
>
> Show code
>
> ``` python
> print(pixel_float, pixel_float.dtype)
> ```
>
>     tensor(0.) torch.float32
>
> Show code
>
> ``` python
> print(pixel_int == pixel_float)
> ```
>
>     tensor(True)
>
> In [Solution 2](#sol-mean-pixel), the first image is `torch.int64` as loaded and `torch.float32` after `.float()`. One of its pixels, printed both ways, shows the difference: the float prints with a decimal point, the two compare as equal, and only the kind of number storing the value differs.

## 2.3 Reshaping and averaging

> **NOTE:**
>
> **Exercise 3 (Back to rows of 64)** Turn `images` into a [tensor](#def-tensor) with one row of 64 pixels per image, the arrangement the file stores. Check that the result holds the same numbers as the file, in the same order.

> **NOTE:**
>
> *Solution 3*.
>
> Show code
>
> ``` python
> rows = images.reshape(-1, 64)
> print(rows.shape, torch.equal(rows, torch.tensor(pixels)))
> ```
>
>     torch.Size([1797, 64]) True
>
> `reshape(-1, 64)` asks for rows of 64 and lets PyTorch work out the number of rows, the `-1`. It reads the entries in order, last dimension fastest, so each image’s row 0 comes first, then its row 1, and so on, which undoes the `reshape(-1, 8, 8)` of [Solution 1](#sol-load-tensor). `torch.equal` confirms that the [shapes](#def-shape) and every entry match. The course overview’s [remark on the two arrangements](../chapters/course-overview.llms.md#rem-grid-or-vector) says which models take which.

> **NOTE:**
>
> **Exercise 4 (The average image of each digit)** Compute the average image of each digit, 0 to 9, as one [tensor](#def-tensor) of [shape](#def-shape) (10, 8, 8), and draw the ten averages.

> **NOTE:**
>
> *Solution 4*.
>
> > **NOTE:**
> >
> > Show code
> >
> > ``` python
> > averages = torch.stack(
> >     [images[labels == k].float().mean(dim=0) for k in range(10)]
> > )
> > print(averages.shape)
> > ```
> >
> > Show code
> >
> > ``` python
> > def draw_digits(grids):
> >     """Draw each 8 by 8 grid in `grids` side by side, titled by its index."""
> >     fig, axes = plt.subplots(1, len(grids), figsize=(8, 1.2), layout="constrained")
> >     for k, ax in enumerate(axes):
> >         ax.imshow(grids[k], cmap="gray_r", vmin=0, vmax=16)
> >         ax.set_title(str(k))
> >         ax.set_axis_off()
> >     plt.show()
> >
> > draw_digits(averages)
> > ```
> >
> >     torch.Size([10, 8, 8])
> >
> > [![Ten small grayscale images in a row, the average image of each digit from 0 to 9. Each is a blurred but readable digit: the 0 is a ring, the 1 a vertical bar, and the others have the outline of their digit, darker where most images of it have ink.](pytorch-basics_files/figure-html/unnamed-chunk-8-1.png)](pytorch-basics_files/figure-html/unnamed-chunk-8-1.png "Figure 1: The average image of each digit among the 1797 images.")
> >
> > Figure 1: The average image of each digit among the 1797 images.
>
> `labels == k` is a [tensor](#def-tensor) of `True` and `False`, one per image, and indexing `images` with it keeps the images of digit `k`. `.mean(dim=0)` averages over dimension 0, the images, and leaves one average for each row and column. `torch.stack` puts the ten 8 by 8 averages along a new first dimension.

## 2.4 Tensors and NumPy arrays

> **NOTE:**
>
> **Exercise 5 (A tensor and its array)** [`torch.from_numpy`](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html) also makes a [tensor](#def-tensor) from a NumPy array. Make one from a copy of `pixels` this way, and another with `torch.tensor`. Change an entry of each tensor. What happens to the arrays?

> **NOTE:**
>
> *Solution 5*.
>
> Show code
>
> ``` python
> shared_array = pixels.copy()
> shared = torch.from_numpy(shared_array)
> shared[0, 0] = 99
>
> copied_array = pixels.copy()
> copied = torch.tensor(copied_array)
> copied[0, 0] = 99
>
> print(pixels[0, 0], shared_array[0, 0], copied_array[0, 0])
> ```
>
>     0 99 0
>
> The [tensor](#def-tensor) from `torch.from_numpy` uses the array’s own memory, so changing the tensor changed the array, from 0 to 99. `torch.tensor` copies the numbers, so its array kept 0. Sharing saves time and memory on large data, at the cost of a change in one place showing up in the other. The tensor method `.numpy()` shares memory the same way, in the other direction.

## 2.5 Devices

> **NOTE:**
>
> **Definition 4 (Device)** A [tensor](#def-tensor)’s **device** is the processor whose memory holds its entries, and on which operations on the tensor run: the CPU, or a GPU (in PyTorch, `"cuda"` for an NVIDIA GPU). PyTorch makes a tensor on the CPU unless told otherwise, `.to()` copies a tensor to another device, and an operation on two [tensors](#def-tensor) needs both on the same device.

> **NOTE:**
>
> **Example 4 (Where the digits live)**  
>
> Show code
>
> ``` python
> has_gpu = torch.cuda.is_available()
> print(images.device, has_gpu)
> ```
>
>     cpu False
>
> `images` was made without naming a [device](#def-device), so it lives on the CPU. `torch.cuda.is_available()` says whether this computer has an NVIDIA GPU that PyTorch can use; it returned `False` on the computer that built this page. Where it returns `True`, `images.to("cuda")` makes a copy of `images` on the GPU.

> **NOTE:**
>
> PyTorch’s [tensors tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) shows more ways to make, index and combine tensors. Its notes on [broadcasting](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) give the rules for an operation on two tensors of different [shapes](#def-shape).

# 3 Automatic differentiation

## 3.1 The soft 0-or-1 rule

The rest of this page differentiates the [soft 0-or-1 rule](../chapters/ml-review.llms.md#exr-log-loss-curve) of the [machine learning review](../chapters/ml-review.llms.md). Among the images of 0s and 1s, it gives each image the probability \\\hprob = \sigmoidf{\predvar - \param}\\ of being a 1, where \\\predvar\\ is the image’s [center ink](../chapters/course-overview.llms.md#sol-zero-or-one), its average pixel over rows 2 to 5 and columns 3 and 4, and \\\param\\ is the rule’s cutoff. Its mean [log loss](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#def-log-loss) over those \\\nobs\\ images is

\\ \hR(\param) = -\frac{1}{\nobs} \sumin \sb{\outvar_i \log \hprob_i + (1 - \outvar_i) \log (1 - \hprob_i)}, \\

where \\\outvar_i\\ is 1 if image \\i\\ is a 1 and 0 if it is a 0.

> **NOTE:**
>
> **Exercise 6 (The data for the soft rule)** From `images` and `labels`, make a [tensor](#def-tensor) of the center ink \\\predvar_i\\ of each image of a 0 or a 1, stored as 64-bit floating-point numbers, and a tensor of the matching \\\outvar_i\\.

> **NOTE:**
>
> *Solution 6*.
>
> Show code
>
> ``` python
> center_ink = images[:, 2:6, 3:5].double().mean(dim=(1, 2))
> is_01 = (labels == 0) | (labels == 1)
> ink_01 = center_ink[is_01]
> is_one = (labels[is_01] == 1).double()
> n_01 = len(ink_01)
> print(n_01)
> ```
>
>     360
>
> Show code
>
> ``` python
> print(ink_01[:5])
> ```
>
>     tensor([ 0.3750, 15.8750,  3.0000, 14.1250,  2.5000], dtype=torch.float64)
>
> Show code
>
> ``` python
> print(is_one[:5])
> ```
>
>     tensor([0., 1., 0., 1., 0.], dtype=torch.float64)
>
> `images[:, 2:6, 3:5]` keeps rows 2 to 5 and columns 3 and 4 of every image, and `.mean(dim=(1, 2))` averages over those rows and columns, leaving one number per image. `.double()` converts to 64-bit floats, `torch.float64`, so that the comparison with PyTorch in [Solution 8](#sol-autograd-slope) is not limited by 32-bit rounding. There are \\\nobs\\ = 360 images of 0s and 1s.

## 3.2 The slope by hand

> **NOTE:**
>
> **Exercise 7 (The slope of the log loss)** Find the derivative \\\dif{\hR}{\param}\\ of the mean [log loss](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#def-log-loss) with respect to the cutoff.
>
> > **TIP:**
> >
> > The [logistic function](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#exm-logistic-function) has derivative \\\sigmoid'(z) = \sigmoidf{z}\paren{1 - \sigmoidf{z}}\\ ([lds](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#thm-logistic-derivative)).

> **NOTE:**
>
> *Solution 7*. Write \\\obsloss{i}\\ for image \\i\\’s [log loss](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#def-log-loss), \\\obsloss{i} = -\outvar_i \log \hprob_i - (1 - \outvar_i) \log (1 - \hprob_i)\\. Its derivative with respect to \\\hprob_i\\ is
>
> \\ \dif{\obsloss{i}}{\hprob_i} = -\frac{\outvar_i}{\hprob_i} + \frac{1 - \outvar_i}{1 - \hprob_i}. \\
>
> Since \\\hprob_i = \sigmoidf{\predvar_i - \param}\\, the chain rule and the derivative of the logistic function give
>
> \\ \begin{aligned} \dif{\hprob_i}{\param} &= \sigmoid'(\predvar_i - \param) \cdot \dif{}{\param} (\predvar_i - \param) \\ &= \hprob_i (1 - \hprob_i) \cdot (-1) \\ &= -\hprob_i (1 - \hprob_i). \end{aligned} \\
>
> Multiplying the two,
>
> \\ \begin{aligned} \dif{\obsloss{i}}{\param} &= \dif{\obsloss{i}}{\hprob_i} \cdot \dif{\hprob_i}{\param} \\ &= \paren{-\frac{\outvar_i}{\hprob_i} + \frac{1 - \outvar_i}{1 - \hprob_i}} \cdot \paren{-\hprob_i (1 - \hprob_i)} \\ &= \outvar_i (1 - \hprob_i) - (1 - \outvar_i) \hprob_i \\ &= \outvar_i - \outvar_i \hprob_i - \hprob_i + \outvar_i \hprob_i \\ &= \outvar_i - \hprob_i. \end{aligned} \\
>
> \\\hR\\ is the mean of the \\\obsloss{i}\\, so its derivative is the mean of theirs:
>
> \\ \dif{\hR}{\param} = \frac{1}{\nobs} \sumin (\outvar_i - \hprob_i). \\

> **NOTE:**
>
> **Theorem 1 (The slope of the soft rule’s log loss)** The mean [log loss](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#def-log-loss) \\\hR\\ of the [soft 0-or-1 rule](../chapters/ml-review.llms.md#exr-log-loss-curve) has derivative
>
> \\ \dif{\hR}{\param} = \frac{1}{\nobs} \sumin (\outvar_i - \hprob_i) \\
>
> with respect to its cutoff \\\param\\.

> **NOTE:**
>
> *Proof*. By [Solution 7](#sol-log-loss-slope). The rule is a [logistic regression](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#def-logistic-regression) whose coefficient on \\\predvar\\ is fixed at 1 and whose intercept is \\-\param\\, so the formula is also minus the intercept component of [lds’s gradient of the log loss](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#thm-logistic-gradient), divided by \\\nobs\\ to give the mean.

> **NOTE:**
>
> **Example 5 (The slope at a cutoff of 5)**  
>
> Show code
>
> ``` python
> x_np, y_np = ink_01.numpy(), is_one.numpy()
> p_np = 1 / (1 + np.exp(-(x_np - 5)))
> share_of_ones = np.mean(y_np)
> mean_prob_at_5 = np.mean(p_np)
> slope_by_hand = share_of_ones - mean_prob_at_5
> print(share_of_ones, mean_prob_at_5, slope_by_hand)
> ```
>
>     0.5055555555555555 0.5473628863066621 -0.04180733075110654
>
> By [Theorem 1](#thm-log-loss-slope), the slope is the share of 1s among the images minus the average probability the rule gives them. The 1s make up a share of 0.506 of the 360 images, and at \\\param = 5\\ the rule’s average probability of a 1 is 0.547. So the slope at \\\param = 5\\ is -0.0418. It is negative, so raising the cutoff from 5 lowers the [log loss](https://morrison-lab.github.io/lds/chapters/logistic-regression.html#def-log-loss). The slope is 0 at the cutoff where the average probability equals the share of 1s.

## 3.3 The slope by autograd

> **NOTE:**
>
> **Exercise 8 (Asking PyTorch for the slope)** Write a function that computes \\\hR(\param)\\ with PyTorch operations. Make \\\param = 5\\ a [tensor](#def-tensor) created with `requires_grad=True`, compute \\\hR(5)\\, and pass both to [`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html). Compare what it returns with the slope computed by hand in [Example 5](#exm-log-loss-slope).

> **NOTE:**
>
> *Solution 8*.
>
> Show code
>
> ``` python
> def mean_log_loss(theta):
>     """Mean log loss of the soft 0-or-1 rule with cutoff `theta`."""
>     p = torch.sigmoid(ink_01 - theta)
>     return -torch.mean(is_one * torch.log(p) + (1 - is_one) * torch.log(1 - p))
>
> theta = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
> loss = mean_log_loss(theta)
> (slope_by_autograd,) = torch.autograd.grad(loss, theta)
> slope_gap = abs(slope_by_autograd.item() - slope_by_hand)
> print(loss)
> ```
>
>     tensor(0.0853, dtype=torch.float64, grad_fn=<NegBackward0>)
>
> Show code
>
> ``` python
> print(slope_by_autograd, slope_gap)
> ```
>
>     tensor(-0.0418, dtype=torch.float64) 2.0816681711721685e-17
>
> `torch.autograd.grad` returns -0.0418, the slope computed by hand in [Example 5](#exm-log-loss-slope), and the two differ by 2.1e-17. The code for \\\hR\\ says nothing about its derivative. Because `theta` was made with `requires_grad=True`, PyTorch recorded each operation that turned it into `loss` (the subtraction `ink_01 - theta`, the logistic function, and each operation after them, down to the mean and its change of sign), and `torch.autograd.grad` applied the chain rule to those operations, one at a time. The `grad_fn` that `loss` prints is the record of its last operation.

> **NOTE:**
>
> **Definition 5 (Automatic differentiation (autodiff))** **Automatic differentiation**, also called **autodiff**, computes the derivatives of a function that is given as code, by applying the chain rule to the elementary operations the code performs as it runs [\[1\]](#ref-paszke2019pytorch), [\[3, Sec. 6.5\]](#ref-goodfellow2016deep).
>
> PyTorch’s automatic differentiation is called autograd. The result is the exact derivative of the operations the code performs, up to floating-point rounding. Automatic differentiation is neither symbolic differentiation, which manipulates a formula for the function, nor numerical differentiation, which approximates a derivative from the function’s values at nearby points. lds’s [backpropagation](https://morrison-lab.github.io/lds/chapters/backpropagation.html) chapter derives by hand the derivatives that autodiff computes for a neural network.

> **NOTE:**
>
> **Example 6 (The cutoff’s slope by autodiff)** [Solution 8](#sol-autograd-slope) is [automatic differentiation](#def-automatic-differentiation): PyTorch was given only the code for \\\hR\\, and returned its derivative at \\\param = 5\\, which matches the formula of [Theorem 1](#thm-log-loss-slope) to within 2.1e-17.

## 3.4 Gradient descent with autograd

> **NOTE:**
>
> **Exercise 9 (Finding the cutoff by gradient descent)** Run 100 steps of [gradient descent](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#the-algorithm) on \\\hR\\, starting from \\\param = 0\\ with learning rate \\\learnrate = 4\\, with each step’s derivative from `torch.autograd.grad`. Where does \\\param\\ end up, compared with the lowest point on the grid of cutoffs in the [machine learning review](../chapters/ml-review.llms.md#sol-smooth-loss)?

> **NOTE:**
>
> *Solution 9*.
>
> Show code
>
> ``` python
> learning_rate = 4.0
> theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
> for step in range(100):
>     (slope,) = torch.autograd.grad(mean_log_loss(theta), theta)
>     with torch.no_grad():
>         theta -= learning_rate * slope
> theta_descent = theta.item()
> (final_slope,) = torch.autograd.grad(mean_log_loss(theta), theta)
>
> cutoffs = np.arange(0, 16.005, 0.01)
> grid_step = cutoffs[1] - cutoffs[0]
> with torch.no_grad():
>     grid_loss = [mean_log_loss(torch.tensor(c)).item() for c in cutoffs]
> theta_grid = cutoffs[np.argmin(grid_loss)]
> gap = abs(theta_descent - theta_grid)
> print(theta_descent, final_slope.item(), theta_grid, gap)
> ```
>
>     6.59960743783184 -6.833215346312323e-06 6.6000000000000005 0.00039256216816063727
>
> [Gradient descent](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#the-algorithm) ends at \\\param\\ = 6.5996, where the slope is -6.8e-06, and the grid’s lowest point is at 6.60, the cutoff found in [the review’s solution](../chapters/ml-review.llms.md#sol-smooth-loss). The two differ by 0.0004, less than the grid’s spacing of 0.01.
>
> The update runs inside `torch.no_grad()`, which tells PyTorch not to record it: the step is a change to \\\param\\, not part of the loss. Without it, PyTorch would record the in-place subtraction on `theta`, and it raises an error for a recorded in-place change to a [tensor](#def-tensor) that was created with `requires_grad=True`.

# References

\[1\]

A. Paszke *et al.*, “PyTorch: An imperative style, high-performance deep learning library,” in *Advances in neural information processing systems*, 2019. Available: <https://arxiv.org/abs/1912.01703>

\[2\]

E. Alpaydin and C. Kaynak, “Optical recognition of handwritten digits.” UCI Machine Learning Repository, 1998. doi: [10.24432/C50P49](https://doi.org/10.24432/C50P49).

\[3\]

I. Goodfellow, Y. Bengio, and A. Courville, *Deep learning*. MIT Press, 2016. Available: <https://www.deeplearningbook.org/>

Back to top
