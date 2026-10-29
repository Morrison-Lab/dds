# Normalization and residual connections

Code

- [Show All Code](javascript:void(0))
- [Hide All Code](javascript:void(0))

Published

Last modified: 2026-10-29

Show code

``` downlit
# The Python chunks on this page import these; reticulate provisions them.
# Attaching reticulate gives the inline R its `py` object.
library(reticulate)
source("../tools/use-torch.R")
use_torch(c("numpy", "pandas", "matplotlib"))
```

Brian Hutchinson’s CSCI 481/581 sets the order of topics in these notes, which follow it with his permission; this page covers the part of his module on optimization and [regularization](https://morrison-lab.github.io/lds/chapters/big-ideas.html#def-regularization) that deals with normalization and [residual connections](https://morrison-lab.github.io/lds/chapters/backpropagation.html#def-residual-connection). The exercises are new.

# 1 Welcome

## 1.1 Plan for this lecture

The [backpropagation chapter](../chapters/backpropagation.llms.md#sol-deep-grad-norms) found that the [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) of a deep [MLP](../chapters/feedforward-networks.llms.md#def-feedforward-network) can shrink layer by layer, and the [optimization chapter](../chapters/optimization.llms.md#sol-init-scale) chose starting weights to keep each layer’s values at a steady size. This lecture tries two more ways to train a network of 20 such layers:

1.  rescaling each layer’s values during training, across the images of a minibatch or across the [unit](../chapters/feedforward-networks.llms.md#def-unit)s of one image;
2.  adding each layer’s input to its output.

## 1.2 The digits and a deep network

Show code

``` python
import numpy as np
import pandas as pd
import torch
from torch import nn

_ = torch.manual_seed(481)
digits = pd.read_csv("data/optdigits-test.csv", header=None)
x = torch.tensor(digits.iloc[:, :64].to_numpy(), dtype=torch.float32) / 16
y = torch.tensor(digits[64].to_numpy())
n_train = 1200
order = torch.randperm(len(x))
x_train, y_train = x[order[:n_train]], y[order[:n_train]]
x_test, y_test = x[order[n_train:]], y[order[n_train:]]
loss_fn = nn.CrossEntropyLoss()


class Block(nn.Module):
    # One hidden layer of `width` ReLU units.
    # make_norm(width) builds a layer applied to its pre-activations;
    # nn.Identity leaves them unchanged.

    def __init__(self, width, make_norm=nn.Identity, residual=False):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = make_norm(width)
        self.residual = residual

    def forward(self, h):
        out = torch.relu(self.norm(self.linear(h)))
        if self.residual:
            return h + out
        return out


def make_deep(make_norm=nn.Identity, residual=False, depth=20, width=64, seed=1):
    # A first layer from the 64 pixels, `depth` blocks, then 10 logits.
    torch.manual_seed(seed)
    blocks = [Block(width, make_norm, residual) for _ in range(depth)]
    return nn.Sequential(nn.Linear(64, width), nn.ReLU(), *blocks, nn.Linear(width, 10))


def train(net, n_epochs=5, lr=0.01, batch_size=32, seed=0):
    # Minibatch SGD; returns the training loss and the test accuracy at the end.
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(n_epochs):
        net.train()
        shuffled = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, batch_size):
            batch = shuffled[start:start + batch_size]
            optimizer.zero_grad()
            loss_fn(net(x_train[batch]), y_train[batch]).backward()
            optimizer.step()
    net.eval()
    with torch.no_grad():
        train_loss = loss_fn(net(x_train), y_train).item()
        test_acc = (net(x_test).argmax(dim=1) == y_test).float().mean().item()
    return train_loss, test_acc


def first_layer_grad(net):
    # Size of the loss gradient for the first layer's weights, on 256 training images.
    net.train()
    net.zero_grad()
    loss_fn(net(x_train[:256]), y_train[:256]).backward()
    return net[0].weight.grad.norm().item()


def last_block_std(net):
    # Standard deviation of the last block's output, on 256 training images.
    net.train()
    with torch.no_grad():
        return net[:-1](x_train[:256]).std().item()


chance_loss = np.log(10)

print(make_deep()[:3])
```

    Sequential(
      (0): Linear(in_features=64, out_features=64, bias=True)
      (1): ReLU()
      (2): Block(
        (linear): Linear(in_features=64, out_features=64, bias=True)
        (norm): Identity()
      )
    )

The exercises use the digits split of the [feedforward networks chapter](../chapters/feedforward-networks.llms.md), with 1,200 training images. `make_deep()` builds an [MLP](../chapters/feedforward-networks.llms.md#def-feedforward-network) with 21 [hidden layers](../chapters/feedforward-networks.llms.md#def-hidden-layer) of 64 [ReLU](../chapters/feedforward-networks.llms.md#def-relu) [units](../chapters/feedforward-networks.llms.md#def-unit): a first layer from the pixels, then 20 copies of `Block`. `train()` runs 5 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) of minibatch stochastic [gradient descent](https://morrison-lab.github.io/mds/optimization.html#def-gradient-descent), with batch size 32 and [learning rate](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-learning-rate) 0.01. `first_layer_grad()` measures the size of the [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) that reaches the first layer. The exercises count a network as having learned something when its training loss after `train()` is below 90% of \\\log 10 \approx 2.303\\, the loss of predicting \\\frac{1}{10}\\ for every digit.

# 2 A deep network

## 2.1 Training 20 layers

> **NOTE:**
>
> **Exercise 1 (A 20-block network)** Does `make_deep()` learn anything in the 5 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) of `train()`?

> **NOTE:**
>
> *Solution 1*.
>
> Show code
>
> ``` python
> plain_grad = first_layer_grad(make_deep())
> plain_loss, plain_acc = train(make_deep())
> plain_std = last_block_std(make_deep())
> print(plain_grad, plain_loss, plain_acc, plain_std)
> ```
>
>     1.927740633789199e-09 2.3031463623046875 0.1072026789188385 0.035105034708976746
>
> No. Its training loss after 5 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) is 2.303, against \\\log 10 \approx 2.303\\ for probabilities of \\\frac{1}{10}\\, and it classifies 10.7% of the test images correctly. The [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) that reaches its first layer has size 1.9e-09, and the last block’s output has standard deviation 3.5e-02. So the gradient vanishes before it reaches the first layer, as in the [backpropagation chapter](../chapters/backpropagation.llms.md#def-vanishing-gradient).

# 3 Batch normalization

## 3.1 Definition

> **NOTE:**
>
> **Definition 1 (Batch normalization)** **Batch normalization** rescales each [pre-activation](../chapters/feedforward-networks.llms.md#def-pre-activation) of a layer across the examples of a minibatch. For a minibatch of \\B\\ examples, let \\z\_{ij}\\ be pre-activation \\j\\ for example \\i\\, with mean and variance over the minibatch
>
> \\ \mu_j = \frac{1}{B} \sumto{i}{B} z\_{ij}, \qquad \sigma_j^2 = \frac{1}{B} \sumto{i}{B} \paren{z\_{ij} - \mu_j}^2. \\
>
> Batch normalization replaces \\z\_{ij}\\ with
>
> \\ \gamma_j \frac{z\_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \smallconst}} + \beta_j, \\
>
> where \\\gamma_j\\ and \\\beta_j\\ are [parameters](https://morrison-lab.github.io/lds/chapters/big-ideas.html#def-parameter) learned with the rest and \\\smallconst\\ is a small constant that keeps the division finite [\[1\]](#ref-ioffe2015batch), [\[2, Sec. 8.7.1\]](#ref-goodfellow2016deep). When the network makes predictions, running averages of \\\mu_j\\ and \\\sigma_j^2\\ from training take the place of the minibatch’s. PyTorch’s running average of the variance divides by \\B - 1\\ instead of \\B\\ [\[3\]](#ref-pytorch_batchnorm1d).

> **NOTE:**
>
> **Example 1 (`nn.BatchNorm1d` on a minibatch)**  
>
> Show code
>
> ``` python
> batch_norm = nn.BatchNorm1d(64)
> pre = nn.Linear(64, 64)(x_train[:32]).detach()
> normed = batch_norm(pre).detach()
> print(pre.mean(dim=0)[:4], pre.std(dim=0, correction=0)[:4])
> ```
>
>     tensor([ 0.1275, -0.3611,  0.5323,  0.4476]) tensor([0.1708, 0.1277, 0.1898, 0.1267])
>
> Show code
>
> ``` python
> print(normed.mean(dim=0)[:4], normed.std(dim=0, correction=0)[:4])
> ```
>
>     tensor([4.4703e-08, 1.9372e-07, 1.1176e-08, 2.1607e-07]) tensor([0.9998, 0.9997, 0.9999, 0.9997])
>
> Show code
>
> ``` python
> largest_mean = normed.mean(dim=0).abs().max().item()
> std_min = normed.std(dim=0, correction=0).min().item()
> std_max = normed.std(dim=0, correction=0).max().item()
> ```
>
> `nn.BatchNorm1d(64)` applies [batch normalization](#def-batch-norm) to 64 [pre-activation](../chapters/feedforward-networks.llms.md#def-pre-activation)s. It starts with every \\\gamma_j = 1\\ and every \\\beta_j = 0\\ [\[3\]](#ref-pytorch_batchnorm1d). For 32 training images, the first 4 pre-activations of a random layer have the means and standard deviations on the first line. After batch normalization, on the second line, no column’s mean is farther from 0 than 3.5e-07, and the standard deviations run from 0.999 to 1.000.

## 3.2 A batch of one

> **NOTE:**
>
> **Exercise 2 (Predicting for one image)** Why can [batch normalization](#def-batch-norm) not use the mean and variance of the minibatch when a network makes a prediction for a single image?

> **NOTE:**
>
> *Solution 2*. A minibatch of one image has \\\mu_j = z\_{1j}\\ and \\\sigma_j^2 = 0\\, so every normalized [pre-activation](../chapters/feedforward-networks.llms.md#def-pre-activation) would be 0, whatever the image. The prediction would not depend on the image at all.
>
> Show code
>
> ``` python
> try:
>     _ = batch_norm(pre[:1])
> except ValueError as error:
>     batch_of_one_error = str(error)
> print(batch_of_one_error)
> ```
>
>     Expected more than 1 value per channel when training, got input size torch.Size([1, 64])
>
> Show code
>
> ``` python
> _ = batch_norm.eval()
> one_image_output = batch_norm(pre[:1])
> print(one_image_output.shape)
> ```
>
>     torch.Size([1, 64])
>
> `nn.BatchNorm1d` treats a minibatch of one differently in its two modes:
>
> - **Training mode.** It refuses the minibatch: “Expected more than 1 value per channel when training, got input size torch.Size(\[1, 64\])”.
> - **Evaluation mode.** It uses its running averages instead, so it accepts a single image.
>
> Evaluation mode is what makes predictions use the running averages. That is why `train()` calls `net.eval()` before it computes the test [accuracy](https://morrison-lab.github.io/lds/chapters/metrics.html#def-accuracy).

## 3.3 Batch normalization in the deep network

> **NOTE:**
>
> **Exercise 3 (Batch normalization in the 20-block network)** Does `make_deep(make_norm=nn.BatchNorm1d)` learn in the 5 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) of `train()`?

> **NOTE:**
>
> *Solution 3*.
>
> Show code
>
> ``` python
> batch_grad = first_layer_grad(make_deep(make_norm=nn.BatchNorm1d))
> batch_loss, batch_acc = train(make_deep(make_norm=nn.BatchNorm1d))
> batch_std = last_block_std(make_deep(make_norm=nn.BatchNorm1d))
> print(batch_grad, batch_loss, batch_acc, batch_std)
> ```
>
>     20.276771545410156 1.2505766153335571 0.7721943259239197 0.6066935658454895
>
> Yes. With [batch normalization](#def-batch-norm) in every block, the training loss after 5 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) is 1.251, and the network classifies 77.2% of the test images correctly. The [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) at the first layer has size 2.0e+01, against 1.9e-09 without it. Each block rescales its [pre-activation](../chapters/feedforward-networks.llms.md#def-pre-activation)s to standard deviation 1 before the [ReLU](../chapters/feedforward-networks.llms.md#def-relu), so the values cannot shrink from block to block: the last block’s output has standard deviation 0.607, against 3.5e-02 without batch normalization.

# 4 Layer normalization

## 4.1 Definition

> **NOTE:**
>
> **Definition 2 (Layer normalization)** **Layer normalization** rescales the [pre-activation](../chapters/feedforward-networks.llms.md#def-pre-activation)s of a layer across the [units](../chapters/feedforward-networks.llms.md#def-unit) of one example. For an example with pre-activations \\z_1, \dots, z_L\\, it uses their mean and variance
>
> \\ \mu = \frac{1}{L} \sumto{j}{L} z_j, \qquad \sigma^2 = \frac{1}{L} \sumto{j}{L} \paren{z_j - \mu}^2, \\
>
> and replaces each \\z_j\\ with \\\gamma_j \paren{z_j - \mu} / \sqrt{\sigma^2 + \smallconst} + \beta_j\\ [\[4\]](#ref-ba2016layer). It does the same in training and in prediction, since it never looks at other examples.

> **NOTE:**
>
> **Example 2 (`nn.LayerNorm` on one image)**  
>
> Show code
>
> ``` python
> layer_norm = nn.LayerNorm(64)
> row_normed = layer_norm(pre[:1]).detach()
> print(row_normed.mean().item(), row_normed.std(correction=0).item())
> ```
>
>     1.0244548320770264e-08 0.9999433159828186
>
> `nn.LayerNorm(64)` applies [layer normalization](#def-layer-norm) to each row of its input [\[5\]](#ref-pytorch_layernorm). For the first training image alone, its 64 outputs have mean 1.0e-08 and standard deviation 1.000.

## 4.2 Layer normalization in the deep network

> **NOTE:**
>
> **Exercise 4 (Layer normalization in the 20-block network)** Does `make_deep(make_norm=nn.LayerNorm)` learn in the 5 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) of `train()`?

> **NOTE:**
>
> *Solution 4*.
>
> Show code
>
> ``` python
> layer_grad = first_layer_grad(make_deep(make_norm=nn.LayerNorm))
> layer_loss, layer_acc = train(make_deep(make_norm=nn.LayerNorm))
> print(layer_grad, layer_loss, layer_acc)
> ```
>
>     0.3943977653980255 1.9132837057113647 0.15745393931865692
>
> Yes. With [layer normalization](#def-layer-norm), the training loss after 5 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) is 1.913, and the test [accuracy](https://morrison-lab.github.io/lds/chapters/metrics.html#def-accuracy) is 15.7%, against 1.251 and 77.2% with [batch normalization](#def-batch-norm). The [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) at the first layer has size 3.9e-01. Here layer normalization learns less than batch normalization in the same 5 epochs. Because layer normalization never uses other examples, it also works for a minibatch of one and for sequences of different lengths [\[4\]](#ref-ba2016layer).

# 5 Residual connections

## 5.1 A residual block

> **NOTE:**
>
> *Remark 1* (Residual connections in `Block`). `Block(width, residual=True)` returns \\\vhidval + \operatorname{ReLU}\paren{\matr{W} \vhidval + \vecf{b}}\\ for its input \\\vhidval\\, a [residual connection](https://morrison-lab.github.io/lds/chapters/backpropagation.html#def-residual-connection) around one [hidden layer](../chapters/feedforward-networks.llms.md#def-hidden-layer). This block is simpler than those of He et al. [\[6\]](#ref-he2016resnet), which go around two or three convolutional layers and apply the [ReLU](../chapters/feedforward-networks.llms.md#def-relu) after the sum. The input and output have the same [width](https://morrison-lab.github.io/lds/chapters/neural-nets.html#def-width), 64, so the sum is defined.

## 5.2 Gradients with residual connections

> **NOTE:**
>
> **Exercise 5 (The gradient through residual connections)** How large is the [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) at the first layer of `make_deep(residual=True)`, compared with `make_deep()`?

> **NOTE:**
>
> *Solution 5*.
>
> Show code
>
> ``` python
> residual_grad = first_layer_grad(make_deep(residual=True))
> print(residual_grad, plain_grad)
> ```
>
>     25.320859909057617 1.927740633789199e-09
>
> The [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) has size 2.5e+01, against 1.9e-09 without [residual connections](https://morrison-lab.github.io/lds/chapters/backpropagation.html#def-residual-connection). A block computes \\\vhidval + f(\vhidval)\\, with \\f(\vhidval) = \operatorname{ReLU}\paren{\matr{W} \vhidval + \vecf{b}}\\ from [Remark 1](#rem-residual-block), so its derivative with respect to \\\vhidval\\ is the identity matrix plus the derivative of \\f\\. Going back through a block, the gradient keeps a copy of itself through the identity. It keeps that copy however small the derivative of \\f\\ is. Without the [residual connection](https://morrison-lab.github.io/lds/chapters/backpropagation.html#def-residual-connection), the gradient passes only through the derivative of \\f\\, and shrinks when that derivative is small. Across 20 blocks these derivatives multiply. So when each factor is below 1 in size, the gradient shrinks geometrically.

## 5.3 Four networks

> **NOTE:**
>
> **Exercise 6 (Four 20-block networks)** Which of `make_deep()`, `make_deep(make_norm=nn.BatchNorm1d)`, `make_deep(residual=True)` and `make_deep(make_norm=nn.BatchNorm1d, residual=True)` reaches the lowest training loss in the 5 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) of `train()`?

> **NOTE:**
>
> *Solution 6*.
>
> Show code
>
> ``` python
> settings = {
>     "no normalization or residual connections": {},
>     "batch normalization": {"make_norm": nn.BatchNorm1d},
>     "residual connections": {"residual": True},
>     "batch normalization and residual connections": {"make_norm": nn.BatchNorm1d, "residual": True},
> }
> four_rows = []
> for name, options in settings.items():
>     loss, acc = train(make_deep(**options))
>     four_rows.append({"network": name, "train_loss": loss, "test_accuracy": acc})
> four_table = pd.DataFrame(four_rows)
> best_network = four_table.loc[four_table["train_loss"].idxmin()]
> residual_only_loss = four_table.set_index("network").loc["residual connections", "train_loss"]
> print(four_table.round(3))
> ```
>
>                                             network  train_loss  test_accuracy
>     0      no normalization or residual connections       2.303          0.107
>     1                           batch normalization       1.251          0.772
>     2                          residual connections       1.435          0.558
>     3  batch normalization and residual connections       0.040          0.953
>
> The network with batch normalization and residual connections reaches the lowest training loss, 0.040, and classifies 95.3% of the test images correctly. The residual network without [batch normalization](#def-batch-norm) reaches 1.435. The residual networks of He et al. [\[6\]](#ref-he2016resnet) also put batch normalization inside each block.

## 5.4 Further reading

> **NOTE:**
>
> *Deep Learning*’s [section 8.7.1](https://www.deeplearningbook.org/contents/optimization.html) covers batch normalization [\[2\]](#ref-goodfellow2016deep). Ioffe and Szegedy [\[1\]](#ref-ioffe2015batch) introduced batch normalization, Ba et al. [\[4\]](#ref-ba2016layer) [layer normalization](#def-layer-norm), and He et al. [\[6\]](#ref-he2016resnet) residual networks.

# 6 References

\[1\]

S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in *Proceedings of the 32nd international conference on machine learning*, in Proceedings of machine learning research, vol. 37. 2015, pp. 448–456. Available: <https://proceedings.mlr.press/v37/ioffe15.html>

\[2\]

I. Goodfellow, Y. Bengio, and A. Courville, *Deep learning*. MIT Press, 2016. Available: <https://www.deeplearningbook.org/>

\[3\]

PyTorch Contributors, “Torch.nn.BatchNorm1d.” PyTorch documentation, 2026. Available: <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>

\[4\]

J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization.” 2016. Available: <https://arxiv.org/abs/1607.06450>

\[5\]

PyTorch Contributors, “Torch.nn.LayerNorm.” PyTorch documentation, 2026. Available: <https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>

\[6\]

K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2016, pp. 770–778. doi: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).

Back to top
