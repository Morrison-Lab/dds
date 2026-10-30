# Recurrent networks

Code

- [Show All Code](javascript:void(0))
- [Hide All Code](javascript:void(0))

Published

Last modified: 2026-10-30

Show code

``` downlit
# The Python chunks on this page import these; reticulate provisions them.
# Attaching reticulate gives the inline R its `py` object.
library(reticulate)
source("../tools/use-torch.R")
use_torch(c("numpy", "pandas", "matplotlib"))
```

Brian Hutchinson’s CSCI 481/581 sets the order of topics in these notes, which follow it with his permission; this page covers the part of his module on recurrent networks that deals with training them, and with turning symbols into vectors. The exercises are new.

# 1 Welcome

## 1.1 Plan for this lecture

The DATA 471 notes defined [recurrent layers](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-recurrent-layer), found that their [gradients](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) can [vanish or explode](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#thm-jacobian-bound) over long sequences, and defined the [LSTM](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm) layer that keeps a memory with [gates](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-gate). This lecture trains these layers in PyTorch:

1.  the [shape](../chapters/pytorch-basics.llms.md#def-shape)s that `nn.RNN` takes and returns;
2.  reading a digit one row at a time, then one pixel at a time;
3.  capping the size of the [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient);
4.  [LSTM](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm) layers and a lighter gated layer;
5.  turning symbols such as letters into vectors that a network can learn.

## 1.2 The digits as sequences

Show code

``` python
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

_ = torch.manual_seed(481)
digits = pd.read_csv("data/optdigits-test.csv", header=None)
pixels = torch.tensor(digits.iloc[:, :64].to_numpy(), dtype=torch.float32) / 16
y = torch.tensor(digits[64].to_numpy())
n_train = 1200
order = torch.randperm(len(pixels))
x_train, y_train = pixels[order[:n_train]], y[order[:n_train]]
x_test, y_test = pixels[order[n_train:]], y[order[n_train:]]
loss_fn = nn.CrossEntropyLoss()
LR = 0.01
BATCH_SIZE = 32


class Reader(nn.Module):
    # Reads each image as a sequence of `step_size` pixels at a time,
    # with the recurrent layer `layer`, and classifies it from the last state.

    def __init__(self, layer, step_size, hidden_size=32):
        super().__init__()
        self.step_size = step_size
        self.recurrent = layer(step_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 10)

    def forward(self, images):
        sequence = images.reshape(len(images), -1, self.step_size)
        states, _ = self.recurrent(sequence)
        return self.out(states[:, -1])


def make_reader(layer, step_size, seed=1):
    torch.manual_seed(seed)
    return Reader(layer, step_size)


def train(net, n_epochs=10, lr=LR, batch_size=BATCH_SIZE, seed=0, max_norm=math.inf):
    # Adam on the training digits; returns the gradient norm before each step.
    # A finite `max_norm` rescales any larger gradient to that norm.
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    norms = []
    for epoch in range(n_epochs):
        net.train()
        shuffled = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, batch_size):
            batch = shuffled[start:start + batch_size]
            optimizer.zero_grad()
            loss_fn(net(x_train[batch]), y_train[batch]).backward()
            norm = nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            norms.append(norm.item())
            optimizer.step()
    return norms


def accuracy(net):
    # Fraction of the test digits that `net` classifies correctly.
    net.eval()
    with torch.no_grad():
        return (net(x_test).argmax(dim=1) == y_test).float().mean().item()


print(make_reader(nn.RNN, 8))
```

    Reader(
      (recurrent): RNN(8, 32, batch_first=True)
      (out): Linear(in_features=32, out_features=10, bias=True)
    )

The exercises use the UCI optical digits [\[1\]](#ref-alpaydin1998optdigits), split as in the [feedforward networks chapter](../chapters/feedforward-networks.llms.md), with 1,200 training images. `Reader` reads each image as a sequence, `step_size` pixels at a time, and classifies it with a linear layer on the last [hidden state](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-hidden-state). With `step_size=8`, each step reads one row of the image. `train()` runs [Adam](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-adam) with [learning rate](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-learning-rate) 0.01 and batch size 32.

# 2 Recurrent layers in PyTorch

## 2.1 What a recurrent layer returns

> **NOTE:**
>
> **Exercise 1 (What `nn.RNN` returns)** `nn.RNN(8, 32, batch_first=True)` is a [recurrent layer](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-recurrent-layer) with inputs of length 8 and a [hidden state](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-hidden-state) of length 32. It returns two [tensors](../chapters/pytorch-basics.llms.md#def-tensor). What are their [shapes](../chapters/pytorch-basics.llms.md#def-shape) for the first 16 training digits, each read as a sequence of 8 rows?

> **NOTE:**
>
> *Solution 1*.
>
> Show code
>
> ``` python
> recurrent = nn.RNN(8, 32, batch_first=True)
> rows = x_train[:16].reshape(16, 8, 8)
> states, last_state = recurrent(rows)
> same_last = torch.equal(states[:, -1], last_state[0])
> print(rows.shape, states.shape, last_state.shape, same_last)
> ```
>
>     torch.Size([16, 8, 8]) torch.Size([16, 8, 32]) torch.Size([1, 16, 32]) True
>
> With `batch_first=True`, the input has [shape](../chapters/pytorch-basics.llms.md#def-shape) \\16 \times 8 \times 8\\: 16 sequences of 8 steps, each step a row of 8 pixels. The first output holds the [hidden state](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-hidden-state) after every step, with shape \\16 \times 8 \times 32\\. The second holds the hidden state after the last step only, with shape \\1 \times 16 \times 32\\. Its first dimension counts stacked layers times directions, here 1 layer read in 1 direction [\[2\]](#ref-pytorch_rnn). Its entries equal the last step of the first output. Without `batch_first=True`, PyTorch expects the steps first and the sequences second.

## 2.2 Row by row

> **NOTE:**
>
> **Exercise 2 (Reading a digit row by row)** Train `make_reader(nn.RNN, 8)` for 10 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) with `train()`. What fraction of the test digits does it classify correctly?

> **NOTE:**
>
> *Solution 2*.
>
> Show code
>
> ``` python
> row_rnn = make_reader(nn.RNN, 8)
> _ = train(row_rnn)
> row_acc = accuracy(row_rnn)
> print(row_acc)
> ```
>
>     0.9430485963821411
>
> It classifies 94.3% of the 597 test digits correctly, after reading each digit in 8 steps.

## 2.3 Pixel by pixel

> **NOTE:**
>
> **Exercise 3 (Reading a digit pixel by pixel)** `make_reader(nn.RNN, 1)` reads each digit one pixel at a time, in 64 steps. Train it for 20 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch) with `train()`. What fraction of the test digits does it classify correctly?

> **NOTE:**
>
> *Solution 3*.
>
> Show code
>
> ``` python
> pixel_rnn = make_reader(nn.RNN, 1)
> rnn_norms = train(pixel_rnn, n_epochs=20)
> pixel_acc = accuracy(pixel_rnn)
> print(pixel_acc)
> ```
>
>     0.0837520956993103
>
> Show code
>
> ``` python
> # The prose below says this reader does worse; stop the render if it does not.
> assert pixel_acc < row_acc
> ```
>
> It classifies only 8.4% of the test digits correctly, against 94.3% for the row reader of [Solution 2](#sol-rows). Both see every pixel, and the pixel reader trains for twice as many [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch). What differs is the length of each sequence: 64 steps rather than 8. The DATA 471 notes [bound how fast](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#thm-jacobian-bound) a [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) can shrink or grow over that many steps.

# 3 Large gradients

## 3.1 The size of the gradient

> **NOTE:**
>
> **Exercise 4 (The size of the gradient during training)** `train()` returned the [norm](https://morrison-lab.github.io/mds/linear-algebra-inner-products.html#def-norm) of the [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) before each step of training the pixel reader in [Solution 3](#sol-pixels), as `rnn_norms`. How large does the gradient get?

> **NOTE:**
>
> *Solution 4*.
>
> > **NOTE:**
> >
> > Show code
> >
> > ``` python
> > _ = plt.figure(figsize=(7, 3))
> > _ = plt.semilogy(rnn_norms, linewidth=0.7)
> > _ = plt.xlabel("training step")
> > _ = plt.ylabel("gradient norm")
> > plt.tight_layout()
> > plt.show()
> > ```
> >
> > Show code
> >
> > ``` python
> > median_norm = np.median(rnn_norms)
> > max_norm_seen = max(rnn_norms)
> > share_above_10 = np.mean(np.array(rnn_norms) > 10)
> > last_steps = 100
> > late_median = np.median(rnn_norms[-last_steps:])
> > ```
> >
> > [![Gradient norm against training step, on a log scale, for the pixel-by-pixel recurrent network. The norms mostly stay between 1 and 10 for the first 500 or so steps, then spike to between 100 and over 1,000 several times, and settle near 1 for the last steps.](rnns_files/figure-html/unnamed-chunk-6-1.png)](rnns_files/figure-html/unnamed-chunk-6-1.png "Figure 1: The norm of the gradient before each training step of the pixel reader.")
> >
> > Figure 1: The [norm](https://morrison-lab.github.io/mds/linear-algebra-inner-products.html#def-norm) of the [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) before each [training step](../chapters/pytorch-basics.llms.md#def-training-step) of the pixel reader.
>
> The median [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) [norm](https://morrison-lab.github.io/mds/linear-algebra-inner-products.html#def-norm) is 3.3, but the largest is 43, 13 times the median. 4% of the steps have a norm above 10. Over the last 100 steps the median norm is 5.2, and the reader ends at 8.4% test [accuracy](https://morrison-lab.github.io/lds/chapters/metrics.html#def-accuracy), near the 10% of guessing. So after the spikes the network sat in a state that training did not move it out of.

## 3.2 Clipping

> **NOTE:**
>
> **Exercise 5 (Clipping the gradient)** `train(net, max_norm=1)` applies [gradient clipping](https://morrison-lab.github.io/lds/chapters/backpropagation.html#def-gradient-clipping): before each step, `nn.utils.clip_grad_norm_` rescales any [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) with [norm](https://morrison-lab.github.io/mds/linear-algebra-inner-products.html#def-norm) above 1 to norm 1 [\[3\]](#ref-pytorch_clip_grad_norm). Trained this way for 20 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch), does the pixel reader classify more test digits correctly than in [Solution 3](#sol-pixels)?

> **NOTE:**
>
> *Solution 5*.
>
> Show code
>
> ``` python
> clipped_rnn = make_reader(nn.RNN, 1)
> _ = train(clipped_rnn, n_epochs=20, max_norm=1)
> clipped_acc = accuracy(clipped_rnn)
> print(clipped_acc)
> ```
>
>     0.6164153814315796
>
> Show code
>
> ``` python
> # The prose below says the reader is still below the row reader.
> assert clipped_acc < row_acc
> ```
>
> With [gradient clipping](https://morrison-lab.github.io/lds/chapters/backpropagation.html#def-gradient-clipping), the pixel reader classifies 61.6% of the test digits correctly, against 8.4% without it. So clipping helps. Clipping caps the size of each [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient), so no gradient [Adam](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-adam) sees can be as large as the spikes of [Figure 1](#fig-grad-norms). The reader is still 32.7 percentage points below the row reader of [Solution 2](#sol-rows).

# 4 Gated layers

## 4.1 An LSTM

> **NOTE:**
>
> **Exercise 6 (An LSTM reading pixels)** `make_reader(nn.LSTM, 1)` reads the pixels with an [LSTM](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm) layer instead, with no clipping. Trained for 20 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch), does it classify more test digits correctly than the pixel reader of [Solution 5](#sol-clipped)?

> **NOTE:**
>
> *Solution 6*.
>
> Show code
>
> ``` python
> pixel_lstm = make_reader(nn.LSTM, 1)
> lstm_norms = train(pixel_lstm, n_epochs=20)
> lstm_acc = accuracy(pixel_lstm)
> print(lstm_acc, max(lstm_norms))
> ```
>
>     0.8207705020904541 12.21899127960205
>
> Show code
>
> ``` python
> # The prose below says the LSTM does better; stop the render if it does not.
> assert lstm_acc > max(clipped_acc, pixel_acc)
> ```
>
> Yes. The [LSTM](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm) classifies 82.1% of the test digits correctly, against 61.6% for the clipped pixel reader. Its [cell state](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm) \\\vecf{c}\_t\\ changes only by the [gates’](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-gate) products and sums, with no squashing activation between one step and the next. So a [gradient](https://morrison-lab.github.io/lds/chapters/linear-regression.html#def-gradient) can pass back along the cell state through many steps when the [forget gate](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm) stays near 1 [\[4, Sec. 10.10.1\]](#ref-goodfellow2016deep). The LSTM’s largest gradient [norm](https://morrison-lab.github.io/mds/linear-algebra-inner-products.html#def-norm) during training was 12.2, against 43 for the pixel reader of [Solution 3](#sol-pixels).

## 4.2 Two gates instead of three

> **NOTE:**
>
> **Definition 1 (Gated recurrent unit (GRU))** A **gated recurrent unit** (GRU) layer keeps only a [hidden state](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-hidden-state) \\\vhidval_t \in \reals^{H}\\, with no separate [cell state](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm), and two [gates](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-gate) [\[5\]](#ref-cho2014gru). In the form PyTorch computes [\[6\]](#ref-pytorch_gru),
>
> \\\begin{aligned} \vecf{z}\_t &= \sigmoidf{\matr{W}\_z \vx_t + \matr{U}\_z \vhidval\_{t-1} + \vecf{b}\_z}, &&\text{(update gate)} \\ \vecf{r}\_t &= \sigmoidf{\matr{W}\_r \vx_t + \matr{U}\_r \vhidval\_{t-1} + \vecf{b}\_r}, &&\text{(reset gate)} \\ \vecf{u}\_t &= \tanhf{\matr{W}\_u \vx_t + \vecf{b}\_u + \vecf{r}\_t \odot \paren{\matr{U}\_u \vhidval\_{t-1} + \vecf{c}\_u}}, &&\text{(candidate)} \\ \vhidval_t &= \vecf{z}\_t \odot \vhidval\_{t-1} + \paren{1 - \vecf{z}\_t} \odot \vecf{u}\_t. \end{aligned}\\
>
> Here \\\vecf{c}\_u\\ is a second bias vector, and PyTorch also adds one to \\\matr{U}\_z \vhidval\_{t-1}\\ and to \\\matr{U}\_r \vhidval\_{t-1}\\. Cho et al. [\[5\]](#ref-cho2014gru) apply the reset gate before the matrix, writing \\\matr{U}\_u \paren{\vecf{r}\_t \odot \vhidval\_{t-1}}\\ in the candidate, and also put \\\vecf{z}\_t\\ on the old state. Chung et al. [\[7\]](#ref-chung2014empirical) and the [DATA 471 notes](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#rem-gru) put \\1 - \vecf{z}\_t\\ on the old state instead, which swaps the meaning of \\\vecf{z}\_t\\ near 0 and near 1.

> **NOTE:**
>
> **Example 1 (Parameters of `nn.GRU` and `nn.LSTM`)**  
>
> Show code
>
> ``` python
> def n_params(net):
>     return sum(p.numel() for p in net.parameters())
>
>
> gru_count = n_params(nn.GRU(1, 32))
> lstm_count = n_params(nn.LSTM(1, 32))
> print(gru_count, lstm_count)
> ```
>
>     3360 4480
>
> A [GRU](#def-gru) layer has three blocks of weights, for \\\vecf{z}\_t\\, \\\vecf{r}\_t\\ and \\\vecf{u}\_t\\, where an [LSTM](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm) layer has four. PyTorch gives each block two bias vectors, one added to \\\matr{W} \vx_t\\ and one to \\\matr{U} \vhidval\_{t-1}\\ [\[6\]](#ref-pytorch_gru). So `nn.GRU(1, 32)` has \\3 \times 32 \times (1 + 32 + 2) = 3360\\ [parameters](https://morrison-lab.github.io/lds/chapters/big-ideas.html#def-parameter), and `nn.LSTM(1, 32)` has \\4 \times 32 \times (1 + 32 + 2) = 4480\\.

## 4.3 Comparing the readers

> **NOTE:**
>
> **Exercise 7 (A GRU reading pixels)** Train `make_reader(nn.GRU, 1)` for 20 [epochs](https://morrison-lab.github.io/lds/chapters/gradient-descent.html#def-epoch). Does the [GRU](#def-gru) reader classify more test digits correctly than the [LSTM](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm) reader of [Solution 6](#sol-lstm-pixels)?

> **NOTE:**
>
> *Solution 7*.
>
> Show code
>
> ``` python
> pixel_gru = make_reader(nn.GRU, 1)
> _ = train(pixel_gru, n_epochs=20)
> gru_acc = accuracy(pixel_gru)
> print(gru_acc)
> ```
>
>     0.8442211151123047
>
> Show code
>
> ``` python
> gap_digits = round(abs(gru_acc - lstm_acc) * len(y_test))
> # The prose below says both gated readers beat both plain ones; stop if not.
> assert min(gru_acc, lstm_acc) > max(pixel_acc, clipped_acc)
> ```
>
> | Layer reading the pixels | [Gradient clipping](https://morrison-lab.github.io/lds/chapters/backpropagation.html#def-gradient-clipping) | Test [accuracy](https://morrison-lab.github.io/lds/chapters/metrics.html#def-accuracy) |
> |:---|:---|---:|
> | `nn.RNN` | None | 8.4% |
> | `nn.RNN` | To 1 | 61.6% |
> | `nn.LSTM` | None | 82.1% |
> | `nn.GRU` | None | 84.4% |
>
> The [GRU](#def-gru) reader classifies 84.4% of the test digits correctly, 2.3 percentage points above the [LSTM](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-lstm) reader, with 75% as many [parameters](https://morrison-lab.github.io/lds/chapters/big-ideas.html#def-parameter) in its [recurrent layer](https://morrison-lab.github.io/lds/chapters/cnn-rnn.html#def-recurrent-layer). So yes: the two readers differ by 14 of the 597 test digits. Both gated readers beat the plain recurrent readers. On other tasks, Chung et al. [\[7\]](#ref-chung2014empirical) found neither gated layer consistently better than the other.

# 5 Symbols as vectors

## 5.1 A one-hot vector times a matrix

> **NOTE:**
>
> **Exercise 8 (Multiplying a one-hot vector by a matrix)** A network that reads text needs each character as a vector. Number the 26 lowercase letters and the space from 0 to 26, so `"c"` is 2. Count vector entries and matrix rows from 0 as well. Let \\\vecf{e}\_2 \in \\0, 1\\^{27}\\ be the [one-hot encoding](https://morrison-lab.github.io/lds/chapters/big-ideas.html#def-one-hot-encoding) of `"c"`, and let \\\matr{E}\\ be the \\27 \times 4\\ matrix printed below. What is \\\tp{\vecf{e}\_2} \matr{E}\\?
>
> Show code
>
> ``` python
> alphabet = "abcdefghijklmnopqrstuvwxyz "
> _ = torch.manual_seed(2)
> E = torch.randn(len(alphabet), 4).round(decimals=2)
> print(E[:4])
> ```
>
>     tensor([[-1.0400,  0.9200, -1.3000, -1.1100],
>             [-1.2200,  1.1700, -1.0600, -0.1200],
>             [-0.9100,  0.3500, -0.5700, -0.2400],
>             [ 1.0100, -0.7500, -0.2200, -0.4300]])

> **NOTE:**
>
> *Solution 8*.
>
> Show code
>
> ``` python
> one_hot_c = nn.functional.one_hot(torch.tensor(alphabet.index("c")), len(alphabet)).float()
> product = one_hot_c @ E
> print(product, E[2], torch.equal(product, E[2]))
> ```
>
>     tensor([-0.9100,  0.3500, -0.5700, -0.2400]) tensor([-0.9100,  0.3500, -0.5700, -0.2400]) True
>
> Entry \\j\\ of \\\tp{\vecf{e}\_2} \matr{E}\\ is \\\sum_i (\vecf{e}\_2)\_i \matr{E}\_{ij} = \matr{E}\_{2j}\\, since \\(\vecf{e}\_2)\_i\\ is 1 for \\i = 2\\ and 0 for every other \\i\\. So the product is row 2 of \\\matr{E}\\, the third row printed above. Multiplying by a [one-hot encoding](https://morrison-lab.github.io/lds/chapters/big-ideas.html#def-one-hot-encoding) only looks up a row. A network can skip multiplying by the 26 zeros and read the row directly.

> **NOTE:**
>
> **Definition 2 (Embedding layer)** An **embedding layer** maps each of \\K\\ symbols, such as characters or words, to a vector in \\\reals^d\\. Symbol \\k\\ maps to row \\k\\ of a \\K \times d\\ matrix \\\matr{E}\\ of [parameters](https://morrison-lab.github.io/lds/chapters/big-ideas.html#def-parameter), learned with the rest of the network [\[4, Sec. 12.4.2\]](#ref-goodfellow2016deep). Row \\k\\ is the vector that represents symbol \\k\\. The layer equals a linear layer with no bias applied to the [one-hot encoding](https://morrison-lab.github.io/lds/chapters/big-ideas.html#def-one-hot-encoding) of the symbol, computed by looking up the row.

> **NOTE:**
>
> **Example 2 (`nn.Embedding`)**  
>
> Show code
>
> ``` python
> embed = nn.Embedding(len(alphabet), 4)
> word = torch.tensor([alphabet.index(ch) for ch in "deep"])
> vectors = embed(word)
> same_rows = torch.equal(vectors, embed.weight[word])
> print(word, vectors.shape, same_rows)
> ```
>
>     tensor([ 3,  4,  4, 15]) torch.Size([4, 4]) True
>
> `nn.Embedding(27, 4)` is an [embedding layer](#def-embedding) for the 27 characters, with a \\27 \times 4\\ matrix `weight` [\[8\]](#ref-pytorch_embedding). It takes a [tensor](../chapters/pytorch-basics.llms.md#def-tensor) of whole-number indices, here the indices of the letters of `"deep"`, and returns one row of `weight` for each. For the 4 letters the result has [shape](../chapters/pytorch-basics.llms.md#def-shape) \\4 \times 4\\. The two `"e"`s get the same vector. The returned rows equal the rows of `embed.weight` at those indices.

## 5.2 Further reading

> **NOTE:**
>
> *Deep Learning*’s [chapter 10](https://www.deeplearningbook.org/contents/rnn.html) covers recurrent networks, LSTMs and gradient clipping [\[4\]](#ref-goodfellow2016deep). Cho et al. [\[5\]](#ref-cho2014gru) introduced the GRU, and Chung et al. [\[7\]](#ref-chung2014empirical) compared it with the LSTM.

# 6 References

\[1\]

E. Alpaydin and C. Kaynak, “Optical recognition of handwritten digits.” UCI Machine Learning Repository, 1998. doi: [10.24432/C50P49](https://doi.org/10.24432/C50P49).

\[2\]

PyTorch Contributors, “Torch.nn.RNN.” PyTorch documentation, 2026. Available: <https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html>

\[3\]

PyTorch Contributors, “Torch.nn.utils.clip_grad_norm\_.” PyTorch documentation, 2026. Available: <https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>

\[4\]

I. Goodfellow, Y. Bengio, and A. Courville, *Deep learning*. MIT Press, 2016. Available: <https://www.deeplearningbook.org/>

\[5\]

K. Cho *et al.*, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in *Proceedings of the 2014 conference on empirical methods in natural language processing*, 2014, pp. 1724–1734. doi: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179).

\[6\]

PyTorch Contributors, “Torch.nn.GRU.” PyTorch documentation, 2026. Available: <https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html>

\[7\]

J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, “Empirical evaluation of gated recurrent neural networks on sequence modeling.” 2014. Available: <https://arxiv.org/abs/1412.3555>

\[8\]

PyTorch Contributors, “Torch.nn.Embedding.” PyTorch documentation, 2026. Available: <https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html>

Back to top
