---
title: Logistic regression
author: Michel Jean Joseph Donnet
date: \today
---

\newpage
# Introduction

There are many method to classify elements thanks to features given.

We are looking at the regression method, and study the performances of this method

# Methodology

- $X$: a set of features ($n\times d$)
- $y$: a set of labels for the given features ($n \times h$)
- $C$: the set of all the classes

## Binary case

In this case, $h = 1$.

We consider that we have only two classes $c_{1}, c_{2} \in C$.

We want to define what is the probability of $p(y | X)$.

To do that, we want to define what is the probability for just one instance:
$$p(y_i = c_1 | x_i)$$

As we have 2 classes, we can consider that this probability follow a Bernoulli law.
So $y_i = 1$ if $y_i = c_1$ and $y_i = 0$ if $y_i \neq c_1$.

The probability to have $1$ is given by a function that depends on $x_i$.
So we have: 

$$p(y_i = 1 | x_i) = g(x_i)$$ {#eq:g}

with $g(x_i)$ which give us a scalar from the vector of features $x_i$.

So it means that we have:

$$
p(y_i | x_i) = \left\{
\begin{aligned}
p(y_i = 1 | x_i) & \ \text{if} \ y_i = c_1 \\
1 - p(y_i = 1 | x_i) &  \ \text{if} \ y_i \neq c_1
\end{aligned}
\right.
$$ {#eq:prob1}

We can write the equation @eq:prob1 like this:

$$
p(y_i | x_i) = p(y_i = 1|x_i)^{y_i}(1 - p(y_i = 1|x_i))^{1 - y_i}
$$ {#eq:prob2}

But we want to know what is the function $g(x_i)$.

According to Bayes theorem, we have:

$$
\begin{aligned}
p(y_i = 1 | x_i) &= \frac{p(x_i| y_i = 1)p(y_i = 1)}{p(x_i)} \\
&= \frac{p(x_i| y_i = 1)p(y_i = 1)}{p(x_i|y_i = 1)p(y_i = 1) + p(x_i|y_i = 0)p(y_i = 0)} \\
&= \frac{\frac{p(x_i| y_i = 1)p(y_i = 1)}{p(x_i|y_i = 0)p(y_i = 0)}}{\frac{p(x_i|y_i = 1)p(y_i = 1)}{p(x_i|y_i = 0)p(y_i = 0)} + 1} \\
&= \frac{\frac{p(x_i| y_i = 1)p(y_i = 1)p(x_i)}{p(x_i|y_i = 0)p(y_i = 0)p(x_i)}}{\frac{p(x_i|y_i = 1)p(y_i = 1)p(x_i)}{p(x_i|y_i = 0)p(y_i = 0)p(x_i)} + 1} \\
&= \frac{\frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)}}{\frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} + 1} \\
\end{aligned}
$$ {#eq:sigma1}

In probability, if $q$ is the probability of an event, so the probability that this event is happening is given by $\frac{q}{1 - q}$.
This ratio is called the odds.

In our case, as $p(y_i = 1|x_i)$ is the probability that the event is happening, and $p(y_i = 0|x_i)$ is the probability that the event is not happening, we have the odds defined like this:

$$Odds = \frac{p(y_i = 1|x_i)}{1 - p(y_i = 1|x_i)}$$ {#eq:odds}

In logistic regression, we want that the logarithm of the odds is linear.
So according to equation @eq:odds, we want that:
$$
\begin{aligned}
log\left( \frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} \right) &= w_0 + w_1 \cdot x_{i1} + \cdots + w_d \cdot x_{id} \\
\Leftrightarrow \ \ \      log\left( \frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} \right) &= w_0 + x_iw \\
\Leftrightarrow \ \ \      log\left( \frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} \right) &= x_iw
& \ \text{if}\ x = \begin{bmatrix} 1 & x_{i1} & \cdots & x_{id} \end{bmatrix} \\
&& \ \text{and}\ w^T = \begin{bmatrix} w_0 & \cdots & w_d \end{bmatrix} \\
\Leftrightarrow \ \ \      \frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} &= e^{x_iw}
\end{aligned}
$$ {#eq:e}

So according to the result of the equation @eq:e, we can rewrite the equation @eq:sigma1 like this:

$$
\begin{aligned}
p(y_i = 1 | x_i) 
&= \frac{\frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)}}{\frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} + 1} \\
&= \frac{e^{x_iw}}{e^{x_iw} + 1} \\
&= \frac{1}{1 + e^{-x_iw}} \\
&= \sigma(x_iw) &\ \text{with}\ \sigma(z) = \frac{1}{1 + e^{-z}}
\end{aligned}
$$ {#eq:sigma}

$\sigma$ in @eq:sigma is called the sigmoïd function. (As $e^{-x}$ is always negative or equal to zero, $\sigma$ is always between $0$ and $1$)

So we can write the equation @eq:prob2 like this:
$$
\begin{aligned}
p(y_i | x_i) &= p(y_i = 1|x_i)^{y_i}(1 - p(y_i = 1|x_i))^{1 - y_i} \\
&= \sigma(x_iw)^{y_i}(1 - \sigma(x_iw))^{1 - y_i}
\end{aligned}
$$ {#eq:prob}

### Cost function

Now, we want to define a cost function to optimize the parameter $w$.

Note that $p(y_i | x_i)$ is called the likelihood.

In fact, we want to maximise the probability of $p(y_i = 1 |x_i)$ when $y_i = 1$ and maximise the probability of $(1 - p(y_i = 1|x_i))$ when $y_i \neq 1$.

So we want to maximize the likelihood, which is given by equation @eq:prob.
And we want to maximize the probability for all instances and not only for instance $i$.

So we want to maximize:
$$
\begin{aligned}
\text{max arg}_w \frac{1}{n} \sum_i^n p(y_i | x_i)
\end{aligned}
$$ {#eq:total}

The $\frac{1}{n}$ allow us to normalize our datas, to have an answer in the space definition of $p(y_i | x_i)$.

We want to use the gradient descent. So we want to have something to minimise.
As we have to maximize the equation @eq:prob, we only have to invert this equation to have a minimisation problem.

We can easily see that the likelihood is always non zeros:
we know that the exponential function is strictly positive, and according to equation @eq:sigma, we have:

$$
\begin{aligned}
&0 &< e^{x_iw} &< 1 + e^{x_iw}\\
\Leftrightarrow \ \ \ &0 &< \frac{e^{x_iw}}{1 + e^{x_iw}} &< 1 \\
\Leftrightarrow \ \ \ &0 &< sigma(x_iw) &< 1 \\
\Leftrightarrow \ \ \ &0 &< p(y_i | x_i) &< 1
\end{aligned}
$$

because else,
it means that all the elements are in one class, so we don't have two classes and we don't need to make some classification.

So we have according to equation @eq:total:

$$
\begin{aligned}
\text{max arg}_w \frac{1}{n} \sum_i^n p(y_i | x_i)
&= \text{min arg}_w \frac{1}{n} \sum_i^n \frac{1}{p(y_i | x_i)}\\
&= \text{min arg}_w \frac{1}{n} \sum_i^n\frac{1}{\sigma(x_iw)^{y_i}(1 - \sigma(x_iw))^{1 - y_i}} \\
&= \text{min arg}_w \frac{1}{n} \ log\left(\sum_i^n\frac{1}{\sigma(x_iw)^{y_i}(1 - \sigma(x_iw))^{1 - y_i}}\right) \\
&= \text{min arg}_w \frac{1}{n} \sum_i^n\left(- log(\sigma(x_iw)^{y_i}(1 - \sigma(x_iw))^{1 - y_i}) \right) \\
&= \text{min arg}_w \frac{1}{n} \sum_i^n\left( -y_ilog(\sigma(x_iw)) - (1 - y_i)log(1 - \sigma(x_iw)) \right)
\end{aligned}
$$ {#eq:cost}

In the equation @eq:cost, we can apply log to the equation because the log function is always positive and increasing, and since we have said that $p(y_i | x_i)$ is always positive.

So we have inverted the likelihood, and then we have apply log function to result.
As the log of the invert is equal to negative log (example: $log\left(\frac{1}{x}\right) = -log(x)$), we call the cost function the negative log likelihood function.

As $y$ has a size of $n \times h$, $X$ a size of $n \times d$ and $w$ a size of $d\times h$, we can write the cost function with matrix product:

$$
\begin{aligned}
\text{min arg}_w \frac{1}{n} \sum_i^n\left( -log(\sigma(x_iw))y_i - log(1 - \sigma(x_iw))(1 - y_i) \right) \\
= \text{min arg}_w \frac{1}{n} \left( -y^Tlog(\sigma(Xw)) - (1 - y^T)log(1 - \sigma(Xw)) \right) \\
&= NLL(X, y)
\end{aligned}
$$ {#eq:prod}

### Gradient

Now we have to compute the derivate of the cost function @eq:prod.

First, we want to find the derivate of the sigmoid function.

We have:

$$
\begin{aligned}
\frac{d\sigma(z)}{dz}
&= ((1 + e^{-z})^{-1}) \\
&= -1 \cdot \left(- e ^{-z}\right) \cdot (1 + e^{-z})^{-2} \\
&=\frac{e^{-z}}{(1 + e^{-z})^2} \\
&=\frac{1}{1 + e^{-z}}\cdot \frac{e^{-z}}{1 + e^{-z}} \\
&=\sigma (z) \frac{e^{-z}}{1 + e^{-z}} \\
&=\sigma (z) \frac{1 + e^{-z} - 1}{1 + e^{-z}} \\
&=\sigma (z) (\frac{1 + e^{-z}}{1 + e^{-z}} - \frac{1}{1 + e^{-z}}) \\
&=\sigma (z) (1 - \frac{1}{1 + e^{-z}}) \\
&=\sigma (z) (1 - \sigma (z)) \\
\end{aligned}
$$ {#eq:gradsig}

So according to the chain rule, we have:
$$
\begin{aligned}
\frac{\partial}{\partial w_k} NLL(X, y)
&= \frac{1}{n} \left( \frac{\partial}{\partial w_k} \left( -y^Tlog(\sigma(Xw)) - (1 - y^T)log(1 - \sigma(Xw)) \right)\right)\\
&= \frac{1}{n} \left( -y^T \frac{\partial}{\partial w_k}log(\sigma(Xw)) - (1 - y^T)\frac{\partial}{\partial w_k}log(1 - \sigma(Xw))\right) \\
&= \frac{1}{n} \left( -y^T \frac{\partial log(\sigma(Xw))}{\partial \sigma} \frac{\partial \sigma}{\partial Xw} \frac{\partial Xw}{\partial w_k} \right. \\
&\ \ \ \left.- (1 - y^T)\frac{\partial log(1 - \sigma(Xw))}{\partial (1 - \sigma(Xw))} \frac{\partial (1 - \sigma(Xw))}{\partial \sigma} \frac{\partial \sigma(Xw)}{\partial Xw}  \frac{\partial Xw}{\partial w_k}\right) \\
&= \frac{1}{n} \left( -y^T \frac{1}{\sigma(Xw)} \sigma(Xw)(1 - \sigma(Xw)) X_{:,k}\right. \\
&\ \ \ \left. -(1 - y^T)\frac{1}{1 - \sigma(Xw)}(-1)\sigma(Xw)(1 - \sigma(Xw)) X_{:, k} \right)\\
&= \frac{1}{n} \left( ((1 - y^T)\sigma(Xw) -y^T (1 - \sigma(Xw))) X_{:, k} \right)\\
&= \frac{1}{n} \left( (\sigma(Xw) -y^T\sigma(Xw) -y^T 1_{n, 1} + y^T\sigma(Xw)) X_{:, k}\right) \\
&= \frac{1}{n} \left( \left(\sigma(Xw)  - \sum_i^n y_i \right)X_{:, k}\right)\\
\end{aligned}
$$ {#eq:grad}

So the gradient of the cost function give us:

$$
\nabla NLL(X, y) = 
\begin{bmatrix}
\frac{\partial}{\partial w_0} NLL(X, y) \\
\vdots \\
\frac{\partial}{\partial w_d} NLL(X, y) \\
\end{bmatrix}
$$

Now, we have to make a gradient descent:

$$
w^{i+ 1} = w^{i} - \eta \nabla NLL(X, y)
$$ {#eq:graddesc}

with $\eta$ the learning rate.

## Multinomial case

Now, we have $h$ classes.

If $y_i = class_k$, we define $\tilde{y}_i$ a vector of size $1 \times h$ like this:

$$
\tilde{y}_{ij} = \left\{
\begin{aligned}
1 & \ \text{if} j = k \\
0 & \ \text{else}
\end{aligned}
\right.
$$ {#eq:tildey}

So we have for instance $\tilde{y}_i = \begin{bmatrix} 0 & 0 & 1 & 0 \end{bmatrix}$ if $h = 4$.

This class representation is called one-hot encoding.

Now, we note $y_i$ the result of the one-hot encoding of $y_i$: $y_i = \tilde{y}_i$. So $y$ is of size $n \times h$.

If we remember the equation @eq:sigma, we have (binary case):
$$
\begin{aligned}
p(y_i = 1 | x_i) 
&= \frac{e^{x_iw}}{e^{x_iw} + 1} \\
\end{aligned}
$$ {#eq:3}

We want to generalize the sigmoid function to more than two variables to have $p(c | x_i)$ the probability of class $c$ in one-hot representation according to the features $x_i$.

Now, we suppose that we have
$$
W = \begin{bmatrix} w_1 & \cdots & w_h \end{bmatrix}
$$ {#eq:w}
with each parameter $w_k$ defined specially for the class $k$. (Note that the size of $w_k$ is $d \times 1$, so the size of $W$ is $d \times h$)

In binary case, we can define $W = \begin{bmatrix} w_1 & w_2 \end{bmatrix}$, with $w_1$ a column vector of $0$.

So we can write the equation @eq:3 like this:

$$
\begin{aligned}
p(y_i = 1| x_i)
&= \frac{e^{x_iw_2}}{e^{x_iw_2} + e^0} \\
&= \frac{e^{x_iw_2}}{e^{x_iw_2} + e^{x_iw_1}} \\
&= \frac{e^{x_iw_2}}{\sum_{k = 1}^2 e^{x_iw_k}} \\
&= \frac{e^{x_iw_2}}{\sum_{k = 1}^h e^{x_iw_k}} \\
\end{aligned}
$$ {#eq:4}

So if we have $h$ classes, we can define a function like this:

$$
\begin{aligned}
p(y_{ik} = 1 | x_i)
&= \frac{e^{x_iw_k}}{\sum_{j = 1}^{h} e^{x_iw_j}} \\
&= softmax(W, x_i, k)
\end{aligned}
$$ {#eq:softmax}

And then, we can write this function in matricial form:

$$
\begin{aligned}
p(y_i | x_i)
&= \begin{bmatrix} p(y_{i1} = 1 | x_i) & \cdots & p(y_{ih} = 1 | x_i) \end{bmatrix} \\
&= \begin{bmatrix} \frac{e^{x_iw_1}}{\sum_{j = 1}^{h} e^{x_iw_j}} & \cdots & \frac{e^{x_iw_h}}{\sum_{j = 1}^{h} e^{x_iw_j}}\end{bmatrix} \\
&= \frac{e^{x_iW}}{\sum_{j = 1}^h e^{x_iw_j}} \\
&= softmax(x_iW)
\end{aligned}
$$ {#eq:softmaxM}


We can easily see that $softmax$ is between $0$ and $1$:
As the exponential function is always positive, we have that:

$$
\begin{aligned}
&0 &< e^{x_iw_k} &< e^{x_iw_k} + \sum_{j \neq k}^h e^{x_iw_j} \\
\Leftrightarrow \ \ \ &0 &< e^{x_iw_k} &< \sum_{j = 1}^h e^{x_iw_j} \\
\Leftrightarrow \ \ \ &0 &< e^{x_iw_k} &< \sum_{j = 1}^h e^{x_iw_j} \\
\Leftrightarrow \ \ \ &0 &< \frac{e^{x_iw_k}}{\sum_{j = 1}^h e^{x_iw_j}} &< 1 \\
\Leftrightarrow \ \ \ &0 &< softmax(W, x_i, k) &< 1 \\
\end{aligned}
$$ {#eq:softzeros}

and we can also see that $p(y_i | x_i)$ is a distribution of probability because we have:

$$
\begin{aligned}
\sum_k^h p(y_{ik} | x_i) &= \sum_k^h \frac{e^{x_iw_k}}{\sum_j^h e^{x_iw_j}} \\
&=  \frac{\sum_k^h e^{x_iw_k}}{\sum_j^h e^{x_iw_j}} \\
&=  \frac{\sum_j^h e^{x_iw_j}}{\sum_j^h e^{x_iw_j}} \\
&= 1
\end{aligned}
$$ {#eq:distrib}

### Cost function

Now we want to define a cost function.

We remember that we have $X$, the training features and $y$ the labels corresponding to the features.
We define $W$ and we want to optimize this parameter.

We have:

| Matrix | Size |
|:------:|:----:|
|$X$|$n \times d$|
|$y$|$n \times h$|
|$W$|$d \times h$|

For all the instances, according to the result of the equation @eq:softmaxM and to the set of features $X$, the predicted values are given by:
$$
p(c | X) = softmax(XW)
$$ {#eq:predsoft}

We can compute the cross-entropy between our predictions and the real values.
The cross-entropy give us the difference between two distributions of probability.
If we have two distributions $P$ and $Q$, the cross-entropy is given by $H(P, Q) = - \frac{1}{m} \sum_x^m P(x) log Q(x)$

As the labels $y$ are on one-hot form, we can considerate that $y$ give us a distribution of probability because, due to the definition of $y_i$ on one-hot encoding form, we have for each instance $i$ the result below: $\sum_k y_{ik} = 1$.
So for the labels $y$ on one-hot form and for our predictions $p(c | X)$, we have the cross entropy which give us:

$$
\begin{aligned}
H(y, p(c | X)) &= - \frac{1}{n} \sum_i^n y_i \odot log(p(c|x_i)) \\
&= - \frac{1}{n} \sum_i^n log(softmax(x_iW)) \odot y \\
\end{aligned}
$$ {#eq:h}

With $\odot$ the Hadamard product.

As the cross-entropy is the difference between two distributions, we want to minimise this cross-entropy because we want that our predictions are equal to the real values $y$.

So we want to find an optimal $W$ that minimise equation @eq:h.

As we want to use a gradient descent to find the optimal $W$, we are searching for the gradient of the cross-entropy @eq:h.

### Gradient

First, we want to compute the partial derivate $\frac{\partial}{\partial w_k}$ of the cross-entropy:
$$
\begin{aligned}
\frac{\partial}{\partial w_k} H(y, p(c|X))
&= - \frac{1}{n} \sum_i^n \frac{\partial}{\partial w_k} log(softmax(x_iW))\odot y_i \\
&= - \frac{1}{n} \sum_i^n \frac{\partial}{\partial w_k} log(softmax(W, x_i, j)) & \text{with} \ y_{ij} = 1 \\
&= - \frac{1}{n} \sum_i^n \frac{\partial}{\partial w_k} log\left(\frac{e^{x_iw_j}}{\sum_{l = 1}^h e^{x_iw_l}}\right) \\
&= - \frac{1}{n} \sum_i^n \frac{\partial}{\partial w_k} \left(x_iw_j - log\left(\sum_{l = 1}^h e^{x_iw_l}\right)\right) \\
&=  \frac{1}{n} \sum_i^n \left( \frac{\partial \sum_{l = 1}^h e^{x_iw_l}}{\partial w_k} \frac{\partial log\left(\sum_{l = 1}^h e^{x_iw_l}\right)}{\partial \sum_{l = 1}^h e^{x_iw_l}} - \frac{\partial}{\partial w_k} x_iw_j \right) \\
&=  \frac{1}{n} \sum_i^n \left( \frac{ x_i^Te^{x_iw_k}}{\sum_{l = 1}^h e^{x_iw_l}} - x_i^T y_{ik} \right) \\
&=  \frac{1}{n} \sum_i^n x_i^T \left(softmax(W, x_i, k) - y_{ik} \right) \\
\end{aligned}
$$ {#eq:partial}

So the gradient of the cross-entropy is given by:

$$
\begin{aligned}
\nabla_W H(y, p(c | X)) &=
\begin{bmatrix} \frac{\partial }{\partial w_1} H(y, p(c | X)) & \cdots & \frac{\partial }{\partial w_h} H(y, p(c | X)) \end{bmatrix} \\
&= \begin{bmatrix}  \frac{1}{n} x_i^T \sum_i^n \left( softmax(W, x_i, 1) - y_{i1} \right)& \cdots &  \frac{1}{n} \sum_i^n x_i^T \left( softmax(W, x_i, h) - y_{ih} \right)\end{bmatrix} \\
&= \begin{bmatrix}  \frac{1}{n} \sum_i^n x_i^T\left( softmax(x_iW) - y_i \right) \end{bmatrix} \\
\end{aligned}
$$

Here we can constate that the gradient give us the features multiply by the error between what we predict ($softmax$) and the real values ($y$).
If the error is big, the $x_i$ will have more weight in the computation of the gradient.

So we can compute the optimal $W$ thanks to a gradient descent:

$$
\begin{aligned}
W^{i + 1} &= W^{i} - \eta \nabla_W H(y, p(c | X)) \\
&= W^i - \eta  \frac{1}{n} \sum_i^n x_i^T(softmax(x_iW) - y_i)
\end{aligned}
$$
