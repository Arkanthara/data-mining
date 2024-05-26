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
\text{max arg}_w \sum_i^n p(y_i | x_i)
\end{aligned}
$$ {#eq:total}

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
\text{max arg}_w \sum_i^n p(y_i | x_i)
&= \text{min arg}_w \frac{1}{\sum_i^n p(y_i | x_i)}\\
&= \text{min arg}_w \sum_i^n\frac{1}{\sigma(x_iw)^{y_i}(1 - \sigma(x_iw))^{1 - y_i}} \\
&= \text{min arg}_w \ log\left(\sum_i^n\frac{1}{\sigma(x_iw)^{y_i}(1 - \sigma(x_iw))^{1 - y_i}}\right) \\
&= \text{min arg}_w \sum_i^n\left(- log(\sigma(x_iw)^{y_i}(1 - \sigma(x_iw))^{1 - y_i}) \right) \\
&= \text{min arg}_w \sum_i^n\left( -y_ilog(\sigma(x_iw)) - (1 - y_i)log(1 - \sigma(x_iw)) \right)
\end{aligned}
$$ {#eq:cost}

In the equation @eq:cost, we can apply log to the equation because the log function is always positive and increasing, and since we have said that $p(y_i | x_i)$ is always positive.

So we have inverted the likelihood, and then we have apply log function to result.
As the log of the invert is equal to negative log (example: $log\left(\frac{1}{x}\right) = -log(x)$), we call the cost function the negative log likelihood function.

As $y$ has a size of $n \times h$, $X$ a size of $n \times d$ and $w$ a size of $d\times h$, we can write the cost function with matrix product:

$$
\begin{aligned}
\text{min arg}_w \sum_i^n\left( -log(\sigma(x_iw))y_i - log(1 - \sigma(x_iw))(1 - y_i) \right) \\
= \text{min arg}_w \left( -y^Tlog(\sigma(Xw)) - (1 - y^T)log(1 - \sigma(Xw)) \right) \\
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
&= \frac{\partial}{\partial w_k} \left( -y^Tlog(\sigma(Xw)) - (1 - y^T)log(1 - \sigma(Xw)) \right)\\
&= -y^T \frac{\partial}{\partial w_k}log(\sigma(Xw)) - (1 - y^T)\frac{\partial}{\partial w_k}log(1 - \sigma(Xw)) \\
&= -y^T \frac{\partial log(\sigma(Xw))}{\partial \sigma} \frac{\partial \sigma}{\partial Xw} \frac{\partial Xw}{\partial w_k} \\
&\ \ \ - (1 - y^T)\frac{\partial log(1 - \sigma(Xw))}{\partial (1 - \sigma(Xw))} \frac{\partial (1 - \sigma(Xw))}{\partial \sigma} \frac{\partial \sigma(Xw)}{\partial Xw}  \frac{\partial Xw}{\partial w_k} \\
&= -y^T \frac{1}{\sigma(Xw))} \sigma(Xw)(1 - \sigma(Xw)) X_{:,k} \\
&\ \ \ -(1 - y^T)\frac{1}{1 - \sigma(Xw)}(-1)\sigma(Xw)(1 - \sigma(Xw)) X_{:, k} \\
&= ((1 - y^T)\sigma(Xw) -y^T (1 - \sigma(Xw))) X_{:, k} \\
&= (\sigma(Xw) -y^T\sigma(Xw) -y^T 1_{n, 1} + y^T\sigma(Xw)) X_{:, k} \\
&= \left(\sigma(Xw)  - \sum_i^n y_i \right)X_{:, k}\\
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

We have $h$ classes $c_1, \cdots, c_h \in C$.

If $y_i = c_k \in C$, we define $\tilde{y}_i$ a vector of size $1 \times h$ like this:

$$
\tilde{y}_{ij} = \left\{
\begin{aligned}
1 & \ \text{if} j = k \\
0 & \ \text{else}
\end{aligned}
\right.
$$ {#eq:tildey}

So we have for instance $\tilde{y}_i = \begin{bmatrix} 0 & 0 & 1 & 0 \end{bmatrix}$ if $h = 4$.

Now, we note $y_i = \tilde{y}_i$. So $y$ is of size $n \times h$.

We have for each $k \in [1, h]$ that the probability that $y_{ik} = 1$ or $0$ depends on $x_i$.

So we have:

$$p(y_{ik} | x_i) = h_k(x_i)$$ {#eq:1}

Thus, we have:

$$
p(y_{i} | x_i) = \prod_{k = 1}^h p(y_{ik} | x_i)^{y_{ik}}
$$ {#eq:2}

But what is the function $h_k(x_i)$ ? Is it the sigmoid function ?

No, it's not the sigmoid function, because this function work only for two variables.
So we want a function like the sigmoid function that works for more than two variables, that's why, we'd try to generalize the sigmoid function to more than one variable

If we remember the equation @eq:sigma, we have (binary case):
$$
\begin{aligned}
p(y_i = 1 | x_i) 
&= \frac{e^{x_iw}}{e^{x_iw} + 1} \\
\end{aligned}
$$ {#eq:3}

Now, we suppose that we have
$$
W = \begin{bmatrix} w_1 & \cdots & w_h \end{bmatrix}
$$ {#eq:w}
with each parameter $w_k$ defined specially for the class $k$.

In binary case, we can define $W = \begin{bmatrix} w_1 & w_2 \end{bmatrix}$, with $w_1$ a column vector of $0$.

So we can write the equation @eq:3 like this:

$$
\begin{aligned}
p(y_i = 1 | x_i) 
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
&= \frac{e^{x_iw_k}}{\sum_{j = 1}^{h} (e^{x_iW})_{1, j}} \\
&= \text{softmax}(W, x_i, k)
\end{aligned}
$$ {#eq:softmax}

We can easily see that $\text{softmax}$ is between $0$ and $1$:
As the exponential function is always positive, we have that:

$$
\begin{aligned}
&0 &< e^{x_iw_k} &< e^{x_iw_k} + \sum_{j \neq k}^h e^{x_iw_j} \\
\Leftrightarrow \ \ \ &0 &< e^{x_iw_k} &< \sum_{j = 1}^h e^{x_iw_j} \\
\Leftrightarrow \ \ \ &0 &< e^{x_iw_k} &< \sum_{j = 1}^h e^{x_iw_j} \\
\Leftrightarrow \ \ \ &0 &< \frac{e^{x_iw_k}}{\sum_{j = 1}^h e^{x_iw_j}} &< 1 \\
\Leftrightarrow \ \ \ &0 &< \text{softmax}(W, x_i, k) &< 1 \\
\end{aligned}
$$ {#eq:softzeros}

So thanks to equation @eq:softmax, we have:

$$
\begin{aligned}
p(y_{i} | x_i) &= \prod_{k = 1}^h p(y_{ik} = 1 | x_i)^{y_{ik}} \\
&= \prod_{k = 1}^h \text{softmax}(W, x_i, k)^{y_{ik}} \\
\end{aligned}
$$ {#eq:probsoft}

### Cost function

Now, as for the logistic regression, we want to find a cost function that maximise $p(y_{i} |x_i)$ for each instance.
So we want to maximise the likelihood.
According to equation @eq:softmax and equation @eq:softzeros, we have that $p(y_i |x_i) \neq 0$.

And we also want to use the gradient descent to optimize the parameter $W$, so we need a cost function to minimise.

So we have:
$$
\begin{aligned}
\arg \max_{W} \sum_i^n p(y | X) 
&= \arg \min_{W} \frac{1}{\sum_i^n p(y | X)} \\
&= \arg \min_{W} log\left(\frac{1}{\sum_i^n p(y | X)}\right) \\
&= \arg \min_{W} - log\left(\sum_i^n p(y | X)\right) \\
&= \arg \min_{W} - \sum_i^n log\left(p(y | X)\right) \\
&= \arg \min_{W} - \sum_i^n log\left(\prod_{k = 1}^h \text{softmax}(W, x_i, k)^{y_{ik}}\right) \\
&= \arg \min_{W} - \sum_i^n \sum_{k = 1}^h log\left(\text{softmax}(W, x_i, k)^{y_{ik}}\right) \\
&= \arg \min_{W} - \sum_i^n \sum_{k = 1}^h y_{ik} log\left(\text{softmax}(W, x_i, k)\right) \\
&= \arg \min_{W} - \sum_i^n \sum_{k = 1}^h y_{ik} log\left(\frac{e^{x_iw_k}}{\sum_{j = 1}^{h} (e^{x_iW})_{1, j}}\right) \\
&= \arg \min_{W} - \sum_i^n \sum_{k = 1}^h y_{ik} \left(x_iw_k - log\left(\sum_{j = 1}^{h} (e^{x_iW})_{1, j}\right) \right) \\
&= \arg \min_{W}  \sum_i^n \sum_{k = 1}^h y_{ik} \left(log\left(\sum_{j = 1}^{h} (e^{x_iW})_{1, j}\right) - x_iw_k \right) \\
&= \arg \min_{W}  \sum_k^h 1_{1, h} y^T \left(log\left(e^{XW}\right) - Xw_k \right) 1_{h, 1} \\
&= NLL(W, X, y)
\end{aligned}
$$ {#eq:nll}

Note that we have $W$ defined in equation @eq:w and that:

| Matrix | Size |
|:------:|:----:|
|$X$|$n \times d$|
|$y$|$n \times h$|
|$W$|$d \times h$|

So if we look at the sizes, we have:
$$
\begin{aligned}
\ \ \  \sum_k^h 1_{1, h} y^T \left(log\left(e^{XW}\right) - Xw_k \right) 1_{h, 1} \\
= \sum_k^h 1 \times h \times (n\times h)^T \times \left( n \times d \times d \times h - n \times d \times d \times 1 \right) \times h \times 1 \\
= \sum_k^h 1 \times n \times \left( n \times h - n \times 1 \right) \times h \times 1 \\
= \sum_k^h 1 \times n \times n \times h \times h \times 1\\
= \sum_k^h 1  \\
= scalar
\end{aligned}
$$

This function is called the negative logarithm likelihood.

### Gradient

We want to find the gradient of the $NLL$ function obtained in equation @eq:nll.

First, we want to compute the gradient of the softmax function.

We have:
