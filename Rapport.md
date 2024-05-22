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

$$Odds = \frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)}$$ {#eq:odds}

In logistic regression, we want that the logarithm of the odds is linear.
So according to equation @eq:odds, we want that:
$$
\begin{aligned}
log\left( \frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} \right) &= w_0 + w_1 \cdot x_{i1} + \cdots + w_d \cdot x_{id} \\
\Leftrightarrow log\left( \frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} \right) &= w_0 + x_iw \\
\Leftrightarrow log\left( \frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} \right) &= x_iw
& \ \text{if}\ x = \begin{bmatrix} 1 & x_{i1} & \cdots & x_{id} \end{bmatrix} \\
&& \ \text{and}\ w^T = \begin{bmatrix} w_0 & \cdots & w_d \end{bmatrix} \\
\Leftrightarrow \frac{p(y_i = 1|x_i)}{p(y_i = 0|x_i)} &= e^{x_iw}
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

We can easily suppose that the likelihood is always non zeros because else,
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
\text{min arg}_w \sum_i^n\left( -y_ilog(\sigma(x_iw)) - (1 - y_i)log(1 - \sigma(x_iw)) \right) \\
= \text{min arg}_w \left( -y^Tlog(\sigma(Xw)) - (1 - y^T)log(1 - \sigma(Xw)) \right)
\end{aligned}
$$ {#eq:prod}

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
$$

So according to the chain rule, we have:
$$
\begin{aligned}
\frac{\partial}{\partial w_k} \left( -y^Tlog(\sigma(Xw)) - (1 - y^T)log(1 - \sigma(Xw)) \right)
\\
= -y^T \frac{\partial}{\partial w_k}log(\sigma(Xw)) - (1 - y^T)\frac{\partial}{\partial w_k}log(1 - \sigma(Xw)) \\
= -y^T \frac{\partial log(\sigma(Xw))}{\partial \sigma} \frac{\partial \sigma}{\partial Xw} \frac{\partial Xw}{\partial w_k}
-(1 - y^T)\frac{\partial log(1 - \sigma(Xw))}{\partial \sigma} \frac{\partial \sigma}{\partial Xw} \frac{\partial Xw}{\partial w_k}
 \\
= -y^T \frac{1}{\sigma(Xw))} \sigma(Xw)(1 - \sigma(Xw)) X_{:,k}
-(1 - y^T)\left(-\frac{1}{\sigma(Xw)}\sigma(Xw)(1 - \sigma(Xw)) X_{:, k} \right)
 \\
= -y^T (1 - \sigma(Xw)) X_{:,k} +(1 - y^T)(1 - \sigma(Xw)) X_{:, k}\\
= (1 - 2y^T) (1 - \sigma(Xw)) X_{:,k}\\
\end{aligned}
$$
