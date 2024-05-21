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
So we have $p(y_i = 1 | x_i) = g(x_i)$

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
&= \frac{1}{1 + \frac{p(x_i|y_i = 0)p(y_i = 0)}{p(x_i|y_i = 1)p(y_i = 1)}} \\
&= \frac{1}{1 + \frac{p(x_i|y_i = 0)p(y_i = 0)}{p(x_i|y_i = 1)p(y_i = 1)}} \\
&= \frac{1}{1 + e^{-a}} \\
&= \sigma(a)
\end{aligned}
$$ {#eq:sigma1}

Here, in the equation @eq:sigma1, we consider that we can write $\frac{p(x_i|y_i = 0)p(y_i = 0)}{p(x_i|y_i = 1)p(y_i = 1)} = e^{-a}$ with $a$ a scalar.

So we have:

$$
p(y_i = 1 | x_i) = \sigma(a)
$$ {#eq:sigma2}

Now, we want to map all our datas $x_i$ in a scalar.
