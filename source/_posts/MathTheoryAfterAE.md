---
title: Mathematical Theory after Variational Autoencoder
tags: ["VAE", "MachineLearning", "Math"]
category:
- [MachineLearning]
date: 2022-08-28 22:51:12
---
## Prior and Posterior Distribution of VAE

- Posterior Distribution of Encoder
$$
P(z|x) \propto P(z)P(x|z)
$$
- Prior Distribution of latent variables
$$
P(z)
$$
- the likelyhood function defining the decoder
$$
P(x|z)
$$

## Bayesian Analysis

### reference
- [Bayesian Analysis](https://www.britannica.com/science/Bayesian-analysis)
- [Bayesian Statistics explained to Beginners in Simple English](https://www.analyticsvidhya.com/blog/2016/06/bayesian-statistics-beginners-simple-english/)

> "Bayesian statistics is a mathematical procedure that applies probabilities to statistical problems. It provides people the tools to update their beliefs in the evidence of new data."

## ELBO loss

__ELBO loss__ consists of two parts:
- KL loss
- Reconstruction loss

The formular of ELBO loss:
$$
L = \min \mathbb{E}_q [\log q(z|x) - \log p(z)] - \mathbb{E}[\log p(x|z)]
$$

*note*: in the loss function above, $x$ means the ground truth.

The KL Divergence term in the  formula above is a little different from the form widely used, which is (in continuous case):
$$
KL(P||Q) = \int_{x \in X} P(x) \log\frac{P(x)}{Q(x)}dx
$$
But it is much easier to understand the one represented in the expectation form.

### KL Divergence

#### reference

- [how to calculate the KL divergence for machine learning](https://machinelearningmastery.com/divergence-between-probability-distributions/)

- [Variational Autoencoder demystified with pytorch implementation](https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed)

##### The Meaning of  KL Divergence

> It is often desirable to quantify the difference between probability distributions for a given random variable.
>
> This can be achieved using techniques from information theory, such as the Kullback-Leibler Divergence (KL divergence), or relative entropy, and the Jensen-Shannon Divergence that provides a normalized and symmetrical version of the KL divergence. 

Here are 3 ways to measure the difference between probability distributions.

- KL divergence (Relative entropy)
- JS divergence



##### Monte-Carlo KL Divergence

```python
def kl_divergence(z, mu, std):
	# ------------------------------
	# Monte Carlo KL Divergence
	# ------------------------------
	# 1. two normal distributions
	p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
	q = torch.distributions.Normal(mu, std)
	
	# 2. log_prob
	log_qzx = q.log_prob(z)
	log_pz = p.log_prob(z)

	# 3. kl
	kl = (log_qzx - log_pz)

	# 4. sum over the last dim
	kl = kl.sum(-1)

	return kl
```

### Reconstruction Loss

The meaning of reconstruction loss is easy to understand and often implemented by MSE loss or some other criterions instead of the probability shown above.

It can prevent the VAE from collapsing:

> Some things may not be obvious still from this explanation. First, **each** image will end up with its own q. The KL term will push all the qs towards the **same** p (called the prior). But if all the qs, collapse to p, then the network can cheat by just mapping everything to zero and thus the VAE will collapse.
>
> The reconstruction term, forces each q to be unique and spread out so that the image can be reconstructed correctly. This keeps all the qs from **collapsing** onto each other.
