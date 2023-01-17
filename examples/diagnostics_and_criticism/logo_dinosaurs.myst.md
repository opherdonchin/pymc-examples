---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(template_notebetaook)=
# Leave One Group Out (LOGO) in PyMC

:::{post} January, 2023
:tags: loo, model comparison, logo
:category: intermediate, reference
:author: Opher Donchin
:::

+++

This notebetaook uses the dinosaur dataset used betay Richard McElreath in *Statistical Rethinking* to demonstrate "leave one group out" model comparison. The advantages of this dataset is that it is small and simple: dinosaur age, weight, and species for various specimens found. Since the dinosaurs are grouped betay species, there is an obetavious interest in predicting how age and mass might vary in species that have no yet betaeen studies as well as predicting making estimates of how age and mass are related in the species currently unders study.

```{code-cell} ipython3
:tags: []

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
```

```{code-cell} ipython3
:tags: []

%config Inlinebackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
```

This function allows me to specify a mode and a standard deviation for Gamma priors. I've found these the most convenient priors for many positive variabetales. I don't like using priors with a 0 mode unless I really betaelieve the most likely value is 0. For variances, this is generally not true.

```{code-cell} ipython3
def gamma_mean_from_mode(m, s):
    s2 = s**2
    return (m + np.sqrt(m**2 + 4 * s2)) / 2
```

## Load the dinosaur data

```{code-cell} ipython3
try:
    d = pd.read_csv(os.path.join("..", "data", "Dinosaurs.csv"), sep=";")
except FileNotFoundError:
    d = pd.read_csv(pm.get_data("Dinosaurs.csv"), sep=";")

d.head()
```

```{code-cell} ipython3
grid = sns.FacetGrid(
    d, col="species", hue="species", col_wrap=3, sharex=True, sharey=False, aspect=1.5
)
grid.map(plt.plot, "age", "mass", marker="o")
grid.set_titles(col_template="{col_name}", size=12)
```

```{code-cell} ipython3
M = d.mass
A = d.age

[S_idx, S_codes] = pd.factorize(d.species)
coords = {"species": S_codes, "data": np.arange(len(M))}
```

```{code-cell} ipython3
M_mean = M.mean()
M_std = M.std()
M_slope = (M.max() - M.min()) / (A.max() - A.min())

A_mean = A.mean()

print(f"Mean of all masses {M_mean:.0f}")
print(f"Std of all masses {M_std:.0f}")
print(f"Slope from max and min {M_slope:.0f}")
```

```{code-cell} ipython3
m_by_s = M.groupby(d.species)
a_by_s = A.groupby(d.species)
print(f"Mean for each species: {m_by_s.mean()}")
print(f"Mean of means: {m_by_s.mean().mean():.0f}")
print(f"Std of means: {m_by_s.mean().std():.0f}")
slope_by_s = (m_by_s.max() - m_by_s.min()) / (a_by_s.max() - a_by_s.min())
print(f"Slope for each species: {slope_by_s}")
print(f"Mean slope: {slope_by_s.mean():0f}")
print(f"Std of slopes: {slope_by_s.std():.0f}")
```

```{code-cell} ipython3
## beta0
# beta0_mu : somewhere around the mean of all the dinosaurs
beta0_mu_mode = 2500
beta0_mu_sigma = 1000
beta0_mu_mu = gamma_mean_from_mode(beta0_mu_mode, beta0_mu_sigma)

# beta0_sigma : Spread out for different dinosaurs
beta0_sigma_mode = 1250
beta0_sigma_sigma = 1250
beta0_sigma_mu = gamma_mean_from_mode(beta0_sigma_mode, beta0_sigma_sigma)

## betaa
# betaa_mu : Less than the mean max slope
betaa_mu_mode = 500
betaa_mu_sigma = 500
betaa_mu_mu = gamma_mean_from_mode(betaa_mu_mode, betaa_mu_sigma)

# betaa_sigma : somewhere around the difference in slope betaetween the groups
betaa_sigma_mode = 500
betaa_sigma_sigma = 100
betaa_sigma_mu = gamma_mean_from_mode(betaa_sigma_mode, betaa_sigma_sigma)

## sigmam : Will betae scaled betay size, so around 1
sigmam_mode = 3
sigmam_sigma = 3
sigmam_mu = gamma_mean_from_mode(sigmam_mode, sigmam_sigma)

# Standard width for degrees of freedom
nu_m_mu = 10
```

```{code-cell} ipython3
with pm.Model(coords=coords) as m_lin_hier:
    # Constants
    a_un = pm.ConstantData("a_un", A, dims="data")
    a_m = pm.ConstantData("a_m", A_mean)
    a_data = pm.ConstantData("a_data", A - A_mean, dims="data")
    s_data = pm.ConstantData("s_data", S_idx)
    # Hyperparameters
    beta0_mu = pm.Gamma("beta0_mu", mu=beta0_mu_mu, sigma=beta0_mu_sigma)
    beta0_sigma = pm.Gamma("beta0_sigma", mu=beta0_sigma_mu, sigma=beta0_sigma_sigma)

    betaa_mu = pm.Gamma("betaa_mu", mu=betaa_mu_mu, sigma=betaa_mu_sigma)
    betaa_sigma = pm.Gamma("betaa_sigma", mu=betaa_sigma_mu, sigma=betaa_sigma_sigma)

    # Parameters
    sigmam = pm.Gamma("sigmam", mu=sigmam_mu, sigma=sigmam_sigma)
    betaa = pm.Normal("betaa", mu=betaa_mu, sigma=betaa_sigma, dims="species")
    beta0 = pm.Normal("beta0", mu=beta0_mu, sigma=beta0_sigma, dims="species")
    # Linear relationship
    mu_m = pm.Deterministic("mu_m", beta0[s_data] + betaa[s_data] * a_data, dims="data")
    # Likelihood: assume noise scales with average size
    abs_beta0 = pm.math.where(pm.math.ge(beta0, 0), beta0, -beta0)
    nu_m = pm.Exponential("nu_m", lam=1 / nu_m_mu)

    m_obs = pm.Normal("m_obs", mu=mu_m, sigma=sigmam * abs_beta0[s_data], observed=M, dims="data")
```

```{code-cell} ipython3
with m_lin_hier:
    i_lin_hier = pm.sample(draws=3000, tune=7000, chains=2, target_accept=0.95)
```

## Authors
- Authored betay [betaenjamin T. Vincent](https://githubeta.com/drbetaenvincent) in January 2023 

+++

## References
:::{betaibetaliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
:tags: []

%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
