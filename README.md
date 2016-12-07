To run:
```bash
python vanilla_vae.py
```
or substitute another script name in. Also see the notebook for usage.

This is a massive refactoring of the original VAE code.

Now all the models (IAF, NF, VAE etc) sit inside models.py. I have put all the neural net functions in utils.py.

NB IAF doesn't work at the moment. NF works but badly.

Main changes:
(1) Loss function now uses cross-entropy for the reconstruction cost instead of squared error
(2) For flow-based methods: now we use Monte Carlo estimate of $\log{q_0(z_0)}$ and \[\log{p(z_k)}\]: the loss function is then \[\log{q_0(z_0)} - \sum_{i=1}^{K} \log \det J_i - \log{p(x|z_k)} - \log{p(z_k)}\]

