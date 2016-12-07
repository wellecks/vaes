To train:
```bash
python train.py [-h] (--basic | --nf | --iaf) [--flow FLOW]

# E.g.
python train.py --basic
python train.py --nf --flow 10
```

This is a massive refactoring of the original VAE code.

All the models (IAF, NF, VAE etc) are in models.py. 

All the neural net functions are in neural_networks.py.

All the loss functions are in loss.py

NB IAF doesn't work at the moment. NF works but badly.

Main changes:

(1) Loss function now uses cross-entropy for the reconstruction cost instead of squared error.

(2) For flow-based methods: now we use Monte Carlo estimate of \[\log{q_0(z_0)}\] and \[\log{p(z_k)}\]: the loss function is then \[\log{q_0(z_0)} - \sum_{1}^{K} \log \det J_i - \log{p(x|z_k)} - \log{p(z_k)}\]

(3) The main is in vanilla_vae.py. That's where all models run from.

(4) The train function accepts an encoder and a decoder. This is where you get to specify the type of encoder/decoder

(5) An encoder takes in (x, e) and spits out z. 

(6) A decoder takes in z and spits out x

(7) We achieve this using lambda notation. To implement a new encoder F with hyperparameters W, we define a hidden function _F_encoder(x, e, W) and then set F_encoder(W) = lambda x, e: _F_encoder(x, e, W). Usually W comprises a neural network, flow lengths and so on. The method for decoders is identical. This allows us to define completely generic encoders and decoders with arbitrary structures and hyperparameters.

(8) Brought the IAF parameters into the same part of the code as the initialization parameters.

(9) Separated the definition of the neural network from the encoder.


