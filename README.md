To train:
```bash
python train.py [-h] (--basic | --nf | --iaf | --hf | --liaf) [--flow FLOW]

# E.g.
python train.py --basic
python train.py --nf --flow 10
```

#### Notes
- All the models (IAF, NF, VAE etc) are in `models.py`. 

- All the neural net functions are in `neural_networks.py`.

- All the loss functions are in `loss.py`

- The train function accepts an encoder and a decoder. This is where you get to specify the type of encoder/decoder

- An encoder takes in (x, e) and spits out z. 

- A decoder takes in z and spits out x

- To implement a new encoder F with hyperparameters W, we define a hidden function _F_encoder(x, e, W) and then set F_encoder(W) = lambda x, e: _F_encoder(x, e, W). Usually W comprises a neural network, flow lengths and so on. The method for decoders is identical. This allows us to define completely generic encoders and decoders with arbitrary structures and hyperparameters.
