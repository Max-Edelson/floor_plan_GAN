# floor_plan_GAN
## Data
### Image data
### Text data
`preprocess.py` filters text elements which are not relevant for generation. This is done to simplify the problem for the model. Once this is complete, the `tokenizer.py` has dataset implementation which tokenizes the string. Our tokenization scheme splits the svg into tags and attributes. The values which are not explicitly categorical are tokenized at the character level. For example, fonts have discrete values like 'Helvetica' which will be treated as its own token, but other things like numbers will be split at the character level. Opening and closing tags are also treated as their own tokens. This scheme allows each token to be highly disctinct and meaningful, as well as incorporate knowledge about xml format to smplify the challenge for the model.
## Train files
There are versions to train the GANs on the original FloorPlanCAD dataset as well as binarized versions of the dataset. There are also seperate train files to train with Wasserstein loss as well as WGAN with gradient penalty. Training for svg generation is also a seperate train file.
## Baseline models
The VAE models are found in `vae_models.py` and can be run with `vae_train.py`. DCGAN and SA-GAN can be found as `Generator` and `Generator2` in `GAN_model.py`. To run the models in various train configurations using the different training files, they must be imported and used in each file as desired.
## SVG GAN
The string generation model is in `SVG_GAN.py`. This can only be run with `train_svg_gan.py` as the data format is not compatible with the other train files.
