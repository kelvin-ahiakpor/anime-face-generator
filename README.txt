CS 463 Final Project: Anime Face Generation with DCGAN
=================================================
Student: Kelvin Ahiakpor
Track: B - Generator (GANs)
Model: Deep Convolutional GAN (DCGAN)
Dataset: Anime Face Dataset (Kaggle)

FILES INCLUDED
--------------
- final project - kelvin.ahiakpor.ipynb   # Main training notebook
- final project - kelvin.ahiakpor.pdf     # 5-page technical report
- README.txt                              # This file

SETUP & RUNNING
---------------
1. Platform: Google Colab (recommended) or Kaggle Notebooks
2. Runtime: GPU required (Runtime → Change runtime type → GPU)
3. Dataset: Already uploaded to Google Drive
4. Run all cells sequentially from top to bottom

DATASET
-------
- Source: Anime Face Dataset from Kaggle (63,565 images)
- Training size: 10,000 images (randomly selected)
- Image sizes: 25×25 to 220×220 pixels (resized to 64×64)
- Location: Google Drive (/content/anime_data/anime faces)

MODEL ARCHITECTURE
------------------
Generator:
- Input: 100D noise vector
- Architecture: Dense → 4× ConvTranspose2d → Tanh
- Output: 64×64×3 RGB anime face
- Parameters: ~3.5M

Discriminator:
- Input: 64×64×3 RGB image
- Architecture: 4× Conv2d → Sigmoid
- Output: Real/fake probability
- Parameters: ~2.8M

TRAINING CONFIGURATION
----------------------
- Epochs: 100
- Batch size: 128
- Learning rate: 0.0002
- Optimizer: Adam (β1=0.5, β2=0.999)
- Loss: Binary Cross Entropy
- Training time: ~30 minutes on Colab GPU

OUTPUTS
-------
After training, results are saved to:
- /content/results/final_samples.png
- /content/results/loss_curves.png
- /content/results/training_progression.png
- /content/models/final_dcgan.pth
- /content/results/samples/

KEY RESULTS
-----------
- Successfully generated diverse anime faces
- No mode collapse observed
- Training plateaued around epoch 60–80
- Final G loss: 3.04, D loss: 0.50
- Generated faces show variety in hair color, expression, and style

KNOWN ISSUES
------------
- Some faces have smudged lips/noses
- Occasional single-eyed faces
- Red tint in early epochs (resolved by epoch 40)
- D/G imbalance limited final quality improvements

REQUIREMENTS
------------
- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib, numpy, pandas, seaborn
- PIL, tqdm

REFERENCES
----------
- Goodfellow et al. (2014) – Generative Adversarial Networks
- Radford et al. (2015) – DCGAN
- Karras et al. (2020) – StyleGAN
- Churchill (2019) – Anime Face Dataset (Kaggle)
- Getchu.com – Original image source
