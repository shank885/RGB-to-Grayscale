----------------------------------------------------------------------------------------------------

# RGB to Grayscale

Converting an RGB Image to a Grayscale Image Using Neural Network

----------------------------------------------------------------------------------------------------

## Introduction

In this method, I have developed an neural network based autoencoder which learns the functions to convert a RGB image to Grayscale image.
An autoencoder is a deep neural network which tries to copy its input to output. First encoder compresses the image to its latent space representation and then the decoder reconstructs the image from its latent space representation. I have used [**Undercomplete Autoencoder**](https://en.wikipedia.org/wiki/Autoencoder) to achieve my desired goal. The autoencoder learns by minimizing the loss function. I have used [**Mean Squarred Error**](https://en.wikipedia.org/wiki/Mean_squared_error) as my loss fuction. The model has been trained for 50 epochs. 

----------------------------------------------------------------------------------------------------

## Package Requirements

- [**TensorFlow == 1.14.X**](https://www.tensorflow.org/versions/r1.14/api_docs/python/tf)
- [**Python == 3.6.X**](https://www.python.org/downloads/release/python-360/)
- [**OpenCV == 3.4.2**](https://docs.opencv.org/3.4.2/)
- [**NumPy == 1.16.5**](https://pypi.org/project/numpy/1.16.5/)
- [**Glob2 == 0.5**](https://pypi.org/project/glob2/) 

----------------------------------------------------------------------------------------------------

## Results

### Original RGB Image

<p align="center"><img width="40%" src="data/input_images/rose.jpg" /></p>

### Encoded Grayscale Image

<p align="center"><img width="40%" src="data/output_images/rose.jpg" /></p>


----------------------------------------------------------------------------------------------------

## References
 
- [**Undercomplete Autoencoder**](https://en.wikipedia.org/wiki/Autoencoder)
- [**Mean Squarred Error**](https://en.wikipedia.org/wiki/Mean_squared_error)

----------------------------------------------------------------------------------------------------
