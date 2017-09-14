# style_kernel
Add different kernels on gram matrix in style transfer algorithm

folder structure will be created on the run.
     
then after download this mat file - [vgg197.mat](https://drive.google.com/open?id=0B4LqTUxVvVoXdHZpSVRfRGlBbEU)
which contains the weights of the vgg19 network. place it in the same location where neural_style_loop.py located (root directory)

you can change the sigma of the exponential kernel weights by changing in the neural_style_loop.py line 118.

`range_sigma = [1e18, 1e19, 1e21, 1e22, 1e25, 1e30]`

you can change the style weights by changing in the neural_style_loop.py line 119.

`range_sw = [1e10, 1e15, 1e20, 1e25, 1e30, 1e40, 1e50]`

then select the which you want to apply form `neural_style_loop.py` line 179
- 0  - dotproduct 
- 1  - exponential
- 2  - mattern
- 3  - polynomial

after that run the code
`python neural_style_loop.py`
