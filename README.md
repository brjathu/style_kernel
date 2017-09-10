# style_kernel
Add different kernels on gram matrix in style transfer algorithm

first run the directory.py file, which will create the requied folder structure. range_sigma list should be same as its in the neural_style_loop.py file (line - 118). this will create a structure of
final
|____dot
|____exp
     |____1e+18
     |____1e+19
     ....
     
then after download this npy file - 
which contains the weights of the vgg19 network.

you can change the sigma of the exponential kernel weights by changing in the neural_style_loop.py line 118
range_sigma = [1e18, 1e19, 1e21, 1e22, 1e25, 1e30]

you can change the style weights by changing in the neural_style_loop.py line 119
range_sw = [1e10, 1e15, 1e20, 1e25, 1e30, 1e40, 1e50]

after that run the code
python neural_style_loop.py 


