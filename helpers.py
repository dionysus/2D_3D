import matplotlib.pyplot as plt
from skimage import io

def plot_img(img):
    '''
    Plots the given image
    '''
    dpi = 72
    w, h = img.shape[1], img.shape[0]
    
    plt.figure(figsize=(w / float(dpi), h / float(dpi)))
    plt.axis('off')
    io.imshow(img)
    io.show()
