import numpy as np
        
def normalize_image(img):
    img = np.asarray(img, dtype='float32')
    img /= 127
    img -= 1
    return img