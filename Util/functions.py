import numpy as np
import cv2 as cv
from PIL import Image


def RGB2YCBCR(img):
    y = 0.257*img[:,:,2]+0.504*img[:,:,1]+0.098*img[:,:,0] +16/255
    cb = -0.148*img[:,:,2]-0.291*img[:,:,1]+0.439*img[:,:,0]+128/255
    cr = 0.439*img[:,:,2]-0.368*img[:,:,1]-0.071*img[:,:,0]+128/255




    temp_img = img
    temp_img[:,:,0] = y
    temp_img[:,:,1] = cb
    temp_img[:,:,2] = cr

    return temp_img

# =====================
# PIL <-> OpenCV
# =====================
def pil2cv(image):
    ''' PIL -> OpenCV '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  
        pass
    elif new_image.shape[2] == 3: 
            new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image


def cv2pil(image):
    ''' OpenCV -> PIL '''
    new_image = image.copy()
    if new_image.ndim == 2:  
        pass
    elif new_image.shape[2] == 3:  
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image

# =====================
# Generate rainy image
# =====================
def SynthesizeRainyImage(img,rain_noise,synthe_type="blend"):
    rain_noise3 = np.zeros_like((img))
    for i in range(3):
        rain_noise3[:,:,i] = rain_noise

    if synthe_type == "blend":
        rainy_image = 1 - (1 - rain_noise3) * (1 - img)
    elif synthe_type == "add" :
        rainy_image = np.clip(rain_noise3+img,0,1)
    elif synthe_type == "random":
        synthe_idx = np.random.randint(2)
        if synthe_idx == 0:
            rainy_image = 1 - (1 - rain_noise3) * (1 - img)
        else:
            rainy_image = np.clip(rain_noise3+img,0,1)

    return rainy_image

def apply_motion_blur(image, size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv.warpAffine(k, cv.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )        
    return cv.filter2D(image, -1, k) 

def CustomizedRain(img, motion_length=15, motion_angle=60, scale=1.2, range_min=0.65, range_max=0.9, uniform_noise_amount=1.2, gauss_radius=0.5):
    [h, w, c] = img.shape

    # Generate uniformaly distributed random numbers.
    # Adjust the noise amount and crop between 0 and 1.
    uniform_noise = (np.random.rand(h, w) - 0.5) * uniform_noise_amount + 0.5
    uniform_noise = np.clip(uniform_noise, 0, 1)
    rain_noise = uniform_noise

    # Apply Gaussian filter. 
    rain_noise = cv.GaussianBlur(rain_noise, (5, 5), np.sqrt(gauss_radius))

    # The noise is scaled by threshold values.
    rain_noise = (rain_noise - range_min) *np.power((range_max - range_min),-1)
    rain_noise = np.clip(rain_noise, 0, 1)

    # Apply motion filter.
    rain_noise = apply_motion_blur(rain_noise, motion_length, motion_angle)

    # Adjust a internsity of rain noise.
    rain_noise = rain_noise * scale

    return rain_noise



def OutputRainNoise(image, rain_number):
    if rain_number == 0:
        # Parameters
        motion_length = 10 + (np.random.randint(6) - np.random.randint(6)) # length (default: 15)
        motion_angle = 90 + 40 * (np.random.rand() - np.random.rand()) # angle
        range_min = 0.58 + 0.04 * (np.random.rand() - np.random.rand()) # default: (0.55 or 0.6)
        scale = 1.3 + 0.2 * (np.random.rand() - np.random.rand()) #intensity of rain noise (default: 1.2)
        range_max = 0.9 #default: 0.9
        uniform_noise_amount = 1.0 + (np.random.rand() - np.random.rand()) #rain noise amount (default: 1.2)
        gauss_radius = 0.5 + 0.2 * (np.random.rand() - np.random.rand()) #radius (default: 0.5)

        # Implementation
        rain_noise = CustomizedRain(image, motion_length, motion_angle, scale, range_min, range_max, uniform_noise_amount, gauss_radius)
    
    elif rain_number > 0:
        rain_number = np.random.randint(14)+1
        # Parameters
        motion_length = 15 # length (default: 15)
        motion_angle = 60 + 10 * (np.floor((rain_number + 1)/2) - 1) # funtion of rain_number
        range_min = 0.6 + 0.05 * np.power(-1, rain_number) # function of rain_numnber
        scale = 1.2 # intensity of rain noise (default: 1.2)
        range_max = 0.9 # default: 0.8
        uniform_noise_amount = 1.2 # rain noise amount(default: 1.5)
        gauss_radius = 0.5 # radius(default: 0.5)

        # Implementation
        rain_noise = CustomizedRain(image, motion_length, motion_angle, scale, range_min, range_max, uniform_noise_amount, gauss_radius)

    return rain_noise
