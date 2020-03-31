import numpy as np

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