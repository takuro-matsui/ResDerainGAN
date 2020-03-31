import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import glob
from PIL import Image
import cv2 as cv
import numpy as np


from Util.functions import *

class ImageDataset(Dataset):
    def __init__(self, hr_shape,rain_type=0,train_phase=1):
        hr_height, hr_width = hr_shape
        
        self.tensor_setup = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.resize = transforms.Resize((hr_height, hr_width), Image.BICUBIC)
        self.randomhorizontalflip = transforms.RandomHorizontalFlip()
        self.randomverticalflip = transforms.transforms.RandomVerticalFlip()
        self.randomresizedcrop = transforms.RandomResizedCrop((hr_height, hr_width))
    

        if train_phase:
            self.files_ground_truth = sorted(glob.glob("./image/train_data/*.jpg"))
        else:
            self.files_ground_truth = sorted(glob.glob("./image/train_data/*.jpg"))

        self.rain_number = rain_type

    def __getitem__(self, img_index):
        np.random.seed()

        # Load image
        ground_truth = Image.open(self.files_ground_truth[img_index % len(self.files_ground_truth)])
        ground_truth = self.randomresizedcrop(ground_truth)
        ground_truth = self.randomhorizontalflip(ground_truth)
        ground_truth = self.randomverticalflip(ground_truth)

        # PIL -> CV2
        cv_ground_truth = pil2cv(ground_truth)/255

        # Generate rainy image
        cv_rain_noise = OutputRainNoise(cv_ground_truth, self.rain_number)
        cv_rainy_image = SynthesizeRainyImage(cv_ground_truth,cv_rain_noise, "random")

       # CV2 -> PIL
        rain_noise = cv2pil((cv_rain_noise*255).astype(np.uint8))
        rainy_image = cv2pil((cv_rainy_image*255).astype(np.uint8))

        return {
            "ground_truth": self.tensor_setup( ground_truth ),
            "rain_noise": self.tensor_setup( rain_noise ),
            "rainy_image": self.tensor_setup( rainy_image ),
            }
    

    def __len__(self):
        return len(self.files_ground_truth)

class TestImageDataset(Dataset):
    def __init__(self, rain_number=0,file_path = "real_world/*.jpg"):
        self.tensor_setup = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.file_path = file_path
        if self.file_path == "synthetic/":
            self.files_rainy = sorted(glob.glob("./image/test_data/" + file_path + '*rain.jpg'))
            self.files_clean = sorted(glob.glob("./image/test_data/" + file_path + '*original.jpg'))
        else:
            self.files_load_img = sorted(glob.glob("./image/test_data/" + file_path))
        self.rain_number = rain_number

    def __len__(self):
        return len(self.files_load_img)
    
    def __getitem__(self, img_index):
        output = {}
        if self.file_path == "synthetic/":
            ground_truth = Image.open(self.files_clean[img_index % len(self.files_clean)])
            rainy_image = Image.open(self.files_rainy[img_index % len(self.files_rainy)])
            cv_rainy_image = functions.pil2cv(rainy_image)
            output["ground_truth"] = self.tensor_setup( ground_truth )
        
        # Generate rainy image
        elif self.rain_number > -1:
            ground_truth = Image.open(self.files_load_img[img_index % len(self.files_load_img)])

            # PIL -> CV2
            cv_ground_truth = functions.pil2cv(ground_truth)
            cv_rain_noise = functions.OutputRainNoise(cv_ground_truth/255, self.rain_number)
            cv_rainy_image = functions.SynthesizeRainyImage(cv_ground_truth/255,cv_rain_noise, "random")

            # CV2 -> PIL
            rain_noise = (cv_rain_noise*255).astype(np.uint8)
            rainy_image = (cv_rainy_image*255).astype(np.uint8)

            rain_noise = functions.cv2pil(rain_noise)
            rainy_image = functions.cv2pil(rainy_image)

            output["ground_truth"] = self.tensor_setup( ground_truth )
            output["rain_noise"] = self.tensor_setup( rain_noise )


        elif self.rain_number == -1:
            
            rainy_image = Image.open(self.files_load_img[img_index % len(self.files_load_img)])
            cv_rainy_image = functions.pil2cv(rainy_image)
        
        temp_img = RGB2YCBCR(cv_rainy_image)
        img = temp_img[:,:,0]

        kernel_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
        kernel_y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])

        sobel_x = cv.filter2D(img, -1, kernel_x)
        sobel_y = cv.filter2D(img, -1, kernel_y)

        # CV2 -> PIL
        sobel_x = (sobel_x*255).astype(np.uint8)
        sobel_x = functions.cv2pil(sobel_x)

        sobel_y = (sobel_y*255).astype(np.uint8)
        sobel_y = functions.cv2pil(sobel_y)

        temp_img = (temp_img*255).astype(np.uint8)
        ycbcr = functions.cv2pil(temp_img)

        output["rainy_image"] = self.tensor_setup( rainy_image )
        output["sobel_x"] = self.tensor_setup( sobel_x )
        output["sobel_y"] = self.tensor_setup( sobel_y )
        output["ycbcr"] = self.tensor_setup( ycbcr)


        return output 

class TestImageDatasetSimple(Dataset):
    def __init__(self, rain_number=0, synthetic=1, file_path = "real_world/*.jpg"):
        
        test_data_path = "../test_dataset/"
        self.tensor_setup = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.file_path = file_path
        self.synthetic = synthetic
        self.rain_number = rain_number

        if synthetic:
            test_data_path = test_data_path+"synthetic/"
            if file_path == "synthetic/": 
                self.files_clean = sorted(glob.glob(test_data_path + file_path + "*original.jpg"))
                self.files_rainy = sorted(glob.glob(test_data_path + file_path + "*rain.jpg"))

        else:
            test_data_path = test_data_path+"real_world/"
            if file_path == "real_world/":
                self.files_rainy = sorted(glob.glob(test_data_path + file_path + "*.jpg"))



        

    def __len__(self):
        return len(self.files_clean) or len(self.files_rainy)
    
    def __getitem__(self, img_index):
        output = {}

        if self.synthetic:
            if self.file_path == "test_syn/":
                img = Image.open(self.files[img_index % len(self.files)])
                w,h = img.size
                ground_truth = img.crop((0,0,w/2,h))
                rainy_image = img.crop((w/2,0, w,h))
            else:
                ground_truth = Image.open(self.files_clean[img_index % len(self.files_clean)])
                rainy_image = Image.open(self.files_rainy[img_index % len(self.files_rainy)])
            output["ground_truth"] = self.tensor_setup( ground_truth )
                
        else:
            rainy_image = Image.open(self.files_rainy[img_index % len(self.files_rainy)])


        output["rainy_image"] = self.tensor_setup( rainy_image )



        return output 
