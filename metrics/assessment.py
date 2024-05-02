"""
Non-Profit Open Software License 3.0 (NPOSL-3.0)
Copyright (c) 2023 Matteo Lai

@authors: 
         Matteo Lai, M.Sc, Ph.D. student
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
         Email address: matteo.lai3@unibo.it

         Chiara Marzi, Ph.D., Junior Assistant Professor
         Department of Statistics, Computer Science, Applications "G. Parenti" (DiSIA)
         University of Firenze, Firenze, Italy
         Email address: chiara.marzi@unifi.it
         
         Luca Citi, Ph.D., Professor
         School of Computer Science and Electronic Engineering
         University of Essex, Colchester, UK
         Email address: lciti@essex.ac.uk
         
         Stefano Diciotti, Ph.D, Associate Professor
         Dept. of Electrical, Electronic and Information Engineering – DEI "Guglielmo Marconi",
         University of Bologna, Bologna, Italy.
         Email address: stefano.diciotti@unibo.it
"""

import torch
from torchvision import transforms
import nibabel as nib
import numpy as np
import pandas as pd
import random
import os

from torchmetrics.image.fid import FrechetInceptionDistance
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.kid import KernelInceptionDistance

seed=0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Quantitative_metrics():
    def __init__(self,
        real_path: str,                 # Path to the real images
        fake_path: str,                 # Path to the fake images
        real_y_path: str,               # Path to the labels of the real images
        fake_y_path: str,               # Path to the labels of the fake images
        img_channels: int,              # Number of channels of the images (1 for grayscale, 3 for RGB)
        class_size: int,                # Number of classes
        output_txt: str,                # Name of the text file where to save the metrics values (e.g. metrics.txt)
        device: str,                    # 'cpu' or 'cuda'
        batch_size: int                 # Dimension of the batch size for KID computation
    ):
        self.real_path = real_path
        self.fake_path = fake_path
        self.real_y_path = real_y_path
        self.fake_y_path = fake_y_path
        self.output_txt = output_txt
        self.img_channels = img_channels
        self.class_size = class_size
        self.device = device
        self.bs = batch_size

    def load_images(
        self,
        path: str,                      # Relative path to directory
        convert_to_RGB: bool = False    # If the image is grayscale, set convert_to_RGB=True to compute the FID.
    ):
        img = nib.load(path) # numpy array of shape (W,H,C,N)
        img = np.asanyarray(img.dataobj)
        img = np.swapaxes(img, 0,3)               # (N,H,C,W)
        img = np.swapaxes(img, 2,3) if convert_to_RGB else np.swapaxes(img, 1,2)               # (N,H,W,C)
        img = torch.tensor(img, dtype=torch.float64)
        if self.img_channels==1 and convert_to_RGB:
            # Convert to RGB
            img = img.repeat(1,1,1,3)
        return img
    
    def rescale01(self, img):
        if img.max() > 1:
            img = img/255 # Rescale from [0,255] to [0,1]
        elif img.min() < 0:
            img = (img+1)/2 # Rescale from [-1,1] to [0,1]
        return img
    
    def rescale11(self, img):
        if img.max() > 1:
            img = (img / 127.5) - 1  # Rescale from [0, 255] to [-1, 1]
        elif img.min() >= 0 and img.max() <= 1:
            img = (img * 2) - 1  # Rescale from [0, 1] to [-1, 1]
        return img

    def preprocessing_FID(self, path):
        '''Preprocessing needed for the Inception V3 net, as suggested in 
            https://pytorch.org/hub/pytorch_vision_inception_v3/'''
        imgs_dist = self.load_images(path, convert_to_RGB=True)
        imgs_dist = self.rescale01(imgs_dist)
        imgs_dist = np.swapaxes(imgs_dist, 1,3)
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            ])
        imgs_dist = preprocess(imgs_dist)
        return imgs_dist

    def FID(self, 
        path1: str,                      # Relative path to directory of the first set of images (real or fake)
        path2: str                       # Relative path to directory of the second set of images (real or fake)
    ):
        imgs_dist1 = self.preprocessing_FID(path1)
        imgs_dist2 = self.preprocessing_FID(path2)
        fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=True)
        fid.update(imgs_dist1, real=True)   
        fid.update(imgs_dist2, real=False)
        fid_score = fid.compute()
        return fid_score.item()

    def KID(self, 
        path1: str,                      # Relative path to directory of the first set of images (real or fake)
        path1y: str,                     # Relative path to the file with the labels of the first set of images (real or fake)
        path2: str,                      # Relative path to directory of the second set of images (real or fake)
        path2y: str,                     # Relative path to the file with the labels of the second set of images (real or fake)
        batch_size: int                  # Dimension of the batch size for KID computation
    ):
        imgs_dist1 = self.preprocessing_FID(path1)
        imgs_dist2 = self.preprocessing_FID(path2)

        # Compute the KID with the whole distribution:
        kid = KernelInceptionDistance(feature=2048, subset_size=batch_size, normalize=True)
        kid.update(imgs_dist1, real=True)
        kid.update(imgs_dist2, real=False)
        kid_mean, kid_std = kid.compute()

        # Compute the KID for each class:
        real_df = pd.read_csv(path1y)
        fake_df = pd.read_csv(path2y)
        kid_classes_m = []; kid_classes_std = []
        for i in range(self.class_size):
            real_IDs = real_df[real_df['Label']==i]['ID'].to_list()
            fake_IDs = fake_df[fake_df['Label']==i]['ID'].to_list()
            x1 = imgs_dist1[real_IDs,:,:,:]
            x2 = imgs_dist2[fake_IDs,:,:,:]
            kidc = KernelInceptionDistance(feature=2048, subset_size=batch_size, normalize=True)
            kidc.update(x1, real=True)
            kidc.update(x2, real=False)
            kid_mean_c, kid_std_c = kidc.compute()
            kid_classes_m.append(kid_mean_c.item())
            kid_classes_std.append(kid_std_c.item())
        return kid_mean.item(), kid_std.item(), kid_classes_m, kid_classes_std

    def SSIM(self, 
        path1: str,                      # Relative path to directory of the first set of images (real or fake)
        path1y: str,                     # Relative path to the file with the labels of the first set of images (real or fake) 
        path2: str,                      # Relative path to directory of the second set of images (real or fake)
        path2y: str                      # Relative path to the file with the labels of the second set of images (real or fake)
    ):
        # Choose 1000 random couples of real and fake images belonging to the same class 
        imgs_real = self.load_images(path1)
        imgs_real = self.rescale11(imgs_real)
        imgs_fake = self.load_images(path2)
        imgs_fake = self.rescale11(imgs_fake)
        real_df = pd.read_csv(path1y)
        fake_df = pd.read_csv(path2y)
        metric_classes_m = []; metric_classes_std = [] 
        metric_all = []
        for i in range(self.class_size):
            real_IDs = real_df[real_df['Label']==i]['ID'].to_list()
            fake_IDs = fake_df[fake_df['Label']==i]['ID'].to_list()
            metric_scores = []
            n_iterations = 1000
            for _ in range(n_iterations):
                real_ID_rand = random.randint(0,len(real_IDs)-1)
                fake_ID_rand = random.randint(0,len(fake_IDs)-1)
                real_ID_list = real_IDs[real_ID_rand]
                fake_ID_list = fake_IDs[fake_ID_rand]

                x1 = imgs_real[real_ID_list,:,:,:]
                x2 = imgs_fake[fake_ID_list,:,:,:]
                x1 = torch.unsqueeze(x1, 0)
                x2 = torch.unsqueeze(x2, 0)

                score = ssim(np.array(x1[0,0,:,:]), np.array(x2[0,0,:,:]), data_range=2)
                metric_scores.append(score)
                metric_all.append(score)
            metric_classes_m.append(np.mean(metric_scores))
            metric_classes_std.append(np.std(metric_scores))
        metric_mean = np.mean(metric_all)
        metric_std = np.std(metric_all)    
        return metric_mean, metric_std, metric_classes_m, metric_classes_std

    def compute(self):
        print('\nCalculating assessment metrics...', flush=True)
        f = open(self.output_txt,"w")
        f.write('Computing evaluation metrics of: {}'.format(self.fake_path))
        f.write('\nConsidering the real images in:  {}'.format(self.real_path))

        # ----------------------------------------------------------------------------------------------------
        # Metrics between real and real images:
        # ----------------------------------------------------------------------------------------------------
        f.write('\n\nMetrics between real and real images:')

        # FID
        fid_score = self.FID(self.real_path, self.real_path)
        f.write('\n\nFID_rr:  {}'.format(fid_score))

        # KID
        kid_mean, kid_std, kid_classes, kid_classes_std = self.KID(self.real_path, self.real_y_path, self.real_path, self.real_y_path, batch_size=self.bs)
        f.write('\n\nKID_rr:  {} +/- {} \nMean +/- std KID for each class: {} +/- {}'.format(kid_mean, kid_std, kid_classes, kid_classes_std))

        # SSIM
        ssim_mean, ssim_std, ssim_classes, ssim_classes_std = self.SSIM(self.real_path, self.real_y_path, self.real_path, self.real_y_path)
        f.write('\n\nSSIM_rr:  {} +/- {} \nMean +/- std SSIM for each class: {} +/- {}'.format(ssim_mean, ssim_std, ssim_classes, ssim_classes_std))


        f.write('\n\n'+'-'*100)
        # ----------------------------------------------------------------------------------------------------
        # Metrics between real and fake images:
        # ----------------------------------------------------------------------------------------------------
        f.write('\n\nMetrics between real and fake images:')

        # FID
        fid_score_rf = self.FID(self.real_path, self.fake_path)
        f.write('\n\nFID_rf:  {}'.format(fid_score_rf))

        # KID
        kid_mean_rf, kid_std, kid_classes, kid_classes_std = self.KID(self.real_path, self.real_y_path, self.fake_path, self.fake_y_path, batch_size=self.bs)
        f.write('\n\nKID_rf:  {} +/- {} \nMean +/- std KID for each class: {} +/- {}'.format(kid_mean_rf, kid_std, kid_classes, kid_classes_std))

        # SSIM
        ssim_mean_rf, ssim_std, ssim_classes, ssim_classes_std = self.SSIM(self.real_path, self.real_y_path, self.fake_path, self.fake_y_path)
        f.write('\n\nSSIM_rf:  {} +/- {} \nMean +/- std SSIM for each class: {} +/- {}'.format(ssim_mean_rf, ssim_std, ssim_classes, ssim_classes_std))

        f.write('\n\n'+'-'*100)

        # ----------------------------------------------------------------------------------------------------
        # Metrics between fake and fake images:
        # ----------------------------------------------------------------------------------------------------
        f.write('\n\nMetrics between fake and fake images:')

        # FID
        fid_score = self.FID(self.fake_path, self.fake_path)
        f.write('\n\nFID_ff:  {}'.format(fid_score))

        # KID
        kid_mean, kid_std, kid_classes, kid_classes_std = self.KID(self.fake_path, self.fake_y_path, self.fake_path, self.fake_y_path, batch_size=self.bs)
        f.write('\n\nKID_ff:  {} +/- {} \nMean +/- std KID for each class: {} +/- {}'.format(kid_mean, kid_std, kid_classes, kid_classes_std))

        # SSIM
        ssim_mean, ssim_std, ssim_classes, ssim_classes_std = self.SSIM(self.fake_path, self.fake_y_path, self.fake_path, self.fake_y_path)
        f.write('\n\nSSIM_ff:  {} +/- {} \nMean +/- std SSIM for each class: {} +/- {}'.format(ssim_mean, ssim_std, ssim_classes, ssim_classes_std))
        f.close()

        print(f'FID: {fid_score_rf:.4f}, KID: {kid_mean_rf:.4f}, SSIM: {ssim_mean_rf:.4f}', flush=True)
        
# real_path = "data/ADNI/ADNI_test/Real256x256.nii.gz"
# fake_path_BestModel = '/home/matteolai/diciotti/GANs-for-brain-MRI-synthesis/Results/3_512_4_[1, 1, 1, 1, 1, 2, 2]_seed8/saved_images_test/PACGAN_256x256/Generated256x256.nii.gz'
# real_y_path = "data/ADNI/ADNI_test/labels.csv"
# fake_y_path = '/home/matteolai/diciotti/GANs-for-brain-MRI-synthesis/Results/3_512_4_[1, 1, 1, 1, 1, 2, 2]_seed8/saved_images_test/PACGAN_256x256/labels_syn.csv'
# img_channels = 1
# class_size = 2
# output_txt = '/home/matteolai/diciotti/GANs-for-brain-MRI-synthesis/Results/3_512_4_[1, 1, 1, 1, 1, 2, 2]_seed8/saved_images_test/PACGAN_256x256/metrics_newtest.txt'
# device = torch.device('cuda')
# Quantitative_metrics(real_path, fake_path_BestModel, real_y_path, fake_y_path, img_channels, class_size, output_txt, device, 50).compute()