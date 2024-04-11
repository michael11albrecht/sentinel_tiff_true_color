import rasterio
import numpy as np
from PIL import Image, ImageEnhance
import os
from tqdm import tqdm
from rgb_js import rgbAdjustment

class Tiff2Rgb:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

    def extract(self):
        with rasterio.open(self.input_file) as src0:
            band = src0.read()
        bands_np = np.array(band)
        return bands_np[3], bands_np[2], bands_np[1] # RGB
    
    def brighten(self, band):
        alpha=0.07
        beta=5
        return np.clip(alpha*band+beta, 0,255)
    
    def normalize(self, band):
        band_min, band_max = (band.min(), band.max())
        return ((band-band_min)/((band_max - band_min)))
    
    def gammacorr(self, band):
        gamma=2.5
        return np.power(band, 1/gamma)
    
    def br_manual(self, bands):
        e_bands = []
        for band in bands:
            band = self.brighten(band)
            band = self.normalize(band)
            e_bands.append(band)
        rgb = np.dstack((e_bands[0], e_bands[1], e_bands[2]))
        return Image.fromarray((rgb * 255).astype(np.uint8))
    
    def gamma(self, bands):
        e_bands = []
        for band in bands:
            band = self.gammacorr(band)
            band = self.normalize(band)
            e_bands.append(band)
        rgb = np.dstack((e_bands[0], e_bands[1], e_bands[2]))
        return Image.fromarray((rgb * 255).astype(np.uint8))
    
    def br_pil(self, bands):
        e_bands = []
        for band in bands:
            band = self.normalize(band)
            e_bands.append(band)
        rgb = np.dstack((e_bands[0], e_bands[1], e_bands[2]))
        img = Image.fromarray((rgb * 255).astype(np.uint8))
        brighter = ImageEnhance.Brightness(img)
        brighter = brighter.enhance(5)
        brighter = ImageEnhance.Color(brighter)
        brighter = brighter.enhance(0.5)
        return brighter
    
    def save(self, img, prefix=''):
        os.makedirs(self.output_dir, exist_ok=True)
        filename = self.input_file.split('/')[-1].split('.')[0:-1]
        filename = '.'.join(filename)
        img.save(f'{self.output_dir}/{prefix}_{filename}'+'.png')

if __name__ == '__main__':
    input_dir = 'data/tiff'
    output_dir = 'data/png'
    for tif in tqdm(os.listdir(input_dir), desc='Processing'):
        if tif.endswith('.tiff') or tif.endswith('.tif'):
            tiff2rgb = Tiff2Rgb(f'{input_dir}/{tif}', output_dir)
            bands = tiff2rgb.extract()
            '''img = tiff2rgb.br_manual(bands)
            tiff2rgb.save(img,'m')
            img = tiff2rgb.gamma(bands)
            tiff2rgb.save(img,'g')
            img = tiff2rgb.br_pil(bands)
            tiff2rgb.save(img,'p')'''
            img = rgbAdjustment().evaluate_pixel(bands[0]/10000, bands[1]/10000, bands[2]/10000)
            tiff2rgb.save(img,'r')
