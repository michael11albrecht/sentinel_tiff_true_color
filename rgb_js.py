import numpy as np
from PIL import Image

class rgbAdjustment():
    def __init__(self):
        # Constants used for image adjustments
        self.max_reflectance = 3.0
        self.midR = 0.13
        self.saturation = 1.3
        self.gamma = 2.3
        self.rayleigh = {'r': 0.013, 'g': 0.024, 'b': 0.041}
        self.gOff = 0.01
        self.gOffPow = self.gOff ** self.gamma
        self.gOffRange = (1 + self.gOff) ** self.gamma - self.gOffPow

    def adjust_reflectance(self, reflectance, midR, max_reflectance):
        reflectance = np.clip(reflectance / max_reflectance, 0, 1)
        return reflectance * (reflectance * (midR / max_reflectance + 1 - midR) - 1) / (reflectance * (2 * midR / max_reflectance - 1) - midR / max_reflectance)

    def adjust_gamma(self, reflectance):
        return ((reflectance + self.gOff) ** self.gamma - self.gOffPow) / self.gOffRange

    def saturation_enhancement(self, r, g, b):
        avg_saturation = (r + g + b) / 3.0 * (1 - self.saturation)
        r_sat = np.clip(avg_saturation + r * self.saturation, 0, 1)
        g_sat = np.clip(avg_saturation + g * self.saturation, 0, 1)
        b_sat = np.clip(avg_saturation + b * self.saturation, 0, 1)
        return r_sat, g_sat, b_sat

    def to_sRGB(self, channel):
        # Ensure channel is a numpy array
        channel = np.asarray(channel)

        # Vectorized operations for the whole array
        return np.where(channel <= 0.0031308,
                        12.92 * channel,
                        1.055 * np.power(channel, 1 / 2.4) - 0.055)

    def evaluate_pixel(self, b04, b03, b02):
        r_lin = self.adjust_gamma(self.adjust_reflectance(b04 - self.rayleigh['r'], self.midR, self.max_reflectance))
        g_lin = self.adjust_gamma(self.adjust_reflectance(b03 - self.rayleigh['g'], self.midR, self.max_reflectance))
        b_lin = self.adjust_gamma(self.adjust_reflectance(b02 - self.rayleigh['b'], self.midR, self.max_reflectance))
        r_sat, g_sat, b_sat = self.saturation_enhancement(r_lin, g_lin, b_lin)
        return Image.fromarray((np.dstack((self.to_sRGB(r_sat), self.to_sRGB(g_sat), self.to_sRGB(b_sat)))*255).astype(np.uint8))