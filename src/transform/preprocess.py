import torch
import copy
from abc import ABCMeta, abstractmethod
from .transforms import *


class Preprocess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image, anns):
        """Implementation of preprocess operation."""


class Compose(Preprocess):
    def __init__(self, preprocess_list):
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
        for p in self.preprocess_list:
            if p is None:
                continue
            args = p(*args)

        return args


class RandomApply(Preprocess):
    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability

    def __call__(self, image, anns):
        if float(torch.rand(1).item()) > self.probability:
            return image, anns
        return self.transform(image, anns)


class CropBox(Preprocess):
    def __init__(self, size=224, padding_mode='pad_resize', jitter_ratio=None):
        self.size = size
        self.padding_mode = padding_mode
        self.jitter_ratio = jitter_ratio

    def __call__(self, image, anns):
        bbox = anns['bbox']
        if self.jitter_ratio is not None:
            bbox = jitter_bbox(image, bbox, 'enlarge', self.jitter_ratio)
            bbox = squarify(bbox, 1, image.size[0])
        bbox = list(map(int, bbox[0:4]))
        img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        img = img_pad(img, mode=self.padding_mode, size=self.size)

        return img, anns


class ImageTransform(Preprocess):
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, image, anns):
        image = self.image_transform(image)

        return image, anns


class DownSizeFrame(Preprocess):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image, anns):
        source = anns['source']
        bbox = copy.deepcopy(anns['bbox'])
        if source in ['JAAD', 'PIE']:
            image, anns['bbox'], anns['bbox_ped'] = crop_and_rescale(image, bbox, 1 / 3, self.width, self.height)
        elif source == 'TITAN':
            image, anns['bbox'], anns['bbox_ped'] = crop_and_rescale(image, bbox, 1 / 5, self.width, self.height)

        return image, anns


class RandomHflip(Preprocess):
    def __init__(self, p=0.5):
        self.probability = p

    def __call__(self, image, anns):
        bbox = copy.deepcopy(anns['bbox'])
        image, anns['bbox'] = random_flip(image, bbox, self.probability)

        return image, anns
        

class JitterBox(Preprocess):
    """
    Change the size of bounding box without processing image.
    """
    def __init__(self, jitter_ratio=1.0, squarify=True):
        self.jitter_ratio = jitter_ratio
        self.squarify = squarify
    def __call__(self, image, anns):
        bbox = anns['bbox']
        bbox = jitter_bbox(image, bbox, 'enlarge', self.jitter_ratio)
        if self.squarify:
            bbox = squarify(bbox, 1, image.size[0])
        anns['bbox'] = list(map(int, bbox[0:4]))

        return image, anns

