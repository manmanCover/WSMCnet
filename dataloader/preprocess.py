import torch
import torchvision.transforms as transforms
import random

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

#__imagenet_stats = {'mean': [0.5, 0.5, 0.5],
#                   'std': [0.5, 0.5, 0.5]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def get_transform(augment=True):

    t_list = [transforms.ToTensor()]

    if augment:
        t_list_color = [
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        ]
        if(random.random() < 0.2):
            t_list_color.append(Grayscale())
        t_list.extend(t_list_color)

    normalize = __imagenet_stats
    t_list.append(transforms.Normalize(**normalize))

    return transforms.Compose(t_list)

class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
        self.order = torch.randperm(len(self.transforms))

    def __call__(self, img):
        if self.transforms is None:
            return img
        for i in self.order:
            img = self.transforms[i](img)
        return img

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec
        self.alpha = torch.zeros(3).normal_(0, self.alphastd)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = self.alpha.type_as(img)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var
        self.alpha = random.uniform(0, self.var)

    def __call__(self, img):
        gs = Grayscale()(img)
        return img.lerp(gs, self.alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var
        self.alpha = random.uniform(0, self.var)

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        return img.lerp(gs, self.alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var
        self.alpha = random.uniform(0, self.var)

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        return img.lerp(gs, self.alpha)


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])

def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])

