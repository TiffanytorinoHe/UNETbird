import torch.utils.data as data
import os
import PIL.Image as Image
import numpy as np
import torch


# data.Dataset:

# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)


class LiverDataset(data.Dataset):

    # 创建LiverDataset类的实例时，就是在调用init初始化

    def __init__(self, root, transform=None, target_transform=None):  # root表示图片路径

        n = len(os.listdir(root))//2  # os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整

        imgs = []

        for i in range(n):

            img = os.path.join(root, "%03d.png" % i)  # os.path.join(path1[,path2[,......]]):将多个路径组合后返回

            mask = os.path.join(root, "%03d_mask.png" % i)

            imgs.append([img, mask])  # append只能有一个参数，加上[]变成一个list

        self.imgs = imgs

        self.transform = transform

        self.target_transform = target_transform

    def __getitem__(self, index):

        x_path, y_path = self.imgs[index]

        img_x = Image.open(x_path)

        img_y = Image.open(y_path)

        if self.transform is not None:

            img_x = self.transform(img_x)

        if self.target_transform is not None:

            img_y = self.target_transform(img_y)

        return img_x, img_y.squeeze(0)  # 返回的是图片

    def __len__(self):

        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]


class VOCDataset(data.Dataset):
    def __init__(self, train, transform=None, label_transform=None):
        self.train = train
        self.transform = transform
        self.label_transform = label_transform
        # 标签中RGB颜色的值
        self.voc_colors = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                           [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                           [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                           [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                           [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                           [0, 64, 128]]
        '''
        颜色对应的类别
        self.voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                            'dining table', 'dog', 'horse', 'motorbike', 'person',
                            'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
        '''
        self.color2class = np.zeros((8, 8, 8), dtype=int)
        for i, color in enumerate(self.voc_colors):
            self.color2class[int(color[0] / 32)][int(color[1] / 32)][int(color[2] / 32)] = i
        if train:
            self.imgs = np.loadtxt('./VOC2012/ImageSets/Segmentation/train.txt', dtype=str)
        else:
            self.imgs = np.loadtxt('./VOC2012/ImageSets/Segmentation/val.txt', dtype=str)

    def __getitem__(self, index):
        img_name = self.imgs[index]
        origin_img = Image.open('./VOC2012/JPEGImages/' + str(img_name) + '.jpg').convert('RGB')
        label_img = Image.open('./VOC2012/SegmentationClass/' + str(img_name) + '.png').convert('RGB')
        if self.transform is not None:
            origin_img = self.transform(origin_img)
            label_img = self.label_transform(label_img)
        label = torch.zeros(label_img.size()[1], label_img.size()[2], dtype=torch.long)
        for i in range(label.size()[0]):
            for j in range(label.size()[1]):
                class_idx = self.color2class[int(label_img[0][i][j] * 8)][int(label_img[1][i][j] * 8)][int(label_img[2][i][j] * 8)]
                label[i][j] = class_idx
        # for row in label:
        #     print(row)
        return origin_img, label

    def __len__(self):
        return self.imgs.size
