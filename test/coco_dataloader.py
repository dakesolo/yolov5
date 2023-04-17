import os

from torchvision.datasets import CocoDetection
import skimage.io as io
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    dataDir = 'data/cat'
    dataType = 'all'
    annFile = '{}/annotations/annotations_{}.json'.format(dataDir, dataType)

    coco = COCO(annFile)
    print(coco.dataset.keys())

    #类别信息
    catIds = coco.getCatIds(catNms=['cat'])
    catInfo = coco.loadCats(catIds)
    print(f"catIds:{catIds}")
    print(f"catcls:{catInfo}")

    #图像信息
    imgIds = coco.getImgIds(catIds=catIds)
    index = 5      #随便选择一张图
    imgInfo = coco.loadImgs(imgIds[index])[0]
    print(f"imgIds:{imgIds}")
    print(f"img:{imgInfo}")

    #标注信息
    annIds = coco.getAnnIds(imgIds=imgInfo['id'], catIds=catIds, iscrowd=None)
    annsInfo = coco.loadAnns(annIds)
    print(f"annIds:{annIds}")
    print(f"annsInfo:{annsInfo}")


    #显示图像
    # i = io.imread(imgInfo['coco_url'])
    i = cv2.imread(os.path.join(r'data/cat/images', imgInfo['file_name']))
    plt.imshow(i)
    plt.axis('off')
    coco.showAnns(annsInfo, True)
    plt.savefig('testbluelinew.jpg')
    # plt.show()