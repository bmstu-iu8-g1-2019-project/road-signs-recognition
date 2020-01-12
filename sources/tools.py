import os
import numpy as np
from keras.preprocessing.image import load_img


def versionize(versions_dir,
               root,
               insides=('configs', 'weights')):
    versions = os.listdir(versions_dir)
    if not versions:
        version = 0
    else:
        version = max([int(ver[len(root) + 2:]) for ver in versions]) + 1
    version_path = versions_dir + root + '_v' + str(version) + '/'
    os.mkdir(version_path)
    dirs = []
    for name in insides:
        dirs.append(version_path + name + '/')
        os.mkdir(dirs[-1])
    return version_path, dirs


if __name__ == '__main__':
    img = load_img('../dataset/validation/0000000.jpg', target_size=(1280, 720))
    img.show()