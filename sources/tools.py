import os


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
