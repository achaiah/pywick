import cv2


def read_cv2_as_rgba(path):
    """
    Reads files from the provided path and returns them as a dictionary of: {'image': rgba, 'mask': rgba[:, :, 3]}
    :param path:        Absolute file path

    :return:        {'image': rgba, 'mask': rgba[:, :, 3]}
    """
    image = cv2.imread(path, -1)
    # By default OpenCV uses BGR color space for color images, so we need to convert the image to RGB color space.
    rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return {'image': rgba, 'mask': rgba[:, :, 3]}


def read_cv2_as_rgb(path):
    """
    Reads files from the provided path and returns them as a dictionary of: {'image': rgb} in RGB format
    :param path:        Absolute file path

    :return:        CV2 / numpy array in RGB format
    """
    image = cv2.imread(path, -1)
    # By default OpenCV uses BGR color space for color images, so we need to convert the image to RGB color space.
    return {'image': cv2.cvtColor(image, cv2.COLOR_BGR2RGB)}
