import os
import cv2


class ImageLoader:

    def __init__(self, directories_path, images_number, type_file='.tif'):
        self.databases = dict()
        self.path = directories_path
        self.type = type_file
        self.images_number = images_number
        self.__get_databases()

    def __get_databases(self):
        for directory in self.path:
            self.databases[directory] = []
            for file in os.listdir(directory):
                if file.endswith(self.type):
                    self.databases[directory].append(str(os.path.join(directory, file)))

    def get_next(self):
        cls = 0
        for base in self.databases:
            count = 0
            for image in self.databases[base]:
                if self.images_number != -1 and count >= self.images_number:
                    break
                count += 1
                yield image, cv2.imread(image), cls
            cls += 1
