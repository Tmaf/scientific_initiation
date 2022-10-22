import os
import cv2


class Files:

    def __init__(self, directories_path, type_file='.tif'):
        self.databases = dict()
        self.path = directories_path
        self.type = type_file
        self.__get_databases()

    def __get_databases(self):
        for directory in self.path:
            self.databases[directory] = []
            for file in os.listdir(directory):
                if file.endswith(self.type):
                    self.databases[directory].append(str(os.path.join(directory, file)))

    def get_image(self, class_limit=-1):
        cls = 0
        for base in self.databases:
            count = 0
            for image in self.databases[base]:
                if class_limit != -1 and count >= class_limit:
                    break
                count += 1
                yield image, cv2.imread(image), cls
            cls += 1
