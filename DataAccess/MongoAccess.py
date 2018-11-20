from pymongo import MongoClient


class DataBase:

    def __init__(self):
        self.__db = MongoClient('localhost', 27017).database_ic

    def save_one(self, base, individual):
        self.__db[base].insert_one(individual)

    def save_many(self, base, individuals):
        self.__db[base].insert_many(individuals)

    def find_one(self, base, individual):
        self.__db[base].find_one(individual)
