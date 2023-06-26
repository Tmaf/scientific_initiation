from typing import List

from individual import Individual
from score_calculator import ScoreCalculator
from data_access import Logger


def create_population(size: int) -> List[Individual]:
    return [Individual() for _ in range(size)]


def mutate_population(
        population: List[Individual],
        best_individual: Individual,
        worst_individual: Individual,
):
    for individual in population:
        if individual != best_individual:
            individual.mutate(best=best_individual, worst=worst_individual)


class Jaya:
    def __init__(self,
                 population_size: int,
                 generations_number: int,
                 score_calculator: ScoreCalculator,
                 logger: Logger,
                 pool):
        self.population_size = population_size
        self.generations_number = generations_number

        self.logger = logger
        self.pool = pool
        self.score_calculator = score_calculator

    def execute(self):
        population = create_population(self.population_size)
        best_results: List = []

        for generation in range(self.generations_number):
            population = self.pool.map(self.score_calculator.execute, population)
            best_individual = max(population)
            worst_individual = min(population)
            best_results.append(best_individual.score)
            self.logger.log_generation(generation, best_individual, worst_individual, population, best_results)
            mutate_population(population, best_individual, worst_individual)
        self.logger.plot_evolution_score(best_results, self.generations_number)
