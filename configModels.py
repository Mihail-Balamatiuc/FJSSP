# Define classes to provide type hints for the config structure
class SimulatedAnnealingConfig:
    initial_temperature: float
    cooling_rate: float
    min_temperature: float
    max_iterations: int

class HillClimbingConfig:
    max_iterations: int
    improvement_tries: int

class TabuSearchConfig:
    tabu_tenure: int
    max_iterations: int

class GeneticAlgorithmConfig:
    population_size: int
    num_generations: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int

class Config:
    simulated_annealing: SimulatedAnnealingConfig
    hill_climbing: HillClimbingConfig
    tabu_search: TabuSearchConfig
    genetic_algorithm: GeneticAlgorithmConfig