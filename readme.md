# Flexible Job Shop Scheduling Solver

This project implements a solver for the Flexible Job Shop Scheduling Problem (FJSP) using various meta-heuristic algorithms and dispatching rules. It allows scheduling of operations across multiple machines while trying to minimize the overall makespan.

## Project Description

The Flexible Job Shop Scheduling Problem involves assigning operations of different jobs to appropriate machines and determining the sequence of operations to minimize the total completion time (makespan). This implementation includes:

- **Dispatching Rules**: SPT (Shortest Processing Time), LPT (Longest Processing Time), MWR (Most Work Remaining), LWR (Least Work Remaining)
- **Meta-heuristics**: Simulated Annealing (SA), Hill Climbing (HC), Tabu Search (TS), Genetic Algorithm (GA), Iterated Local Search (ILS)
- **Visualization**: Gantt charts for schedule visualization
- **Performance Analysis**: Tools to compare different algorithms

## Installation

### Prerequisites
- Python 3.10+
- Required packages:
  - pandas
  - plotly
  - matplotlib

### Install Dependencies
```bash
pip install pandas plotly matplotlib
```

## Usage

### Configuration
The behavior of the meta-heuristic algorithms can be configured in the `config.json` file.

### Running the Scheduler
There are two main scripts:

1. To visualize schedules using Gantt charts:
```bash
python main_schedule.py
```

2. To compare algorithm performance:
```bash
python main_compare.py
```

### Input Data Format
The program reads job data from a file named `dataset_github.txt` with the following format(https://github.com/SchedulingLab/fjsp-instances):
```
<number of jobs> <number of machines>
<number of operations> <number of machines for operation 1> <machine> <processing time> ... <number of machines for operation n> <machine> <processing time> ...
...
```

### Selecting Algorithms
Update the file named `schedule_algorithms.txt` or `compare_algorithms.txt` with the algorithms you want to run, each separated by a space:
```
SPT LPT MWR LWR SA HC TS GA ILS
```

## Project Structure

- `models.py`: Contains the core data structures (Job, Task, Machine)
- `scheduler.py`: Implements the scheduling algorithms
- `utils.py`: Contains utility functions for visualization
- `config_loader.py`: Loads configuration from JSON
- `configModels.py`: Type definitions for configuration
- `main_schedule.py`: Entry point for schedule visualization
- `main_compare.py`: Entry point for algorithm comparison

## Results

Results are saved in the `results/` directory when comparing algorithms.

## Example

After running `main_schedule.py`, you'll see Gantt charts showing the schedule produced by each algorithm, along with the makespan and