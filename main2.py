import copy  # Used to create deep copies of objects to avoid modifying originals
import os    # Not used in this code, but kept for potential file system operations
from typing import List, Tuple, Optional, Dict  # For type hints to clarify data types
import random  # For random number generation and shuffling in SA
import math    # For the exponential function in SA acceptance probability

# Task class represents an individual operation that belongs to a job
class Task:
    def __init__(self, machine_id: int, duration: int):
        self.machine_id: int = machine_id  # ID of the machine required to process this task
        self.duration: int = duration      # Time required to process this task
        self.start_time: int = None        # When the task starts (set during scheduling)
        self.end_time: int = None          # When the task ends (set during scheduling)

    def display_task(self) -> None:
        print(f'[{self.machine_id}, {self.duration}]', end = '')

# Machine class represents a single machine that can process tasks sequentially
class Machine:
    def __init__(self, machine_id: int):
        self.machine_id: int = machine_id                           # Unique ID for the machine
        self.schedule: List[Tuple[int, int, int]] = []              # List of tuples (job_id, start_time, end_time)

    def add_to_schedule(self,  job_id: int, start_time: int, end_time: int) -> None:
        # Adds a task to the machine's schedule
        self.schedule.append((job_id, start_time, end_time))
    

# Job class represents a sequence of tasks that must be completed in order
class Job:
    def __init__(self, job_id: int, tasks: List[List[Task]]):       #the list objects are class Task
        self.job_id: int = job_id                       # Unique ID for the job
        self.tasks:List[List[Task]] = tasks             # List of Task objects for this job
        self.current_task_index: int = 0                # Tracks progress of the job (which task is next)

    def get_next_task(self) -> Optional[List[Task]]:
        # Returns the next task to be scheduled, or None if the job is complete
        if self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index] #returns a task which has machine, duration
        return None

    def complete_task(self) -> None:
        # Moves to the next task in the job sequence
        self.current_task_index += 1

    def is_complete(self) -> bool:
        # Checks if all tasks in the job have been completed
        return self.current_task_index >= len(self.tasks)
    
# Scheduler class manages the entire scheduling process
class Scheduler:
    def __init__(self, jobs: List[Job], machines: List[Machine]):
        self.jobs: List[Job] = copy.deepcopy(jobs)            # List of Job objects
        self.machines: List[Machine] = copy.deepcopy(machines)    # List of Machine objects
        self.global_max: int = 0
        self.work_remaining: Dict[int, List[int]] = {}      #will have the job id as a key and an array which will repesent the work remaining for each position
                                                            #ex: for [2 1 2 3 2 1] [0 3] [1 5] [1 6 1 3] will be [13 ,11 ,8, 4, 3, 0] (we take the min from task list) 

        #here we will compute the reverse pref sums for work remaining

        for job in self.jobs:
        #we initialise the keys first
            self.work_remaining[job.job_id] = [0] #this is goint to be the 0 at the end
            sum: int = 0
            for task_list in reversed(job.tasks):
                mn = task_list[0].duration
                for task in task_list:
                    mn = min(mn, task.duration)
                sum += mn
                self.work_remaining[job.job_id].append(sum)
            
            #we reverse the pref array
            self.work_remaining[job.job_id].reverse()


    def shortest_processing_time(self) -> Optional[Tuple[Job, Task]]:
        next_task_touple: Optional[Tuple[Job, Task]] = None       
        for job in self.jobs:                           # We go throught jobs                 
            if (not job.is_complete()):                 # Check if it's completed (still has tasks)
                curr_task_list: Optional[List[Task]] = job.get_next_task()    # Get it's next task list 
                if (curr_task_list is not None):        # Check if it's not empty
                    if(next_task_touple == None):              
                        next_task_touple = (job, curr_task_list[0]) #format: (job, task)
                    for task in curr_task_list:         # Check the tasks from the current
                        if(task.duration < next_task_touple[1].duration):
                            next_task_touple = (job, task)
        
        return next_task_touple    # format: (job, task)
    
    def longest_processing_time(self) -> Optional[Tuple[Job, Task]]:
        next_task_touple = None       
        for job in self.jobs:                           # We go throught jobs                 
            if (not job.is_complete()):                 # Check if it's completed (still has tasks)
                curr_task_list: Optional[List[Task]] = job.get_next_task()    # Get it's next task list 
                if (curr_task_list is not None):        # Check if it's not empty
                    if(next_task_touple == None):              
                        next_task_touple = (job, curr_task_list[0]) #format: (job, task)
                    for task in curr_task_list:         # Check the tasks from the current
                        if(task.duration > next_task_touple[1].duration):
                            next_task_touple = (job, task)
        
        return next_task_touple    # format: (job, task)
    
    # Most work remaining with picking to execute the task on the machine with least execution time
    def most_work_remaining(self) -> Optional[Tuple[Job, Task]]:
        next_task_touple = None             # Will hold the best (job, task) pair
        next_task_list = None               # Will keep the task list from where we're gonna pick the task
        next_task = None                    # Will keep the next task that will be returned
        next_job = None                     # Will keep the job that will be returned
        mx = None                           # Tracks the current maximum remaining work
        for job in self.jobs:
            if(not job.is_complete()):
                work_remaining: int = self.work_remaining[job.job_id][job.current_task_index]   # keeps the remaining work
                if(mx == None or work_remaining > mx):                                          # if mx is not initialized or we find a new max then we update
                    mx = self.work_remaining[job.job_id][job.current_task_index]
                    next_task_list = job.get_next_task()
                    next_job = job

        # Here we pick the task with min duration from the task list
        next_task: Task = next_task_list[0]             # initialize the task with first task in the task list
        for task in next_task_list:
            if(task.duration < next_task.duration):
                next_task = task
        # Here we create the return tuple
        next_task_touple = (next_job, next_task)

        return next_task_touple
    
    # Least work remaining with picking to execute the task on the machine with least execution time
    def least_work_remaining(self):
        next_task_touple = None             # Will hold the best (job, task) pair
        next_task_list = None               # Will keep the task list from where we're gonna pick the task
        next_task = None                    # Will keep the next task that will be returned
        next_job = None                     # Will keep the job that will be returned
        mn = None                           # Tracks the current minimum remaining work
        for job in self.jobs:
            if(not job.is_complete()):
                work_remaining: int = self.work_remaining[job.job_id][job.current_task_index]   # keeps the remaining work
                if(mn == None or work_remaining < mn):                                          # if mx is not initialized or we find a new min then we update
                    mn = self.work_remaining[job.job_id][job.current_task_index]
                    next_task_list = job.get_next_task()
                    next_job = job

        # Here we pick the task with min duration from the task list
        next_task: Task = next_task_list[0]             # initialize the task with first task in the task list
        for task in next_task_list:
            if(task.duration < next_task.duration):
                next_task = task
        # Here we create the return tuple
        next_task_touple = (next_job, next_task)

        return next_task_touple

    def run(self, heuristic: str) -> None:
        # Main scheduling loop - continues until all jobs are complete
        while any(not job.is_complete() for job in self.jobs): #do until all the jobs are completed
            # Get all available tasks sorted by the chosen heuristic
            if(heuristic == 'SPT'):
                next_task_tuple: Optional[Tuple[Job, Task]] = self.shortest_processing_time()   # format: (job, task)
            if(heuristic == 'LPT'):
                next_task_tuple: Optional[Tuple[Job, Task]] = self.longest_processing_time()    # format: (job, task)   
            if(heuristic == 'MWR'):
                next_task_tuple: Optional[Tuple[Job, Task]] = self.most_work_remaining()        # format: (job, task)
            if(heuristic == 'LWR'):
                next_task_tuple: Optional[Tuple[Job, Task]] = self.least_work_remaining()       # format: (job, task)    

            #print(f'Job: {next_task_tuple[0].job_id},  Machine: {next_task_tuple[1].machine_id}, {next_task_tuple[1].duration}')
            
            if(len(self.machines[next_task_tuple[1].machine_id].schedule) > 0):
                curr = self.machines[next_task_tuple[1].machine_id].schedule[-1] # format: (job_id, start_time, end_time)
                self.machines[next_task_tuple[1].machine_id].add_to_schedule(next_task_tuple[0].job_id, curr[2], curr[2] + next_task_tuple[1].duration)
                next_task_tuple[1].start_time = curr[2]
                next_task_tuple[1].end_time = curr[2] + next_task_tuple[1].duration
                self.global_max = max(self.global_max, next_task_tuple[1].end_time)

            else:
                self.machines[next_task_tuple[1].machine_id].add_to_schedule(next_task_tuple[0].job_id, 0, next_task_tuple[1].duration)
                next_task_tuple[1].start_time = 0
                next_task_tuple[1].end_time = next_task_tuple[1].duration
                self.global_max = next_task_tuple[1].end_time

            
            self.jobs[next_task_tuple[0].job_id].complete_task()

            
    def print_machine_answer(self):
        for machine in self.machines:
            print(f'Machine id: {machine.machine_id}')
            for tuple in machine.schedule:
                print(f'Job: {tuple[0]}: {tuple[0]} -> {tuple[1]}')
            print()

    def print_job_answer(self):
        for job in self.jobs:
            print(f'Job id: {job.job_id}')
            for task_list in job.tasks:
                for task in task_list:
                    if(task.start_time != None):
                        print(f'Machine {task.machine_id}:  {task.start_time} -> {task.end_time}')
            print()

    def reset_scheduler(self):
        #reset the timespan
        self.global_max = 0

        #reset machines
        for machine in self.machines:
            machine.schedule.clear()

        #reset the jobs
        for job in self.jobs:
            job.current_task_index = 0
            for task_list in job.tasks:
                for task in task_list:
                    task.start_time = None
                    task.end_time = None

    def get_makespan(self) -> int:
        return self.global_max
    
    def display_work_remaining_arrays(self):
        for key, value in self.work_remaining.items():
            print(f'Job_Id: {key}')
            print(value)
            print()


#read from file
allJobs = []    #will contain all the jobs with their tasks
jobsNr = 0
machinesNr = 0
ind = 0         #Will keep the job Id nr

with open('dataset2.txt', 'r') as file:
    #read by lines
    first = True
    for line in file:
        allTasks = []# will keep all the tasks for a job before adding it to allJobs
        elements = line.split()
        if first: #we read the size of the problem JobsNr, MachinesNr
            jobsNr, machinesNr = int(elements[0]), int(elements[1])
            first = False
        else: #here we read the tasks for each job and create the jobs
            isMachine = True
            isDuration = False
            add = []                                        #will keep the task list to be added
            currTaskNumbers = []                            #will keep the current task's [machine, duration]
            for element in elements:                        #we go through the elements of the job separated by space as strings
                if(isMachine):          
                    if(element[0] == '['):                  #if it's the first one we remove the bracket
                        currTaskNumbers.append(int(element[1:]))
                    else:
                        currTaskNumbers.append(int(element))
                    #We swap the element's type processing
                    isMachine = False
                    isDuration = True
                else:
                    if(element[len(element) - 1] == ']'):
                        currTaskNumbers.append(int(element[:-1]))
                        add.append(Task(currTaskNumbers[0], currTaskNumbers[1]))
                        currTaskNumbers.clear()
                        allTasks.append(copy.deepcopy(add))
                        add.clear()
                    else:
                        currTaskNumbers.append(int(element))
                        add.append(Task(currTaskNumbers[0], currTaskNumbers[1]))
                        currTaskNumbers.clear()
                    #We swap the element's type processing
                    isMachine = True
                    isDuration = False
            
            allJobs.append(Job(ind, allTasks))
            ind += 1

# Define machines
machines = []
for i in range(machinesNr):
    machines.append(Machine(i))

scheduler = Scheduler(allJobs, machines)
scheduler.run('SPT')
scheduler.print_job_answer()
print(f'The total time is: {scheduler.get_makespan()}')
scheduler.reset_scheduler()

print()

scheduler.run('LPT')
scheduler.print_job_answer()
print(f'The total time is: {scheduler.get_makespan()}')
scheduler.reset_scheduler()

print()

scheduler.run('MWR')
scheduler.print_job_answer()
print(f'The total time is: {scheduler.get_makespan()}')
scheduler.reset_scheduler()

print()

scheduler.run('LWR')
scheduler.print_job_answer()
print(f'The total time is: {scheduler.get_makespan()}')
scheduler.reset_scheduler()

print()

scheduler.display_work_remaining_arrays()

# for curr_job in scheduler.jobs:
#     for curr_task_set in curr_job.tasks:
#         for curr_task in curr_task_set:
#             curr_task.display_task()
#         print()
#     print()









# [acelasi task pentru diferite masini]

# planificarea trebuie reprezentata prin una sau mai multe liste

# lista de indici de job-uri



# [1 2 4 1 3 2 2 4 1] list of job ids, first apparition is first task in the job

# -am o problema cu 4 job-uri, job-ul 1 din 3 operatii, 
# -opartia 1 job 1
# -operatia 2 job 2

# -3 opratii pt job 1, pt ca de 3 ori apare 1

# [5 3 1 2 4 3 5 1 2] the id of the machine where the task will be executed