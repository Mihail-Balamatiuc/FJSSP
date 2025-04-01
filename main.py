import copy
import os
from typing import List

# Task class represents an individual operation that belongs to a job
class Task:
    def __init__(self, machine_id: int, duration: int):
        self.machine_id = machine_id  # ID of the machine required to process this task
        self.duration = duration      # Time required to process this task
        self.start_time = None        # When the task starts (set during scheduling)
        self.end_time = None          # When the task ends (set during scheduling)

# Job class represents a sequence of tasks that must be completed in order
class Job:
    def __init__(self, job_id: int, tasks: list): #the list objects are class Task
        self.job_id = job_id          # Unique ID for the job
        self.tasks = tasks            # List of Task objects for this job
        self.current_task_index = 0   # Tracks progress of the job (which task is next)

    def get_next_task(self):
        # Returns the next task to be scheduled, or None if the job is complete
        if self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index] #returns a task which has machine, duration
        return None

    def complete_task(self):
        # Moves to the next task in the job sequence
        self.current_task_index += 1

    def is_complete(self):
        # Checks if all tasks in the job have been completed
        return self.current_task_index >= len(self.tasks)

# Machine class represents a single machine that can process tasks sequentially
class Machine:
    def __init__(self, machine_id: int):
        self.machine_id = machine_id  # Unique ID for the machine
        self.schedule = []          # List of tuples (start_time, end_time, job_id)

    def add_to_schedule(self, start_time, end_time, job_id):
        # Adds a task to the machine's schedule
        self.schedule.append((start_time, end_time, job_id))

# Scheduler class manages the entire scheduling process
class Scheduler:
    def __init__(self, jobs: list[Job], machines: list[Machine]):
        self.jobs: list[Job] = copy.deepcopy(jobs)            # List of Job objects
        self.machines: list[Machine] = copy.deepcopy(machines)    # List of Machine objects

    def shortest_processing_time_heuristic(self):#it will not do always shortest task first, the shortest task that is available because they have to be don in order
        # Selects tasks based on the Shortest Processing Time (SPT) heuristic
        available_tasks = []
        for job in self.jobs:
            if not job.is_complete():
                next_task = job.get_next_task()
                if next_task is not None:
                    available_tasks.append((job, next_task))
        # Sorts tasks by their processing time (shortest first)
        available_tasks.sort(key=lambda x: x[1].duration)
        return available_tasks

    def longest_processing_time_heuristic(self):
        # Selects tasks based on the Longest Processing Time (LPT) heuristic
        available_tasks = []
        for job in self.jobs:
            if not job.is_complete():
                next_task = job.get_next_task()
                if next_task is not None:
                    available_tasks.append((job, next_task))
        # Sorts tasks by their processing time (longest first)
        available_tasks.sort(key=lambda x: x[1].duration, reverse=True)
        return available_tasks

    def job_order_heuristic(self):
        # Selects tasks based on the Job Order heuristic
        available_tasks = []
        for job in self.jobs:
            if not job.is_complete():
                next_task = job.get_next_task()
                if next_task is not None:
                    available_tasks.append((job, next_task))
        return available_tasks
    
    def most_work_remaining_heuristic(self):
        # Selects tasks based on the Most Work Remaining heuristic
        available_tasks = []
        for job in self.jobs:
            if not job.is_complete():
                next_task = job.get_next_task()
                if next_task is not None:
                    available_tasks.append((job, next_task))
        # Sorts tasks by the remaining work left in the job
        def remaining_work(job):
            remaining_duration = 0
            for i in range(job.current_task_index, len(job.tasks)):
                task = job.tasks[i]
                remaining_duration += task.duration
            return remaining_duration

        available_tasks.sort(key=lambda x: remaining_work(x[0]), reverse=True)
        return available_tasks
    
    def least_work_remaining_heuristic(self):
        # Selects tasks based on the Most Work Remaining heuristic
        available_tasks = []
        for job in self.jobs:
            if not job.is_complete():
                next_task = job.get_next_task()
                if next_task is not None:
                    available_tasks.append((job, next_task))
        # Sorts tasks by the remaining work left in the job
        def remaining_work(job):
            remaining_duration = 0
            for i in range(job.current_task_index, len(job.tasks)):
                task = job.tasks[i]
                remaining_duration += task.duration
            return remaining_duration

        available_tasks.sort(key=lambda x: remaining_work(x[0]))
        return available_tasks

    def run(self, heuristic: str):
        # Main scheduling loop - continues until all jobs are complete
        while any(not job.is_complete() for job in self.jobs): #do until all the jobs are completed
            # Get all available tasks sorted by the chosen heuristic
            if(heuristic == 'SPT'):
                available_tasks = self.shortest_processing_time_heuristic()
            if(heuristic == 'LPT'):
                available_tasks = self.longest_processing_time_heuristic()
            if(heuristic == 'JO'):
                available_tasks = self.job_order_heuristic()
            if(heuristic == 'MWR'):
                available_tasks = self.job_order_heuristic()
            if(heuristic == 'LWR'):
                available_tasks = self.least_work_remaining_heuristic()
            
            for job, task in available_tasks:
                # Get the machine required for the current task
                machine = self.machines[task.machine_id]  # Get the machine object
                
                # Find the earliest time the machine is free
                if machine.schedule:
                    machine_last_task: Task = machine.schedule[-1]  # Get the last scheduled task on this machine
                    if(job.current_task_index == 0):
                        start_time = machine_last_task[1]
                    else:
                        job_previous_task: Task = job.tasks[job.current_task_index - 1]  # Get the previous task in the job
                        if(job_previous_task.end_time > machine_last_task[1]):
                            start_time = job_previous_task.end_time
                        else:
                            start_time = machine_last_task[1]
                else:
                    if(job.current_task_index == 0):
                        start_time = 0
                    else:
                        job_previous_task: Task = job.tasks[job.current_task_index - 1]  # Get the previous task in the job
                        start_time = max(job_previous_task.end_time, 0)

                # Calculate the end time of the task
                end_time = start_time + task.duration

                # Assign the task to the machine's schedule
                machine.add_to_schedule(start_time, end_time, job.job_id)

                # Update task's start and end times
                task.start_time = start_time
                task.end_time = end_time

                # Mark the task as completed in the job
                job.complete_task()

    def print_schedule(self):
        # Prints the schedule for each machine
        for machine in self.machines:
            print(f"Machine {machine.machine_id}:")
            for start, end, job_id in machine.schedule:
                print(f"  Job {job_id}: {start} -> {end}")
            print()

    def print_job_times(self):
        # Prints the start and end times for each job
        for job in self.jobs:
            print(f"Job {job.job_id}:")
            for task in job.tasks:
                print(f"  Task's machine {task.machine_id}: {task.start_time} -> {task.end_time}")
            print()

    def print_total_time(self):
        # Prints the total time taken to complete all jobs
        total = 0
        for machine in self.machines:
            if(len(machine.schedule)):
                total = max(total, machine.schedule[len(machine.schedule) - 1][1])
        print(f"Total time to complete all jobs: {total}")

    def get_total_time(self):
        # Returns the total time taken to complete all jobs
        total = 0
        for machine in self.machines:
            if(len(machine.schedule)):
                total = max(total, machine.schedule[len(machine.schedule) - 1][1])
        return total
    
    def reset_all(self):
        #we will reset the results, clean the machines and reseting job indexes and tasks times
        for machine in self.machines:
            machine.schedule.clear()
        
        for job in self.jobs:
            for task in job.tasks:
                task.start_time = None
                task.end_time = None    
            job.current_task_index = 0


#read from file
allJobs = []#will contain all the jobs with their tasks
jobsNr = 0
machinesNr = 0
ind = 0

with open('dataset.txt', 'r') as file:
    #read by lines
    first = True
    for line in file:
        allTasks = []# will keep all the tasks for a job before adding it to allJobs
        numbers = line.split()
        if first: #we read the size of the problem JobsNr, MachinesNr
            jobsNr, machinesNr = int(numbers[0]), int(numbers[1])
            first = False
        else: #here we read the tasks for each job and create the jobs
            odd = True #will determine if we're in odd position or not (task's machine or task's duration)
            add = []   #will keep a task at a time, [task's machine, task's duration], when we reach task's duration we push to all tasks and clear add
            for number in numbers:
                if(odd):
                    add.append(int(number))
                    odd = False
                else:
                    add.append(int(number))
                    allTasks.append(Task(add[0], add[1]))
                    add = []
                    odd = True
            allJobs.append(Job(ind, allTasks))
            ind += 1


# Define machines
machines = []
for i in range(machinesNr):
    machines.append(Machine(i))

# Create the scheduler with jobs, machines
scheduler = Scheduler(allJobs, machines)#SPT, LPT, JO, MWR, LWR

# Run the scheduling process
scheduler.run("SPT")
scheduler.print_total_time()
scheduler.reset_all()
scheduler.run("LPT")
scheduler.print_total_time()
scheduler.reset_all()
scheduler.run("JO")
scheduler.print_total_time()
scheduler.reset_all()
scheduler.run("MWR")
scheduler.print_total_time()
scheduler.reset_all()
scheduler.run("LWR")
scheduler.print_total_time()
scheduler.print_schedule()
scheduler.reset_all()

# simultated anealing
# algoritmi evolutivi

#######################################################
# The steps and idea of how the whole process works

# 1. Read the dataset from a file with the following format:
#
#   - The first line contains two integers: the number of jobs and the number of machines
#   After each line represents a job with the next format
#   job1 task1's machine, job1 task1's duration, ..., job1 taskN's machine, job1 taskN's duration
#   .
#   .
#   .
#   jobsN task1's machine, jobN task1's duration, ..., jobN taskN's machine, jobN taskN's duration

# The logic steps of creating and reading:
#
#   -We get the tasks for each job(each line), and add them to the job's task list
#   -After we finished the line we create a job using the line index and task array and then push it to all jobs array
#   -After we iterate until the number of machines and create the machines
#   -Finally we create the scheduler with all the jobs, machines and heuristic type 

# The logic when we use scheduler.run()
#
#   -The while loop will run untill all the jobs are completed
#   -The scheduler will get the tasks based on the heuristics (see the heuristics logic in the functions comments)
#       available_tasks: List[Tuple[Job, Task]] (this is the structure)

# 2. Create the jobs and the tasks for each job
# 3. Create the machines
# 4. Create the scheduler
# 5. Run the scheduling process
# 6. Print the final schedule for each machine
# 7. Print the start and end times for each job
# 8. Print the total time taken to complete all jobs
# 9. Return the total time taken to complete all jobs
