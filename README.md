# A CQM approach to the Chicken and Waffles Problem!
This work demonstrates the representation of the Chicken and Waffles problem as a Constrained Quadratic Model (CQM), and its solution using DWAVE Quantum Technology

## What is the Chicken and Waffles problem?
It's a version of the Knapsack problem: when you go to eat Chicken and Waffles, you will be greeted by a plethora of options to design your meal. For instance, you will have control over the kind of waffle, drizzle, chicken and other components that will comprise your meal. Each option comes with its own price, calorie count, color and many other attributes. What if you want to find out which meals you can construct that, for instance, consist of fewer than 700 calories? And what if you want to find such meals whilst keeping the price of the meal as low as possible?

Well, this is the exact problem that has been solved in this example, but there are many such constraints that can be imposed upon the menu. The repository contains the menu data in the `data` folder. Cook up your constraints and your own Chicken and Waffle problem!

## What is the CQM approach?
The problem above can be modeled as an optimization problem with constraints. That's exactly what a CQM is:
- An optimization problem<br>
*We are trying to minimize the price of the meal*
- With constraints<br>
*We can pick only one item from each category that together comprise a meal (you can't have two different kinds of waffles in the same meal, for instance), and keeping the calorie count of the meal below 700*
- Wherein the degree of each term in the objective function and constraint equations is quadratic or lower<br>
*The mathematical details can be found [here](https://docs.ocean.dwavesys.com/en/stable/concepts/cqm.html)*

## How to run the solver?

### Install docker
- Install docker:
  - https://docs.docker.com/install/
- Install docker-compose:
  - https://docs.docker.com/compose/install/

### Prepare repo
- Clone the repo and change into it

### Add your DWave API token
- Open `solver.env` in your favorite text editor
- Replace `my-dwave-api-token` with your API token
- How to find your API token online:
  - https://support.dwavesys.com/hc/en-us/articles/360003682634-How-Do-I-Get-an-API-Token-
A `DEBUG` mode is also supported that will use [simulated annealing](https://docs.ocean.dwavesys.com/en/stable/overview/cpu.html#using-cpu) instead of the QC.
However, this can be very slow to the point of infeasibility for large problem spaces or large numbers of reads.
It is best to only use this mode for a sub-set of your problem when trying to verify or debug your code.

### Build the container
```
sudo docker-compose build
```

### Start the workflow
```
sudo docker-compose up
```

### Obtain Results!
- The results will be printed to the screen
- They will also be stored in the results directory

## Main steps to solve the problem
1. Declare a CQM model. [example](https://github.com/pqb-mb/pqb-cqm-example/blob/main/cqm_solver.py#L439)
2. Define all the binary variables. [example](https://github.com/pqb-mb/pqb-cqm-example/blob/main/cqm_solver.py#L15)
3. Define the objective. [example](https://github.com/pqb-mb/pqb-cqm-example/blob/main/cqm_solver.py#L128)
4. Define the one-hot constraints. [example](https://github.com/pqb-mb/pqb-cqm-example/blob/main/cqm_solver.py#L143)
5. Define the inequality constraints. [example](https://github.com/pqb-mb/pqb-cqm-example/blob/main/cqm_solver.py#L158)
6. Submit the model to D-Wave. [example](https://github.com/pqb-mb/pqb-cqm-example/blob/main/cqm_solver.py#L267)
7. Save the results. [example](https://github.com/pqb-mb/pqb-cqm-example/blob/main/cqm_solver.py#L351)
