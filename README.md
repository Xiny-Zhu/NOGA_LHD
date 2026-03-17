
# NOGA: Genetic Algorithm for Maximin and Near-Orthogonal Latin Hypercube Designs

This repository contains the implementation of the **NOGA algorithm**, a genetic algorithm designed for constructing ** Latin Hypercube Designs (LHDs)** balance space-filling properties with low factor correlation.

The implementation includes:

- Generation of **Good Lattice Points (GLPs)**
- Construction of candidate column sets
- Correlation-based filtering
- Distance-based and Correlation-based design evaluation
- A genetic algorithm framework for design optimization

The code was developed for research experiments related to LHD construction.

---

# Repository Structure

.
├── GLP.py  
├── transformation.py  
├── coef.py  
├── distance.py  
├── candidate.py  
├── ga_experiment.py  
├── run_example.py  
├── Simulation_finall.ipynb  
│  
└── README.md  

---

# Core Modules

## GLP.py

Implements the construction of **Good Lattice Points (GLPs)**.

Main function

```
get_GLP(N)
```

Generates the base GLP matrix used in candidate generation.

---

## transformation.py

Contains transformation functions applied to GLP columns.

Includes:

- Williams transformation
- Auxiliary transformations used during candidate construction.

---

## coef.py

Implements correlation calculations used for candidate filtering.

Main utilities include

- column correlation computation
- maximum correlation evaluation

---

## distance.py

Distance-based evaluation of designs.

Includes functions for computing

- L1 distance
- L2 distance
- minimum pairwise distance

These metrics are used as objective functions during optimization.

---

## candidate.py

Constructs candidate column sets used in the genetic algorithm.

Includes

- candidate set generation
- feasibility checks
- initial population generation for the genetic algorithm

---

## ga_experiment.py

Implements the main **genetic algorithm optimization process**.

Includes

- population initialization
- selection
- crossover
- mutation
- fitness evaluation

---

## run_example.py

A minimal runnable example demonstrating how to execute the algorithm.

---



# Running the Algorithm

The simplest way to run the algorithm is:

```
python run_example.py
```

This script will:

1. Generate the candidate set
2. Initialize the genetic algorithm population
3. Run the optimization
4. Output the resulting design and objective values

---

# Example Workflow

Generate GLP  
↓  
Construct candidate columns  
↓  
Filter by correlation  
↓  
Initialize population  
↓  
Run genetic algorithm  
↓  
Evaluate design quality  

---


# Reproducibility

All algorithmic components are implemented in standalone Python modules.

The algorithm can be reproduced by executing

```
python run_example.py
```

or by importing the modules in custom scripts.

---


# License

This project is released for research and academic use.

---

# Contact

For questions regarding the implementation, please open an issue in the repository.
