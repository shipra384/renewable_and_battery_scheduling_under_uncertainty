# Two-Stage Stochastic Renewable + Battery Scheduling  
## Benders Decomposition Implementation (Pyomo)

This project implements a **two-stage stochastic optimization model** for renewable energy and battery scheduling under uncertain demand and market prices.

The model is solved using a **manual Benders decomposition algorithm**, explicitly generating optimality cuts using dual variables from scenario subproblems.

---

## Problem Overview

We consider a 24-hour scheduling horizon with:

- 3 demand scenarios (low, medium, high)
- Scenario probabilities
- Wind generation uncertainty
- Real-time electricity market participation
- Battery storage with charging/discharging efficiency
- Curtailment and unmet demand penalties

---


## Outputs

When converged, the script generates:

- `x_dayahead_benders.csv`  
  → Optimal first-stage decisions

- `subproblem_results_benders.csv`  
  → Detailed scenario results

- `benders_convergence.csv`  
  → Iteration history (UB/LB/Gap)

- `BendersConvergence.png`  
  → Convergence plot

Console output shows iteration progress
