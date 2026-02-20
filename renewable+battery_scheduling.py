#---------- Benders' Decomposition for Renewable + Battery Scheduling----------#
#------------(3 scenarios | 24 hours | same data as given in Tables (1,2,3,4,5) of project file------------#



import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------------------
# Sets & Scenarios
# ---------------------------------------------------------------------------------------------
T = list(range(1, 25))               # hours 1..24
S = [1, 2, 3]                        # low, medium, high demand
pi = {1: 0.30, 2: 0.40, 3: 0.30}     # scenario probabilities

# ---------------------------------------------------------------------------------------------
# Parameters (Appendix Tables) 
# ---------------------------------------------------------------------------------------------
# Day-ahead cost c_t  (NOK/MWh)
c_t = {t: v for t, v in zip(T, [
  760,740,720,710,720,770,830,900,940,920,900,880,
  870,860,860,880,930,990,1010,990,950,900,860,820
])}

# Energy Demand D_{s,t} (MWh)
D1 = [266.0,256.5,247.0,242.3,247.0,275.5,313.5,342.0,361.0,351.5,346.8,337.3,
      332.5,327.8,332.5,351.5,380.0,399.0,389.5,370.5,342.0,313.5,294.5,275.5]
D2 = [280,270,260,255,260,290,330,360,380,370,365,355,
      350,345,350,370,400,420,410,390,360,330,310,290]
D3 = [302.4,291.6,280.8,275.4,280.8,313.2,356.4,388.8,410.4,399.6,394.2,383.4,
      378.0,372.6,378.0,399.6,432.0,453.6,442.8,421.2,388.8,356.4,334.8,313.2]
D_st = {(1,t): D1[t-1] for t in T}
D_st.update({(2,t): D2[t-1] for t in T})
D_st.update({(3,t): D3[t-1] for t in T})

# Wind energy available  W_{s,t} (MWh)
W1 = [93.1,103.0,124.7,141.7,165.8,185.4,207.8,225.2,207.2,184.1,161.4,147.8,
      136.6,128.8,120.1,130.8,146.8,163.1,177.7,169.0,146.8,121.5,103.4,97.3]
W2 = [118.8,131.7,159.0,181.6,212.0,236.9,265.8,287.7,264.0,234.9,206.9,189.3,
      174.7,164.0,153.9,168.7,189.4,210.7,229.5,218.4,189.5,156.3,133.1,125.2]
W3 = [148.5,164.6,199.1,227.0,265.0,296.1,332.3,359.6,330.1,293.7,258.6,236.6,
      218.4,205.0,192.4,210.9,236.7,263.4,287.0,273.0,236.8,195.4,166.4,156.5]
W_st = {(1,t): W1[t-1] for t in T}
W_st.update({(2,t): W2[t-1] for t in T})
W_st.update({(3,t): W3[t-1] for t in T})

# Real-time market price p_{s,t} (NOK/MWh)
P1 = [837.0,818.4,799.8,790.5,799.8,837.0,911.4,967.2,1023.0,1004.4,985.8,967.2,
      950.6,930.0,930.0,950.6,1004.4,1078.8,1097.4,1078.8,1023.0,967.2,911.4,874.2]
P2 = [900,880,860,850,860,900,980,1040,1100,1080,1060,1040,
      1020,1000,1000,1020,1080,1160,1180,1160,1100,1040,980,940]
P3 = [990.0,968.0,946.0,935.0,946.0,990.0,1078.0,1144.0,1210.0,1188.0,1166.0,1144.0,
      1122.0,1100.0,1100.0,1122.0,1188.0,1276.0,1298.0,1276.0,1210.0,1144.0,1078.0,1034.0]
p_st = {(1,t): P1[t-1] for t in T}
p_st.update({(2,t): P2[t-1] for t in T})
p_st.update({(3,t): P3[t-1] for t in T})

# Market capacity per hour (MWh)
M_cap = {t: v for t, v in zip(T, [
  140,140,140,140,150,170,190,200,200,190,180,170,160,160,160,170,152,168,168,160,180,160,150,145
])}

# Battery/system parameters (Section 2.1)
E_max = 120.0
P_max_charge = 45.0
P_max_discharge = 55.0
eta_charge = 0.95
eta_discharge = 0.95
S0 = 60.0
S_min_final = 55.0
S_max_final = 85.0
PEN_UNMET = 100000.0
beta = 12.0                         # throughput cost on (q + r)
k_st = {(s,t): 320.0 for s in S for t in T}  # curtailment penalty

# Natural capacity for x_t
X_max = {t: max(D_st[(s,t)] for s in S) for t in T}

# ---------------------------------------------------------------------------------------------
# Master problem construction
# ---------------------------------------------------------------------------------------------
def build_master(Cuts):
    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=T, ordered=True)
    m.x = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=lambda m,t: (0.0, X_max[t]))
    # α >= 0 is safe because recourse costs are nonnegative
    m.alpha = pyo.Var(bounds=(0.0, 1e12))

    # Benders cuts
    m.C = pyo.Set(initialize=list(Cuts["idx"]))  # cut indices
    m.Phi    = pyo.Param(m.C, initialize=Cuts["Phi"],    mutable=True, default=0.0)
    m.Lam    = pyo.Param(m.T, m.C, initialize=lambda m,t,c: Cuts["Lam"].get((t,c), 0.0), mutable=True, default=0.0)
    m.xfix   = pyo.Param(m.T, m.C, initialize=lambda m,t,c: Cuts["xfix"].get((t,c), 0.0), mutable=True, default=0.0)

    def cut_rule(m, c):
        return m.alpha >= m.Phi[c] + sum(m.Lam[t,c]*(m.x[t] - m.xfix[t,c]) for t in m.T)
    m.BendersCuts = pyo.Constraint(m.C, rule=cut_rule)

    # Master objective
    m.obj = pyo.Objective(expr = sum(c_t[t]*m.x[t] for t in m.T) + m.alpha, sense=pyo.minimize)
    return m

# ---------------------------------------------------------------------------------------------
# Subproblem construction (per scenario) with linking constraint and duals
# ---------------------------------------------------------------------------------------------
def build_subproblem(x_fixed, s):
    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=T, ordered=True)

    # Param: x fixed from master
    m.x_fixed = pyo.Param(m.T, initialize={t: float(x_fixed[t]) for t in T})

    # Variables (scenario s)
    m.q   = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.r   = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.mkt = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.u   = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.S   = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, E_max))
    m.ls  = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    # Local copy of x to create a constraint with a dual
    m.x   = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # Limits
    m.ChargeLimit    = pyo.Constraint(m.T, rule=lambda m,t: m.q[t]   <= P_max_charge)
    m.DischargeLimit = pyo.Constraint(m.T, rule=lambda m,t: m.r[t]   <= P_max_discharge)
    m.MarketCap      = pyo.Constraint(m.T, rule=lambda m,t: m.mkt[t] <= M_cap[t])

    # State of cherge of battery dynamics
    t1, tT = T[0], T[-1]
    m.S_init = pyo.Constraint(expr = m.S[t1] == S0 + eta_charge*m.q[t1] - eta_discharge*m.r[t1])
    def soc(m,t):
        if t == t1: return pyo.Constraint.Skip
        return m.S[t] == m.S[t-1] + eta_charge*m.q[t] - eta_discharge*m.r[t]
    m.SoC = pyo.Constraint(m.T, rule=soc)

    # Terminal State of cherge of battery window
    m.S_final_low  = pyo.Constraint(expr = m.S[tT] >= S_min_final)
    m.S_final_high = pyo.Constraint(expr = m.S[tT] <= S_max_final)

    # Supply and demand balance
    def balance(m,t):
        return m.x[t] + m.r[t] + m.mkt[t] + (W_st[(s,t)] - m.u[t]) + m.ls[t] == D_st[(s,t)] + m.q[t]
    m.Balance = pyo.Constraint(m.T, rule=balance)

    # Linking: x[t] = x_fixed[t]  (we will read the dual of this constraint)
    m.Link = pyo.Constraint(m.T, rule=lambda m,t: m.x[t] == m.x_fixed[t])

    # Scenario objective (already probability-weighted)
    m.obj = pyo.Objective(
        expr = pi[s]*sum(
            p_st[(s,t)]*m.mkt[t] + k_st[(s,t)]*m.u[t] + PEN_UNMET*m.ls[t] + beta*(m.q[t] + m.r[t])
            for t in m.T
        ),
        sense = pyo.minimize
    )

    # To read duals with GLPK
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m



# ---------------------------------------------------------------------------------------------
# Solve helper
# ---------------------------------------------------------------------------------------------
def solve(m, tee=False):
    opt = SolverFactory("glpk")
    res = opt.solve(m, tee=tee, load_solutions=True)
    return m, res

# ---------------------------------------------------------------------------------------------
# Benders main loop (Steps 1–4) + convergence graph
# ---------------------------------------------------------------------------------------------
def benders_solve(max_iters=200, eps=1e-6, tee=False):
    # Cut storage
    Cuts = {"idx": [], "Phi": {}, "Lam": {}, "xfix": {}}
    UB_hist, LB_hist, it_hist = [], [], []

    for k in range(1, max_iters+1):
        # ------- Step 1 (solve initial master problem) -------
        M = build_master(Cuts)
        M, _ = solve(M, tee=False)
        x_fixed = {t: pyo.value(M.x[t]) for t in T}
        alpha   = pyo.value(M.alpha)
        LB      = pyo.value(M.obj)

        # ------- Step 2 (solve subproblems for all scenario) -------
        Sub = {}
        Qs  = {}
        for s in S:
            Sub[s] = build_subproblem(x_fixed, s)
            Sub[s], _ = solve(Sub[s], tee=False)
            Qs[s] = pyo.value(Sub[s].obj)  # already probability-weighted

        UB = sum(c_t[t]*x_fixed[t] for t in T) + sum(Qs.values())
        gap = UB - LB

        it_hist.append(k); UB_hist.append(UB); LB_hist.append(LB)
        print(f"[Iteration {k:03d}] LB = {LB:,.2f} | UB = {UB:,.2f} | Gap = {gap:,.6f}")

        # ------- Step 3 (Convergence check) -------
        if gap <= eps:
            # Save iteration history
            pd.DataFrame({"iter": it_hist, "LB": LB_hist, "UB": UB_hist, "Gap": [UB_hist[i]-LB_hist[i] for i in range(len(it_hist))]}) \
              .to_csv("benders_convergence.csv", index=False)

            # Convergence plot
            plt.figure()
            plt.plot(it_hist, UB_hist, marker='o', linestyle='-', label='Upper Bound')
            plt.plot(it_hist, LB_hist, marker='s', linestyle='-', label='Lower Bound')
            plt.xlabel('Iteration')
            plt.ylabel('Objective function values (NOK)')
            plt.title('Benders Convergence Plot')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("BendersConvergence.png", dpi=150)

            # Gather detailed outputs
            recs = []
            for s in S:
                m = Sub[s]
                for t in T:
                    recs.append({
                        "Scenario": s, "Hour": t,
                        "q": pyo.value(m.q[t]), "r": pyo.value(m.r[t]),
                        "market": pyo.value(m.mkt[t]), "curtail": pyo.value(m.u[t]),
                        "SoC": pyo.value(m.S[t]), "unmet": pyo.value(m.ls[t]),
                    })
            # Save all the optimal values in csv file
            pd.DataFrame([{"Hour": t, "x_t_MWh": x_fixed[t], "c_t": c_t[t]} for t in T]) \
              .to_csv("x_dayahead_benders.csv", index=False)
            pd.DataFrame(recs).to_csv("subproblem_results_benders.csv", index=False)

            print("\n===== Benders' decomposition algorithm has been converged!=====")
            print(f"Optimal objective value(NOK): {UB:,.2f}")
            print("Saved: x_dayahead_benders.csv, subproblem_results_benders.csv, benders_convergence.csv, BendersConvergence.png")
            return UB, x_fixed, Sub, UB_hist, LB_hist

        # ------- Step 4 (Add new Bneders' cut in the Initial Master problem and then continue with Step2) -------
        cidx = len(Cuts["idx"])
        Cuts["idx"].append(cidx)
        # Phi = sum_s Q_s(x_fixed)  (already probability-weighted)
        Cuts["Phi"][cidx] = sum(Qs.values())
        # Lambda_t = sum_s π_s * dual(Link[t] in scenario s)
        for t in T:
            lam_t = sum(pi[s] * Sub[s].dual[Sub[s].Link[t]] for s in S)
            Cuts["Lam"][(t, cidx)]  = lam_t
            Cuts["xfix"][(t, cidx)] = x_fixed[t]

    raise RuntimeError("Benders did not converge within max_iters")

# Run
if __name__ == "__main__":
    UB, xstar, Subs, UB_hist, LB_hist = benders_solve(max_iters=400, eps=1e-6, tee=False)
