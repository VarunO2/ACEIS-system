

import numpy as np
from ortools.linear_solver import pywraplp

"""
Goal:
- Reduce peak energy demand
- Reduce electricity cost
- Optimize HVAC usage
- Shift flexible loads
"""

# --------------------------------------------------
# MAIN OPTIMIZATION FUNCTION
# --------------------------------------------------

def optimize_energy(
    predicted_load,
    solar_output=None,
    battery_capacity=200,
    battery_initial=100
):
    predicted_load = np.array(predicted_load)
    # ----------------------------------------------
    # DEFAULT SOLAR IF NONE
    # ----------------------------------------------

    if solar_output is None:

        solar_output = np.array(
            [0]*6 +
            [20,40,60,80,100,120,120,100,80,60,40,20] +
            [0]*6
        )

    # ----------------------------------------------
    # ELECTRICITY TARIFF
    # ----------------------------------------------

    tariff = np.array(
        [8]*12 + [18]*4 + [6]*8
    )

    # ----------------------------------------------
    # CREATE SOLVER
    # ----------------------------------------------

    solver = pywraplp.Solver.CreateSolver('SCIP')

    n = len(predicted_load)

    # ----------------------------------------------
    # VARIABLES
    # ----------------------------------------------

    grid = [
        solver.NumVar(0, 10000, f'grid_{i}')
        for i in range(n)
    ]

    hvac = [
        solver.NumVar(
            predicted_load[i] * 0.7,
            predicted_load[i],
            f'hvac_{i}'
        )
        for i in range(n)
    ]

    battery_use = [
        solver.NumVar(0, battery_capacity, f'battery_{i}')
        for i in range(n)
    ]

    # Peak variable
    peak = solver.NumVar(0, 10000, 'peak')

    # ----------------------------------------------
    # CONSTRAINTS
    # ----------------------------------------------

    battery_level = battery_initial

    for i in range(n):

        # Energy balance
        solver.Add(
            grid[i] +
            battery_use[i] +
            solar_output[i]
            >= hvac[i]
        )

        # Peak control
        solver.Add(grid[i] <= peak)

        # Battery constraint
        solver.Add(
            battery_use[i] <= battery_level
        )

    # ----------------------------------------------
    # OBJECTIVE FUNCTION
    # ----------------------------------------------

    total_cost = solver.Sum([
        grid[i] * tariff[i]
        for i in range(n)
    ])

    # Multi-objective:
    # minimize cost + peak demand

    solver.Minimize(
        total_cost + peak * 50
    )

    # ----------------------------------------------
    # SOLVE
    # ----------------------------------------------

    status = solver.Solve()

    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    if status == pywraplp.Solver.OPTIMAL:

        optimized_grid = np.array([
            grid[i].solution_value()
            for i in range(n)
        ])

        optimized_hvac = np.array([
            hvac[i].solution_value()
            for i in range(n)
        ])

        optimized_battery = np.array([
            battery_use[i].solution_value()
            for i in range(n)
        ])

        total_cost_value = sum(
            optimized_grid * tariff
        )

        return {

            "optimized_grid": optimized_grid,

            "optimized_hvac": optimized_hvac,

            "battery_usage": optimized_battery,

            "solar_output": solar_output,

            "total_cost": total_cost_value,

            "peak_load": max(optimized_grid),

            "status": "OPTIMAL"

        }

    else:

        return {

            "status": "FAILED"

        }


# --------------------------------------------------
# TESTING
# --------------------------------------------------

if __name__ == "__main__":

    # Simulated prediction
    predicted_load = np.array([
        250,240,230,220,210,220,
        300,350,400,450,500,520,
        540,530,520,500,480,450,
        400,350,320,300,280,260
    ])

    result = optimize_energy(
        predicted_load
    )

    print("\n========== OPTIMIZATION RESULTS ==========\n")

    print("Status:", result["status"])

    print("\nPeak Load:")
    print(result["peak_load"])

    print("\nTotal Cost:")
    print(result["total_cost"])

    print("\nOptimized Grid Usage:")
    print(result["optimized_grid"])

    print("\nBattery Usage:")
    print(result["battery_usage"])