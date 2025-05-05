import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import gurobipy as gp #! DUE TO MIXED INTEGER QUADRATIC PROGRAM

"""
https://www.cvxpy.org/examples/basic/mixed_integer_quadratic_program.html

The whole optimization program could be applied to realistic scenarios just by tweaking the names of the details.
The main point of this is to show the overall form of the optimization in an interesting manner while being relevant to real situations.
"""

all_items = {
            "Wooden Spear":      {"Attack": 8, "Defense": 1, "Cost": 18, "Weight": 3, "Space": 8},
            "Iron Sword":        {"Attack": 22, "Defense": 2, "Cost": 50, "Weight": 9, "Space": 5},
            "Wooden Shield":     {"Attack": 1, "Defense": 9, "Cost": 16, "Weight": 5, "Space": 4},
            "Iron Shield":       {"Attack": 3, "Defense": 20, "Cost": 40, "Weight": 12, "Space": 5},
            "Fire Club":         {"Attack": 48, "Defense": 2,  "Cost": 80,  "Weight": 16,  "Space": 9},
            "Oil Bottle":        {"Attack": 0,  "Defense": 0,  "Cost": 15,  "Weight": 1,  "Space": 1},
            "Dagger":            {"Attack": 12, "Defense": 1,  "Cost": 25,  "Weight": 1,  "Space": 2}, 
            "Longbow":           {"Attack": 18, "Defense": 1,  "Cost": 45,  "Weight": 6,  "Space": 3}, 
            "Scale Armor":       {"Attack": 0,  "Defense": 40, "Cost": 90, "Weight": 18, "Space": 10},
            "Leather Boots":     {"Attack": 0,  "Defense": 5,  "Cost": 26,  "Weight": 2,  "Space": 1},
            "Battle Axe":        {"Attack": 35, "Defense": 2,  "Cost": 65,  "Weight": 15, "Space": 8},
            "Bronze Mace":       {"Attack": 20, "Defense": 0,  "Cost": 45,  "Weight": 7,  "Space": 4},
            "Cloak":             {"Attack": 0,  "Defense": 7,  "Cost": 25,  "Weight": 2,  "Space": 1},
            "Magic Staff":       {"Attack": 26, "Defense": 3,  "Cost": 60,  "Weight": 7, "Space": 6},
            "Boomerang":         {"Attack": 6, "Defense": 0,  "Cost": 12,  "Weight": 1, "Space": 2},
            "Throwing Stars":    {"Attack": 4, "Defense": 0,  "Cost": 8,  "Weight": 1, "Space": 1},
            "Ice Wand":          {"Attack": 25, "Defense": 0,  "Cost": 50,  "Weight": 1, "Space": 2},
            "Trident":           {"Attack": 28, "Defense": 3,  "Cost": 75,  "Weight": 7, "Space": 6},
            "Steel Bracers":     {"Attack": 1,  "Defense": 8,  "Cost": 35,  "Weight": 3,  "Space": 2},
            "Sapphire Ring":     {"Attack": 10,  "Defense": 4,  "Cost": 99,  "Weight": 0,  "Space": 0},
            }

all_item_names = list(all_items.keys())
num_items = len(all_item_names) # total number of items 

# Pairings for the Synergy Matrix (Row, Column)
pairings = {
    (0, 2): 1,     # Simple Spear & Wooden Shield
    (0, 18): 5,    # Simple Spear & Wrist Bracers
    (4, 5): 14,     # Fire Club & Oil Flask
    (6, 7): 5,     # Dagger & Longbow
    (7, 12): 3,    # Longbow & Cloak combo
    (12, 13): 7,   # Cloak & Magic Staff combo
    (10, 2): 6,    # Battle Axe and wooden shield
    (3, 11): 2,    # Iron Shield & Bronze Mace combo
    (17, 18): 4,   # Bonus for Trident and wrist bracers
    (16, 19): 7,   # Bonus for ice wand and ring
    (14, 18): 5    # Bonus for boomerang and wrist bracers
}

synergy = np.zeros(shape = (num_items, num_items))
for (i, j) in pairings:
    synergy[i, j] = pairings[(i, j)]
    synergy[j, i] = pairings[(i, j)] #To ensure symmetry


#To ensure negative semi-definiteness
eigen = np.linalg.eigvals(synergy)
max_eig = max(eigen)
temp_s = synergy / max_eig
s_nsd = temp_s - np.eye(num_items)
new_eigs = np.linalg.eigvalsh(s_nsd)

#Helper function
def _extract_stats_(key_name: str, data: dict = all_items):
    """
    Uses all_items dictionary by default, must be nested
    """
    n = len(data.keys())
    stats = np.zeros(n)
    sub_data = list(data.values())

    for i in range(n):
        stats[i] = sub_data[i][key_name]

    return stats

def _print_stats_(output = np.array):
    attack_arr = _extract_stats_("Attack")
    defense_arr = _extract_stats_("Defense")
    cost_arr = _extract_stats_("Cost")
    weight_arr = _extract_stats_("Weight")
    space_arr = _extract_stats_("Space")

    names = []
    for i in range(len(output)):
        if output[i]:
            names.append(all_item_names[i])
    
    print(names)
    print(output @ attack_arr, output @ defense_arr, output @ cost_arr, output @ weight_arr, output @ space_arr)


# Define the optimization function
def MIP_optimize(item_dict: dict, synergy_matrix: np.ndarray):
    n = len(item_dict.keys())

    alpha = 0.5
    beta = 0.5

    # Can make these parameters instead but for simplicity will keep them here
    max_gold = 350
    max_weight = 55
    max_space = 35

    min_attack = 70
    min_defense =  80

    # Decision variables: Whether or not the given item was purchased
    x = cvx.Variable(shape = (n), boolean = True, nonneg = False) 

    # Constraint list to be appended to
    constraints = []

    ###############################################################

    attack = _extract_stats_("Attack")
    defense = _extract_stats_("Defense")
    cost = _extract_stats_("Cost")
    weight = _extract_stats_("Weight")
    space = _extract_stats_("Space")

    total_power = (alpha * attack) + (beta * defense)

    objective = cvx.Minimize(-(total_power @ x) - ((1/2) * cvx.quad_form(x, synergy_matrix)) + (x @ cost))
    #objective = cvx.Minimize(-(total_power @ x) + (x @ cost)) #To match the dual problem version

    #Minimum attack constraint
    constraints.append(attack @ x >= min_attack)

    #Minimum defense constraint
    constraints.append(defense @ x >= min_defense)

    # Max Gold/Budget Constraint
    constraints.append(cost @ x <= max_gold)

    #Max Weight Constraint
    constraints.append(weight @ x <= max_weight)

    #Max Space Constraint
    constraints.append(space @ x <= max_space)

    #Cannot have 2 shields constraint
    constraints.append(x[2] + x[3] <= 1)
    
    ###############################################################

    # Solve the QP problem
    problem = cvx.Problem(objective, constraints)
    result = problem.solve(solver=cvx.GUROBI) 

    # Check the problem status
    if problem.status == cvx.OPTIMAL or problem.status == cvx.OPTIMAL_INACCURATE:  

        # Output results
        print("Primal problem status:", problem.status)
        print("Optimal Primal variable value:", x.value)
        print("Optimal Primal value:", problem.value)      

        # Optimal values
        optimal_layout = problem.value
        optimal_mix = x.value 
        return optimal_layout, optimal_mix
    else:
        print("The problem is infeasible or unbounded. Status:", problem.status)
        return "Positive infinity", "No feasible solution"

def dual_problem():

    attack = _extract_stats_("Attack")
    defense = _extract_stats_("Defense")
    cost = _extract_stats_("Cost")
    weight = _extract_stats_("Weight")
    space = _extract_stats_("Space")

    min_attack = 70
    min_defense = 80
    max_gold = 350
    max_weight = 55
    max_space = 35

    alpha = beta = 0.5

    total_power = alpha * attack + beta * defense

    # lambda represents greater than, mu for less than inequalities
    lambda1 = cvx.Variable(nonneg=True)
    lambda2 = cvx.Variable(nonneg=True)
    mu1 = cvx.Variable(nonneg=True)
    mu2 = cvx.Variable(nonneg=True)
    mu3 = cvx.Variable(nonneg=True)

    dual_obj = cvx.Maximize(
        lambda1 * min_attack + lambda2 * min_defense - mu1 * max_gold - mu2 * max_weight - mu3 * max_space
    )

    # Dual constraint for feasibility
    dual_constraints = [
        -lambda1 * attack - lambda2 * defense + mu1 * cost + mu2 * weight + mu3 * space >= total_power - cost
    ]

    dual_problem = cvx.Problem(dual_obj, dual_constraints)
    dual_problem.solve()

    # Output results
    print("Dual problem status:", dual_problem.status)
    print("Optimal dual variable values:", lambda1.value, lambda2.value, mu1.value, mu2.value, mu3.value)
    print("Optimal dual value:", dual_problem.value)
    return dual_problem


#Call dual function 
dual_sol = dual_problem()

print()

# Call the optimization function
result = MIP_optimize(item_dict = all_items, synergy_matrix = s_nsd)

_print_stats_(result[1])

