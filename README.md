# linear-optimizer
An automated Integer Linear Programming (ILP) engine built in Python to mathematically maximize and optimize Daily Fantasy Sports lineups under strict salary constraints.

## Mathematical Formulation (With Flex Configuration)

This tool models the daily fantasy sports roster optimization problem as a **Mixed-Integer Linear Program (MILP)**. The system maximizes total projected yield while accounting for strict salary and multi-tiered positional requirements.

### 1. Decision Variables
Let $N$ be the total number of unique players available in the cleaned dataset. For each player $i \in \{1, 2, \dots, N\}$, we instantiate two distinct binary decision variables:

* $x_i \in \{0, 1\}$: Equal to $1$ if player $i$ is selected to fill their **base roster position**, and $0$ otherwise.
* $y_i \in \{0, 1\}$: Equal to $1$ if player $i$ is selected to fill the unique **Flex roster slot**, and $0$ otherwise.

### 2. Objective Function
The global objective is to maximize the aggregate expected points ($P_i$) across both base and flex selections:

$$\max \sum_{i=1}^{N} P_i (x_i + y_i)$$

### 3. Constraints

#### Budgetary Constraint (Salary Cap)
Let $S_i$ represent the financial cost of player $i$, and let $C$ represent the maximum portfolio budget ceiling (e.g., \$50,000). The total cost of all selected elements cannot cross this boundary:

$$\sum_{i=1}^{N} S_i (x_i + y_i) \le C$$

#### Structural Mutually Exclusive Constraint
A player can only be assigned to a single role in the active roster strategy. They cannot occupy both a base slot and a flex slot simultaneously:

$$x_i + y_i \le 1 \quad \forall i \in \{1, 2, \dots, N\}$$

#### Positional Roster Constraints
Let $QB$, $RB$, $WR$, and $TE$ represent the mutually exclusive subsets of players mapping to each athletic discipline. 

* **Quarterback Allocation:**
$$\sum_{i \in QB} x_i = 1$$

* **Running Back Base Requirement:**
$$\sum_{i \in RB} x_i = 1$$

* **Wide Receiver Base Requirement:**
$$\sum_{i \in WR} x_i = 1$$

* **Tight End Base Requirement:**
$$\sum_{i \in TE} x_i = 1$$

#### Flex Eligibility Constraint
The Flex position must be filled by exactly one player, restricted to the eligible positional sets of Running Backs, Wide Receivers, or Tight Ends:

$$\sum_{i \in \{RB \cup WR \cup TE\}} y_i = 1$$

To ensure structural validity, non-eligible categories (such as Quarterbacks) are strictly bounded to zero within the flex domain:

$$\sum_{i \in QB} y_i = 0$$

### 4. Integrality
All selection profiles are mapped to binary integers, enforcing discrete, non-fractional optimization:

$$x_i, y_i \in \{0, 1\} \quad \forall i \in \{1, 2, \dots, N\}$$

