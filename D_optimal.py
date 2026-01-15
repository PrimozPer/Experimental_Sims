import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm


def normalise(x, xmin, xmax):
    return ((x-xmin) / (xmax - xmin))*2-1 


def model_matrix(X):
    a = X[:,0]
    b = X[:,1]
    d = X[:,2]
    j = X[:,3]

    return np.column_stack([
        np.ones(len(a)),   # intercept

        # linear
        a, b, d, j,

        # quadratic
        a**2, b**2, d**2, j**2,
        
        #cubic
        a**3, b**3, d**3, j**3,

        # interactions 
        a*b, a*d, a*j,
        b*d, b*j,
        d*j
        #three-way interactions 
        # ,a*b*d, a*b*j, a*d*j, b*d*j,
      
    ])


def d_optimal_design(Phi, n_points, n_iter=300,pool_size=400):
    n_candidates = Phi.shape[0]
    idx = np.random.choice(n_candidates, n_points, replace=False)
    X = Phi[idx]
    XtX = X.T @ X
    det_old = np.linalg.det(XtX)
    for _ in tqdm(range(n_iter), desc="D-optimal design"):
        improved = False

    
        for i in range(n_points):
            candidate_pool = np.random.choice(n_candidates, size=pool_size, replace=False)
            for j in candidate_pool:
                if j in idx:
                    continue

                X_new = X.copy()
                X_new[i] = Phi[j]


                sign, logdet = np.linalg.slogdet(X_new.T @ X_new)
                det_new = np.exp(logdet) if sign > 0 else 0


                if det_new > det_old:
                    idx[i] = j
                    X = X_new
                    det_old = det_new
                    improved = True
                    break
            if improved:
                break

        # if not improved:
        #     break

    return idx

def compute_trim_motion_time(df, alpha_rate=0.5, beta_rate=0.25,
                              alpha_col='alpha', beta_col='beta',
                              alpha_start=0.0, beta_start=0.0):
    """
    Compute total motion time assuming alpha and beta move simultaneously.

    Parameters:
    - df: pandas DataFrame with alpha and beta columns
    - alpha_rate:  degree per second for alpha
    - beta_rate: degree per second for beta
    - alpha_start: starting alpha position (deg)
    - beta_start: starting beta position (deg)

    Returns:
    - total_time (seconds)
    - step_times (list of step times)
    """

    total_time = 0.0
    step_times = []

    prev_alpha = alpha_start
    prev_beta = beta_start

    for _, row in df.iterrows():
        da = abs(row[alpha_col] - prev_alpha)
        db = abs(row[beta_col] - prev_beta)

        t_alpha = da / alpha_rate
        t_beta = db / beta_rate

        step_time = max(t_alpha, t_beta)+ 20  # add 20 seconds buffer

        step_times.append(step_time)
        total_time += step_time

        prev_alpha = row[alpha_col]
        prev_beta = row[beta_col]

    return total_time, step_times


def compute_total_motion_time(df, alpha_rate=0.5, beta_rate=0.25,
                              alpha_col='alpha', beta_col='beta',
                              alpha_start=0.0, beta_start=0.0):
    """
    Compute total motion time assuming alpha and beta move simultaneously.

    Parameters:
    - df: pandas DataFrame with alpha and beta columns
    - alpha_rate:  degree per second for alpha
    - beta_rate: degree per second for beta
    - alpha_start: starting alpha position (deg)
    - beta_start: starting beta position (deg)

    Returns:
    - total_time (seconds)
    - step_times (list of step times)
    """

    total_time = 0.0
    step_times = []

    prev_alpha = alpha_start
    prev_beta = beta_start

    for _, row in df.iterrows():
        da = abs(row[alpha_col] - prev_alpha)
        db = abs(row[beta_col] - prev_beta)

        t_alpha = da / alpha_rate
        t_beta = db / beta_rate

        step_time = max(t_alpha, t_beta) + 30+30+30  # add 30 seconds for measuring, 30 sec for settling and 30 sec for prop speed

        step_times.append(step_time)
        total_time += step_time

        prev_alpha = row[alpha_col]
        prev_beta = row[beta_col]

    return total_time, step_times

###########################################################################
###########################################################################


alpha_bounds = [-4, 14]   # deg
beta_bounds  = [-5, 5]    # deg 
J_bounds = [1.2, 4]          # advance ratio


#Create variable sets

alpha_vals = np.linspace(*alpha_bounds, 19)
beta_vals  = np.linspace(*beta_bounds, 11)
J_vals     = np.linspace(*J_bounds, 10)
         
delta_e_vals = np.array([-10, 0, 10])                  # fixed
candidates = np.array(list(product(alpha_vals, beta_vals, J_vals, delta_e_vals)))


alpha_n = normalise(candidates[:,0], *alpha_bounds)
beta_n  = normalise(candidates[:,1], *beta_bounds)

J_n     = normalise(candidates[:,2], *J_bounds)
de_n    = normalise(candidates[:,3], -10, 10)

Xc = np.column_stack([alpha_n, beta_n, J_n, de_n])


Phi = model_matrix(Xc)


########RUnning D-optimal design selection##########

np.random.seed(np.random.randint(1e6))

n_runs = 22   # total runs

best_idx = d_optimal_design(Phi, n_runs, n_iter=1000)
optimal_design = candidates[best_idx]

df_opt = pd.DataFrame(optimal_design, columns=['alpha', 'beta', 'J', 'delta_e'])

print(df_opt.sort_values(by=['delta_e']).reset_index(drop=True))



print("\nTotal used time for D-optimal measurments:")
total_time, step_times = compute_total_motion_time(df_opt)
total_time = total_time + 20*60 + 5*60 # add 20 minutes for ailerons and 5 for 1st startup
print(f"Total time: {total_time:.2f} seconds")
print(f'Total time: {total_time/60:.2f} minutes')

print("Total trim time for required measurments:")
total_time_trim,step_times_trim = compute_trim_motion_time(df_opt)
print(f"Trim time: {total_time_trim:.2f} seconds")
print(f'Trim time: {total_time_trim/60:.2f} minutes')
grand_total = total_time + total_time_trim
print(f"\nGrand total time: {grand_total/60:.2f} minutes")

#plot in 3d scatter plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_opt['alpha'], df_opt['beta'], df_opt['J'], marker='o')
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('J')
plt.show()

