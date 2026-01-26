import torch
import numpy as np



'''

    The code example below demonstrates how to incorporate the basic ideas of PSO and GSA into the `rearrange_mask` function.

Note: 
    In real-world applications, the parameters of PSO and GSA may need to be fine-tuned to suit specific pruning tasks.

    The `fitness_function` in this code snippet evaluates the fitness of a particle's position, i.e., the sum of squared gradients of the pruned neurons.

    In practice, the fitness function may need to be customized based on specific objectives (e.g., minimizing model size while maintaining prediction accuracy).

    Both PSO and GSA are stochastic optimization algorithms, and multiple runs may yield different results. To obtain more stable performance, 
    it may be necessary to run the algorithm multiple times and take the average or best result.
'''
@torch.no_grad()
def pso_gsa_rearrange(mask, grads):
    print("Start using the PSO-GSA algorithm for mask rearrangement...")
    num_unpruned = int(mask.sum())
    num_pruned = mask.shape[0] - num_unpruned
    if num_unpruned == 0 or num_pruned == 0:
        return mask

    # Convert gradients to a form suitable for optimization
    grads = grads.permute(1, 0).contiguous()
    grads_sq = grads.pow(2).sum(dim=1).numpy()

    # PSO parameters (these should be tuned for your specific problem)
    num_particles = 50
    max_iterations = 100
    inertia_weight = 0.7
    cognitive_coefficient = 2.0
    social_coefficient = 2.0

    # GSA parameters
    gravity_constant = 1.0
    initial_mass = 1.0

    # Initialize particles randomly
    particles = np.random.randint(0, 2, size=(num_particles, grads.shape[0]))
    velocities = np.zeros_like(particles)

    # Initialize personal and global best positions
    personal_best_positions = particles.copy()
    global_best_position = particles[0].copy()

    # Calculate initial fitnesses
    personal_best_fitnesses = [fitness_function(particles[i], grads_sq, num_pruned) for i in range(num_particles)]
    global_best_fitness = min(personal_best_fitnesses)

    for iteration in range(max_iterations):
        # Update velocities and positions using PSO
        r1, r2 = np.random.rand(), np.random.rand()
        velocities = (inertia_weight * velocities +
                      cognitive_coefficient * r1 * (personal_best_positions - particles) +
                      social_coefficient * r2 * (global_best_position - particles))
        particles = (particles + velocities).clip(0, 1)

        # Apply GSA mass and gravity updates
        masses = np.array([initial_mass / (fitness_function(particles[i], grads_sq, num_pruned) + 1e-8) for i in range(num_particles)])
        total_mass = masses.sum()
        gravity = gravity_constant * (masses / total_mass)

        # Update personal and global best positions
        for i in range(num_particles):
            current_fitness = fitness_function(particles[i], grads_sq, num_pruned)
            if current_fitness < personal_best_fitnesses[i]:
                personal_best_fitnesses[i] = current_fitness
                personal_best_positions[i] = particles[i].copy()
                if current_fitness < global_best_fitness:
                    global_best_fitness = current_fitness
                    global_best_position = particles[i].copy()

    # Convert the global best position back to a mask
    new_mask = torch.tensor(global_best_position, dtype=torch.float32)
    new_mask[new_mask == 0] = 1
    new_mask[new_mask == 1] = 0
    return new_mask


# The fitness of particle location is evaluated, i.e., the sum of squared gradients of the pruned neurons.
def fitness_function(position, grads_sq, num_pruned):
    # Calculate the fitness based on the sum of squared gradients for the pruned neurons
    pruned_indices = np.where(position == 0)[0]
    grad_vectors = grads_sq[pruned_indices]
    return grad_vectors.sum()


def rearrange_mask(mask, grads):
    # NOTE: temporarily convert to CPU tensors as the arithmetic intensity is very low
    device = mask.device
    mask = mask.cpu()
    grads = grads.cpu()

    num_hidden_layers = mask.shape[0]
    for i in range(num_hidden_layers):
        '''
        Greedy algorithms, also known as the greedy method, are a common approach for finding optimal solutions to problems. This method typically divides the solution process into several steps, but each step applies the greedy principle, selecting the best/optimal choice (the locally most advantageous choice) under the current state, hoping that the final result will also be the best/optimal solution.

        (tips：Other algorithms could be considered, such as dynamic programming (0, 1 knapsack problem), etc.)
        '''
        # mask[i] = greedy_rearrange(mask[i], grads[:, i, :])
        mask[i] = pso_gsa_rearrange(mask[i], grads[:, i, :])

    mask = mask.to(device)
    return mask