import random
import math
from collections import deque
# numpy might be useful for other EA aspects, but not strictly required by the algo as written
# import numpy as np

# Import functions from our other project files
from .llm_interface import M_llm_output
from .fitness_scorer import f_exec_score, f_log_score
from .prompt_operations import calculate_prompt_similarity, mutate_prompt_guided

class IterativePromptOptimizer:
    def __init__(self, D_train, D_val,
                 T_max, P_S, k_e, m,
                 tau_max, beta,
                 delta_fitness, N_stable, D_min,
                 w_exec=0.7, w_log=0.3, epsilon=0.1, alpha=0.5,
                 seed_prompts=None):

        self.D_train = D_train
        self.D_val = D_val
        self.T_max = T_max
        self.P_S = P_S
        self.k_e = k_e
        self.k_o = P_S - k_e # offspring count
        if self.k_o < 0:
            raise ValueError("Population size P_S must be greater than or equal to elite count k_e.")
        self.m = m # Replay buffer capacity
        self.tau_max = tau_max
        self.beta = beta
        self.delta_fitness = delta_fitness
        self.N_stable = N_stable
        self.D_min = D_min # Min diversity for convergence (Note: not in main loop's stopping criteria in provided algo)

        self.w_exec = w_exec
        self.w_log = w_log
        self.epsilon = epsilon # Weight for replay fitness contribution
        self.alpha = alpha # Smoothing factor for fitness (EMA)

        self.population = [] # List of prompts (strings)
        # Stores (prompt_str, fitness_float) tuples, sorted by fitness (best first)
        self.replay_buffer = deque(maxlen=self.m)
        self.smoothed_fitness = {} # Dict: {prompt_str: smoothed_fitness_float}

        if seed_prompts is None:
            self.seed_prompts = ["Initial seed prompt example.", "Another basic instruction template."]
        else:
            self.seed_prompts = seed_prompts


    def _scoring_function(self, prompt: str, context: str, target_answer: str) -> float:
        """ Implements Eq. \ref{eq:scoring_metric} using imported components """
        llm_output_str = M_llm_output(prompt, context) # From llm_interface
        score_exec = f_exec_score(llm_output_str, target_answer) # From fitness_scorer
        score_log = f_log_score(llm_output_str, target_answer) # From fitness_scorer
        return self.w_exec * score_exec + self.w_log * score_log

    def _evaluate_initial_fitness(self, prompt: str, dataset: list) -> float:
        if not dataset: return 0.0
        total_fitness = 0.0
        # For initial fitness, evaluate on the whole training set or a substantial sample
        # The algorithm text implies full D_train: EvaluateInitialFitness(rho, D_train)
        # Let's use the full dataset as per the algo for initialization step.
        for context, answer in dataset:
            total_fitness += self._scoring_function(prompt, context, answer)
        return total_fitness / len(dataset)

    def _initialize_population(self):
        """Phase 1: Initialization"""
        self.population = []
        # Generate initial population U_0 using mutation of seed prompts
        for i in range(self.P_S):
            seed = random.choice(self.seed_prompts)
            # Using tau_max for initial diverse mutations, no specific batch guidance yet
            initial_prompt = mutate_prompt_guided(seed, self.tau_max, [])
            self.population.append(initial_prompt)
            # Initialize smoothed fitness f_bar_0(rho)
            # This is the EvaluateInitialFitness(rho, D_train) part
            initial_f = self._evaluate_initial_fitness(initial_prompt, self.D_train)
            self.smoothed_fitness[initial_prompt] = initial_f
        print(f"Initialized population with {self.P_S} prompts. Smoothed fitnesses calculated.")
        # Replay buffer B_replay is initialized empty as per algo line 2

    def _calculate_mean_pairwise_similarity(self, current_population: list) -> float:
        if len(current_population) < 2:
            return 0.0 # Max diversity (or 1.0 if similarity means identity)
        
        total_similarity = 0
        pair_count = 0
        for i in range(len(current_population)):
            for j in range(i + 1, len(current_population)):
                # Uses imported similarity function
                total_similarity += calculate_prompt_similarity(current_population[i], current_population[j])
                pair_count += 1
        return total_similarity / pair_count if pair_count > 0 else 0.0

    def optimize(self) -> str:
        """
        Runs the iterative prompt optimization algorithm.
        Ensure D_train and D_val are lists of (context, answer) tuples.
        """
        # Phase 1: Initialization (already called if you structure it this way, or call here)
        self._initialize_population() # Call initialization
        # self.replay_buffer is already an empty deque from __init__

        t = 1 # Current generation
        generations_without_improvement = 0
        # Stores the best smoothed fitness found in the previous generation for delta check
        last_gen_best_smoothed_fitness = -float('inf')
        if self.population: # Initialize from current best if pop exists
            last_gen_best_smoothed_fitness = max(self.smoothed_fitness.get(p, -float('inf')) for p in self.population)


        # Phase 2: Evolutionary Loop
        while t <= self.T_max and generations_without_improvement < self.N_stable:
            print(f"\n--- Generation {t}/{self.T_max} ---")

            # A. Mini-Batch Sampling
            # Formula: n_t = ceil(0.1 * |D_train| * (1 + t/T_max))
            # Note: Algorithm's T is T_max. For t/T, use (t-1)/T_max if t is 1-indexed and progress is 0 to 1.
            # Or t/T_max if progress is 1/T_max to 1. Let's use t/T_max.
            sampling_rate_factor = 0.1 * (1 + t / self.T_max) # Ensures factor increases
            n_t = math.ceil(sampling_rate_factor * len(self.D_train))
            n_t = min(n_t, len(self.D_train)) # Cap at dataset size
            n_t = max(n_t, 1 if self.D_train else 0) # Ensure at least 1 sample if D_train not empty
            
            current_batch_D_train = []
            if n_t > 0:
                current_batch_D_train = random.sample(self.D_train, n_t)
            print(f"Sampled mini-batch B_t of size {n_t} from D_train.")

            # This will store f_t(rho) for each rho in U_{t-1}
            current_generation_stochastic_fitness = {}

            # B. Stochastic Fitness Evaluation for Population U_{t-1} (which is self.population)
            for rho in self.population:
                # Calculate f_replay(rho)
                f_replay_rho = 0.0
                if self.replay_buffer: # Check if |B_replay| > 0
                    sum_weighted_sim_fitness = 0
                    # sum_similarities = 0 # Not in algo, but could be used for normalization
                    for rho_prime, f_prime_in_buffer in self.replay_buffer:
                        sim_rho_rho_prime = calculate_prompt_similarity(rho, rho_prime)
                        sum_weighted_sim_fitness += sim_rho_rho_prime * f_prime_in_buffer
                        # sum_similarities += sim_rho_rho_prime
                    if len(self.replay_buffer) > 0: # Algo: (1/|B_replay|)
                        f_replay_rho = self.epsilon * (1 / len(self.replay_buffer)) * sum_weighted_sim_fitness
                
                # Calculate f_batch(rho)
                f_batch_rho = 0.0
                if current_batch_D_train: # Check if |B_t| > 0
                    batch_fitness_sum = 0
                    for context, answer in current_batch_D_train:
                        batch_fitness_sum += self._scoring_function(rho, context, answer)
                    f_batch_rho = batch_fitness_sum / len(current_batch_D_train)
                
                # Total stochastic fitness f_t(rho)
                current_generation_stochastic_fitness[rho] = f_batch_rho + f_replay_rho

            # C. Update Smoothed Fitness and Select Elites from U_{t-1}
            # Store \bar{f}_t(\rho) for prompts in current population U_{t-1}
            updated_smoothed_fitness_for_selection = {}
            for rho in self.population:
                # Get \bar{f}_{t-1}(\rho) (previous smoothed fitness)
                # For a prompt rho, self.smoothed_fitness[rho] holds \bar{f}_{t-1}(rho) at this point.
                prev_smoothed_f = self.smoothed_fitness.get(rho, 0.0) # Default to 0 if new (should not happen for existing pop)
                stochastic_f_t = current_generation_stochastic_fitness.get(rho, 0.0)
                
                # Calculate \bar{f}_t(\rho)
                new_smoothed_f = self.alpha * prev_smoothed_f + (1 - self.alpha) * stochastic_f_t
                updated_smoothed_fitness_for_selection[rho] = new_smoothed_f
            
            # Update the global smoothed_fitness dict with these new values for the current population members
            self.smoothed_fitness.update(updated_smoothed_fitness_for_selection)

            # Select k_e elites E_t based on \bar{f}_t(.)
            # Sort U_{t-1} (self.population) by their newly calculated \bar{f}_t(\rho)
            sorted_population_by_new_smoothed_fitness = sorted(
                self.population,
                key=lambda r: self.smoothed_fitness.get(r, -float('inf')), # Use updated values
                reverse=True
            )
            elites_Et = sorted_population_by_new_smoothed_fitness[:self.k_e]
            if elites_Et: print(f"Selected {len(elites_Et)} elites. Top elite smoothed fitness: {self.smoothed_fitness.get(elites_Et[0], -1):.4f}")
            else: print("No elites selected (population might be smaller than k_e or empty).")

            # D. Offspring Generation
            mutation_temp_tau_t = self.tau_max * math.exp(-self.beta * (t-1)) # t is 1-indexed, so t-1 for correct exponent
            
            # Select parents P_parents (e.g., tournament or use E_t)
            # Algorithm suggests using E_t (elites) as parents or tournament selection.
            # For simplicity, if elites exist, we use them. Otherwise, the whole sorted population.
            parent_pool = elites_Et if elites_Et else sorted_population_by_new_smoothed_fitness
            if not parent_pool: # Fallback if population is somehow empty
                parent_pool = self.seed_prompts # Use seeds to regenerate
                print("Warning: Parent pool empty, using seed prompts for offspring generation.")
                # Initialize fitness for seed prompts if they are not in smoothed_fitness
                for seed_p in parent_pool:
                    if seed_p not in self.smoothed_fitness:
                        self.smoothed_fitness[seed_p] = self._evaluate_initial_fitness(seed_p, self.D_train[:max(1, len(self.D_train)//10)])


            offspring_Ot = []
            for _ in range(self.k_o): # Generate k_o offspring
                if not parent_pool: # Should not be reached if fallback to seeds works
                    # Create a completely random prompt if no parents
                    parent_rho = "A generic fallback prompt"
                    if parent_rho not in self.smoothed_fitness: self.smoothed_fitness[parent_rho] = 0.0
                else:
                    parent_rho = random.choice(parent_pool) # SelectParent(P_parents)
                
                # Mutate(rho_parent, tau_t, B_t) - guided mutation using current batch
                offspring_rho = mutate_prompt_guided(parent_rho, mutation_temp_tau_t, current_batch_D_train)
                offspring_Ot.append(offspring_rho)
                
                # Initialize smoothed fitness for new offspring: \bar{f}_{t-1}(offspring) = \bar{f}_{t-1}(parent)
                # This means offspring inherits the parent's smoothed fitness *from before this generation's update*.
                # Tricky: self.smoothed_fitness[parent_rho] has ALREADY been updated to \bar{f}_t(parent_rho).
                # To get \bar{f}_{t-1}(parent_rho), we'd need to store it or use the value before the EMA update.
                # A common heuristic is to assign parent's *current* smoothed_fitness or re-evaluate.
                # The algorithm is specific: "\bar{f}_{t-1}(\rho_{offspring}) \gets \bar{f}_{t-1}(\rho_{\text{parent}})"
                # Let's assume `updated_smoothed_fitness_for_selection` stored `\bar{f}_t` and
                # `self.smoothed_fitness` was not updated *globally* until after this step,
                # OR we fetch parent's fitness from *before* it was put into `updated_smoothed_fitness_for_selection`.
                # For simplicity now: use the parent's fitness that was used for ITS selection into `elites_Et` or parent_pool.
                # This would be `self.smoothed_fitness.get(parent_rho)` which is effectively \bar{f}_t(parent).
                # A more accurate interpretation of \bar{f}_{t-1} would be to fetch it from the state *before* line 17.
                # Let's use `self.smoothed_fitness.get(parent_rho)` which is the parent's just-updated \bar{f}_t.
                # This is often done in practice.
                # If an offspring is entirely new, it needs an initial smoothed_fitness value.
                parent_fitness_to_inherit = self.smoothed_fitness.get(parent_rho, 0.0) # Fallback to 0 if parent somehow not in dict
                self.smoothed_fitness[offspring_rho] = parent_fitness_to_inherit


            # E. Form New Population U_t for Next Generation
            self.population = elites_Et + offspring_Ot
            # Ensure population size P_S is maintained (e.g., if P_S != k_e + k_o)
            if len(self.population) > self.P_S:
                self.population = self.population[:self.P_S]
            elif len(self.population) < self.P_S: # Refill if too small (e.g. k_o was 0 and k_e < P_S)
                print(f"Population size {len(self.population)} is less than P_S {self.P_S}. Refilling.")
                needed = self.P_S - len(self.population)
                for _ in range(needed):
                    refill_parent = random.choice(parent_pool if parent_pool else self.seed_prompts)
                    new_prompt = mutate_prompt_guided(refill_parent, mutation_temp_tau_t, current_batch_D_train)
                    self.population.append(new_prompt)
                    # Initialize fitness for refilled prompts
                    self.smoothed_fitness[new_prompt] = self.smoothed_fitness.get(refill_parent, 0.0)


            # F. Update Replay Buffer B_replay with top-m performers from U_t
            # We need fitness values for U_t to select top-m. These are in self.smoothed_fitness.
            
            # Create a list of (prompt, fitness) for all unique prompts in current population and replay buffer
            # Then sort and pick top m to form the new replay buffer.
            # This ensures the replay buffer always contains the historical best unique individuals.
            
            combined_candidates_for_replay = {} # Using dict to store unique prompts with their best fitness
            for p_r, f_r in self.replay_buffer: # Add existing replay items
                combined_candidates_for_replay[p_r] = f_r
            
            for p_u in self.population: # Add/update with current population items
                current_f = self.smoothed_fitness.get(p_u, -float('inf'))
                if p_u not in combined_candidates_for_replay or current_f > combined_candidates_for_replay[p_u]:
                    combined_candidates_for_replay[p_u] = current_f
            
            sorted_replay_candidates = sorted(combined_candidates_for_replay.items(), key=lambda item: item[1], reverse=True)
            
            self.replay_buffer = deque(sorted_replay_candidates[:self.m], maxlen=self.m)

            if self.replay_buffer:
                 print(f"Replay buffer updated. Size: {len(self.replay_buffer)}. Best in buffer fitness: {self.replay_buffer[0][1]:.4f} (Prompt: '{self.replay_buffer[0][0][:30]}...')")


            # G. Convergence Check
            current_gen_best_smoothed_fitness = -float('inf')
            if self.population: # Get max \bar{f}_t(rho) from U_t
                current_gen_best_smoothed_fitness = max(self.smoothed_fitness.get(rho, -float('inf')) for rho in self.population)
            
            print(f"Current generation's best smoothed fitness in population: {current_gen_best_smoothed_fitness:.4f}")

            # Check improvement against previous generation's best
            if t > 1 and abs(current_gen_best_smoothed_fitness - last_gen_best_smoothed_fitness) < self.delta_fitness:
                generations_without_improvement += 1
                print(f"Fitness improvement less than delta. Generations without improvement: {generations_without_improvement}")
            else:
                generations_without_improvement = 0 # Reset if improvement
            
            last_gen_best_smoothed_fitness = current_gen_best_smoothed_fitness # Update for next iteration
            
            # Calculate Diversity D(U_t)
            diversity_Ut = 1.0 - self._calculate_mean_pairwise_similarity(self.population)
            print(f"Population diversity D(U_t): {diversity_Ut:.4f}")
            # Note: D_min is a param but not used in the WHILE condition in the provided algo.
            # Could add: if diversity_Ut < self.D_min: print("Diversity below D_min!"); # break or other action

            t += 1 # Increment generation counter
            if generations_without_improvement >= self.N_stable:
                print(f"Stopping: Stable for {self.N_stable} generations.")
                break
            # Max generations check is in the while loop condition

        if t > self.T_max and generations_without_improvement < self.N_stable:
            print(f"Stopping: Reached max generations {self.T_max}.")

        # Phase 3: Final Selection
        print("\n--- Phase 3: Final Selection on Validation Set D_val ---")
        if not self.D_val:
            print("Warning: Validation dataset D_val is empty.")
            if not self.population:
                print("Error: No population to select from for final result.")
                return "Error: No population available."
            # Fallback: return the best prompt from the final population based on training smoothed fitness
            print("Returning best prompt from final population based on smoothed training fitness.")
            best_prompt_overall = max(self.population, key=lambda r: self.smoothed_fitness.get(r, -float('inf')))
            return best_prompt_overall

        # Evaluate prompts on D_val using the main scoring function (Eq. \ref{eq:scoring_metric})
        # The algorithm implies evaluating on D_val to select rho*.
        # Consider which prompts to evaluate: final population U_t? Or all unique good ones found (e.g. in replay buffer)?
        # The algorithm snippet is a bit vague: "Using Eq. X to select rho* on D_val"
        # Let's evaluate all unique prompts from the final population and the replay buffer.
        
        candidate_prompts_for_final_eval = set(self.population)
        for p_replay, _ in self.replay_buffer:
            candidate_prompts_for_final_eval.add(p_replay)
        
        print(f"Evaluating {len(candidate_prompts_for_final_eval)} unique candidate prompts on D_val...")

        final_prompt_evaluations = {} # {prompt: validation_score}
        for rho_candidate in candidate_prompts_for_final_eval:
            val_fitness_sum = 0
            for context, answer in self.D_val:
                val_fitness_sum += self._scoring_function(rho_candidate, context, answer)
            final_score = val_fitness_sum / len(self.D_val) if self.D_val else 0.0
            final_prompt_evaluations[rho_candidate] = final_score
            print(f"  Prompt: '{rho_candidate[:60]}...' Val_Score: {final_score:.4f}")

        if not final_prompt_evaluations:
            print("Error: No prompts evaluated for final selection.")
            return "Error: No prompts for final selection."

        # Select rho* with the best score on D_val
        optimal_prompt_rho_star = max(final_prompt_evaluations, key=final_prompt_evaluations.get)
        best_val_score = final_prompt_evaluations[optimal_prompt_rho_star]
        print(f"\nOptimal prompt rho*: '{optimal_prompt_rho_star}' with validation score: {best_val_score:.4f}")
        return optimal_prompt_rho_star