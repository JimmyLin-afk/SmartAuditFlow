import random
import math
from collections import deque
import re # Import re for sentence splitting
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
                 lambda_complexity=0.01, 
                 w_cov=0.6, w_det=0.4, w_F=0.25, w_V=0.25, w_M=0.25, w_E=0.25, # Added new weights
                 seed_prompts=None, history_file_path=None):

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
        self.D_min = D_min # Min diversity for convergence (Note: not in main loop\\'s stopping criteria in provided algo)

        self.w_exec = w_exec
        self.w_log = w_log
        self.epsilon = epsilon # Weight for replay fitness contribution
        self.alpha = alpha # Smoothing factor for fitness (EMA)
        self.lambda_complexity = lambda_complexity # Store lambda_complexity
        self.w_cov = w_cov # Store new weights
        self.w_det = w_det
        self.w_F = w_F
        self.w_V = w_V
        self.w_M = w_M
        self.w_E = w_E

        self.population = [] # List of prompts (strings)
        # Stores (prompt_str, fitness_float) tuples, sorted by fitness (best first)
        self.replay_buffer = deque(maxlen=self.m)
        self.smoothed_fitness = {} # Dict: {prompt_str: smoothed_fitness_float}
        self.evolution_history = [] # To store metrics for each generation
        self.history_file_path = history_file_path

        if seed_prompts is None:
            self.seed_prompts = ["Initial seed prompt example.", "Another basic instruction template."]
        else:
            self.seed_prompts = seed_prompts

    def _calculate_complexity(self, prompt: str) -> float:
        """
        Calculates the complexity of a prompt using TokenCount and SentenceCount.
        Complexity(rho) = w1 * TokenCount(rho) + w2 * SentenceCount(rho)
        where w1 = 0.7 and w2 = 0.3.
        """
        # Simple tokenization by splitting on whitespace
        token_count = len(prompt.split())
        # Simple sentence segmentation by splitting on common sentence endings
        sentence_count = len(re.split(r'[.!?]+', prompt.strip()))
        if prompt.strip() == "": # Handle empty string case
            sentence_count = 0

        w1 = 0.7
        w2 = 0.3
        return (w1 * token_count) + (w2 * sentence_count)

    def _scoring_function(self, prompt: str, context: str, target_answer: str) -> float:
        """
        Implements Eq. \ref{eq:scoring_metric} using imported components and adds complexity regularization.
        Objective: f(rho, c, A) - lambda * Complexity(rho)
        """
        llm_output_str = M_llm_output(prompt, context) # From llm_interface
        score_exec = f_exec_score(llm_output_str, target_answer, 
                                  w_cov=self.w_cov, w_det=self.w_det, 
                                  w_F=self.w_F, w_V=self.w_V, w_M=self.w_M, w_E=self.w_E) # Pass new weights
        score_log = f_log_score(llm_output_str, target_answer) # From fitness_scorer
        
        # Calculate the base quality score
        quality_score = self.w_exec * score_exec + self.w_log * score_log
        
        # Calculate complexity and apply regularization
        complexity_score = self._calculate_complexity(prompt)
        regularized_score = quality_score - (self.lambda_complexity * complexity_score)
        
        return regularized_score

    def _evaluate_initial_fitness(self, prompt: str, dataset: list) -> float:
        if not dataset: return 0.0
        total_fitness = 0.0
        for context, answer in dataset:
            total_fitness += self._scoring_function(prompt, context, answer)
        return total_fitness / len(dataset)

    def _initialize_population(self):
        """Phase 1: Initialization"""
        self.population = []
        for i in range(self.P_S):
            seed = random.choice(self.seed_prompts)
            initial_prompt = mutate_prompt_guided(seed, self.tau_max, [])
            self.population.append(initial_prompt)
            initial_f = self._evaluate_initial_fitness(initial_prompt, self.D_train)
            self.smoothed_fitness[initial_prompt] = initial_f
        print(f"Initialized population with {self.P_S} prompts. Smoothed fitnesses calculated.")

    def _calculate_mean_pairwise_similarity(self, current_population: list) -> float:
        if len(current_population) < 2:
            return 0.0
        
        total_similarity = 0
        pair_count = 0
        for i in range(len(current_population)):
            for j in range(i + 1, len(current_population)):
                total_similarity += calculate_prompt_similarity(current_population[i], current_population[j])
                pair_count += 1
        return total_similarity / pair_count if pair_count > 0 else 0.0

    def optimize(self) -> str:
        """
        Runs the iterative prompt optimization algorithm.
        Ensure D_train and D_val are lists of (context, answer) tuples.
        """
        self._initialize_population()

        t = 1
        generations_without_improvement = 0
        last_gen_best_smoothed_fitness = -float("inf")
        if self.population:
            last_gen_best_smoothed_fitness = max(self.smoothed_fitness.get(p, -float("inf")) for p in self.population)

        while t <= self.T_max and generations_without_improvement < self.N_stable:
            print(f"\n--- Generation {t}/{self.T_max} ---")

            sampling_rate_factor = 0.1 * (1 + t / self.T_max)
            n_t = math.ceil(sampling_rate_factor * len(self.D_train))
            n_t = min(n_t, len(self.D_train))
            n_t = max(n_t, 1 if self.D_train else 0)
            
            current_batch_D_train = []
            if n_t > 0:
                current_batch_D_train = random.sample(self.D_train, n_t)
            print(f"Sampled mini-batch B_t of size {n_t} from D_train.")

            current_generation_stochastic_fitness = {}

            for rho in self.population:
                f_replay_rho = 0.0
                if self.replay_buffer:
                    sum_weighted_sim_fitness = 0
                    for rho_prime, f_prime_in_buffer in self.replay_buffer:
                        sim_rho_rho_prime = calculate_prompt_similarity(rho, rho_prime)
                        sum_weighted_sim_fitness += sim_rho_rho_prime * f_prime_in_buffer
                    if len(self.replay_buffer) > 0:
                        f_replay_rho = self.epsilon * (1 / len(self.replay_buffer)) * sum_weighted_sim_fitness
                
                f_batch_rho = 0.0
                if current_batch_D_train:
                    batch_fitness_sum = 0
                    for context, answer in current_batch_D_train:
                        batch_fitness_sum += self._scoring_function(rho, context, answer)
                    f_batch_rho = batch_fitness_sum / len(current_batch_D_train)
                
                current_generation_stochastic_fitness[rho] = f_batch_rho + f_replay_rho

            updated_smoothed_fitness_for_selection = {}
            for rho in self.population:
                prev_smoothed_f = self.smoothed_fitness.get(rho, 0.0)
                stochastic_f_t = current_generation_stochastic_fitness.get(rho, 0.0)
                
                new_smoothed_f = self.alpha * prev_smoothed_f + (1 - self.alpha) * stochastic_f_t
                updated_smoothed_fitness_for_selection[rho] = new_smoothed_f
            
            self.smoothed_fitness.update(updated_smoothed_fitness_for_selection)

            sorted_population_by_new_smoothed_fitness = sorted(
                self.population,
                key=lambda r: self.smoothed_fitness.get(r, -float("inf")),
                reverse=True
            )
            elites_Et = sorted_population_by_new_smoothed_fitness[:self.k_e]
            if elites_Et: print(f"Selected {len(elites_Et)} elites. Top elite smoothed fitness: {self.smoothed_fitness.get(elites_Et[0], -1):.4f}")
            else: print("No elites selected (population might be smaller than k_e or empty).")

            mutation_temp_tau_t = self.tau_max * math.exp(-self.beta * (t-1))
            
            parent_pool = elites_Et if elites_Et else sorted_population_by_new_smoothed_fitness
            if not parent_pool:
                parent_pool = self.seed_prompts
                print("Warning: Parent pool empty, using seed prompts for offspring generation.")
                for seed_p in parent_pool:
                    if seed_p not in self.smoothed_fitness:
                        self.smoothed_fitness[seed_p] = self._evaluate_initial_fitness(seed_p, self.D_train[:max(1, len(self.D_train)//10)])

            offspring_Ot = []
            for _ in range(self.k_o):
                if not parent_pool:
                    parent_rho = "A generic fallback prompt"
                    if parent_rho not in self.smoothed_fitness: self.smoothed_fitness[parent_rho] = 0.0
                else:
                    parent_rho = random.choice(parent_pool)
                
                offspring_rho = mutate_prompt_guided(parent_rho, mutation_temp_tau_t, current_batch_D_train)
                offspring_Ot.append(offspring_rho)
                
                parent_fitness_to_inherit = self.smoothed_fitness.get(parent_rho, 0.0)
                self.smoothed_fitness[offspring_rho] = parent_fitness_to_inherit

            self.population = elites_Et + offspring_Ot
            if len(self.population) > self.P_S:
                self.population = self.population[:self.P_S]
            elif len(self.population) < self.P_S:
                print(f"Population size {len(self.population)} is less than P_S {self.P_S}. Refilling.")
                needed = self.P_S - len(self.population)
                for _ in range(needed):
                    refill_parent = random.choice(parent_pool if parent_pool else self.seed_prompts)
                    new_prompt = mutate_prompt_guided(refill_parent, mutation_temp_tau_t, current_batch_D_train)
                    self.population.append(new_prompt)
                    self.smoothed_fitness[new_prompt] = self.smoothed_fitness.get(refill_parent, 0.0)

            combined_candidates_for_replay = {}
            for p_r, f_r in self.replay_buffer:
                combined_candidates_for_replay[p_r] = f_r
            
            for p_u in self.population:
                current_f = self.smoothed_fitness.get(p_u, -float("inf"))
                if p_u not in combined_candidates_for_replay or current_f > combined_candidates_for_replay[p_u]:
                    combined_candidates_for_replay[p_u] = current_f
            
            sorted_replay_candidates = sorted(combined_candidates_for_replay.items(), key=lambda item: item[1], reverse=True)
            
            self.replay_buffer = deque(sorted_replay_candidates[:self.m], maxlen=self.m)

            if self.replay_buffer:
                 print(f"Replay buffer updated. Size: {len(self.replay_buffer)}. Best in buffer fitness: {self.replay_buffer[0][1]:.4f} (Prompt: \'{self.replay_buffer[0][0][:30]}...\')")

            current_gen_best_smoothed_fitness = -float("inf")
            if self.population:
                current_gen_best_smoothed_fitness = max(self.smoothed_fitness.get(rho, -float("inf")) for rho in self.population)
            
            print(f"Current generation\'s best smoothed fitness in population: {current_gen_best_smoothed_fitness:.4f}")

            if t > 1 and abs(current_gen_best_smoothed_fitness - last_gen_best_smoothed_fitness) < self.delta_fitness:
                generations_without_improvement += 1
                print(f"Fitness improvement less than delta. Generations without improvement: {generations_without_improvement}")
            else:
                generations_without_improvement = 0
            
            last_gen_best_smoothed_fitness = current_gen_best_smoothed_fitness
            
            diversity_Ut = 1.0 - self._calculate_mean_pairwise_similarity(self.population)
            print(f"Population diversity D(U_t): {diversity_Ut:.4f}")

            self.evolution_history.append({
                "generation": t,
                "best_smoothed_fitness": current_gen_best_smoothed_fitness,
                "population_diversity": diversity_Ut,
                "generation_prompts": [{"prompt": p, "score": self.smoothed_fitness.get(p, 0.0)} for p in self.population]
            })
            self.save_evolution_history(self.history_file_path)

            # Add diversity check for convergence
            if diversity_Ut < self.D_min:
                print(f"Stopping: Population diversity {diversity_Ut:.4f} is below D_min {self.D_min:.4f}.")
                break

            t += 1
            if generations_without_improvement >= self.N_stable:
                print(f"Stopping: Stable for {self.N_stable} generations.")
                break

        if t > self.T_max and generations_without_improvement < self.N_stable:
            print(f"Stopping: Reached max generations {self.T_max}.")

        print("\n--- Phase 3: Final Selection on Validation Set D_val ---")
        if not self.D_val:
            print("Warning: Validation dataset D_val is empty.")
            if not self.population:
                print("Error: No population to select from for final result.")
                return "Error: No population available."
            print("Returning best prompt from final population based on smoothed training fitness.")
            best_prompt_overall = max(self.population, key=lambda r: self.smoothed_fitness.get(r, -float("inf")))
            return best_prompt_overall

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
            print(f"  Prompt: \'{rho_candidate[:60]}...\' Val_Score: {final_score:.4f}")

        if not final_prompt_evaluations:
            print("Error: No prompts evaluated for final selection.")
            return "Error: No prompts for final selection."

        optimal_prompt_rho_star = max(final_prompt_evaluations, key=final_prompt_evaluations.get)
        best_val_score = final_prompt_evaluations[optimal_prompt_rho_star]
        print(f"\nOptimal prompt rho*: \'{optimal_prompt_rho_star}\' with validation score: {best_val_score:.4f}")
        return optimal_prompt_rho_star




    def save_evolution_history(self, file_path: str):
        """
        Saves the recorded evolution history to a specified JSON file.
        """
        import json
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.evolution_history, f, indent=4)
        print(f"Evolution history saved to {file_path}")
