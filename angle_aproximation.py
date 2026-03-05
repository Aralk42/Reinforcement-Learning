import numpy as np
from Models import modelos_indoor

class angle_approx_algorithms:
    def __init__(self, params):
        c = 3e8
        self.params = params    
        self.power_equations = modelos_indoor()

    ##################################################
    #
    # OPTIMAL ANGLE (ANALYTICAL CALCULATION)
    #
    ##################################################
    def ideal_beamforming_phases(self,tx_positions, r_position):
        freq = self.params["f"]
        lam = self.c / freq
        k = 2 * np.pi / lam
        tx_positions = np.array(tx_positions)
        R = np.linalg.norm(r_position - tx_positions, axis=1)
        return k * R  # modulo 2π if desired

    ##################################################
    #
    # HILL CLIMBING ALGORITHM: 
    #
    ##################################################
    def phase_optimization(self,tx_theta,tx_positions,phase_shifter, p_model,r_focus):
        current = tx_theta.copy()
        n_iterations = 1000
        current_eval = p_model(tx_positions,tx_theta, r_focus,self.params)
        P_plot = []
        for j in range(len(tx_theta)):
            # tx_theta[theta] = np.linalg.norm(r_position - tx_positions[theta])*2 * np.pi / (c/f)
            for _ in range(n_iterations):
                new_up = current.copy()
                new_down = current.copy()
                new_up[j] += phase_shifter
                new_down[j] -= phase_shifter
                P_eval = [p_model(tx_positions,new_up, r_focus,self.params),
                        p_model(tx_positions,new_down, r_focus,self.params)]
                best_idx = np.argmax(P_eval)

                if P_eval[best_idx] > current_eval:
                    current = [new_up,new_down][best_idx]
                    current_eval = P_eval[best_idx]
                    P_plot.append(current_eval)
                else:
                    break

        return current, current_eval, P_plot
    
    ##################################################
    #
    # GET OPTIMUM PHASES WITH HILL CLIMBING Y MODELO DE FRIIS:
    #
    ##################################################

    # De momento aproxima con el modelo de Friis. Se podría pasar el modelo con el que se está trabajando.
    def get_optimum_phase(self,pos_beam,r_position, Iterations, phase_shifter):
        all_theta = [0 for _ in range(Iterations)]
        all_p = [0 for _ in range(Iterations)]
        power_all_iterations = []
        for i in range(Iterations):
            tx_theta = [np.random.uniform(0,2*np.pi) for _ in range(len(pos_beam))]
            tx_theta_n, P_r, P_plot = self.phase_optimization(tx_theta,pos_beam,phase_shifter,modelos_indoor.total_power_Friis, r_position)
            power_all_iterations.append(P_plot)
            all_theta[i] = tx_theta_n
            all_p[i] = P_r

        index_opt = all_p.index(max(all_p))
        theta_opt = all_theta[index_opt] #all_theta/N_theta
        return theta_opt, power_all_iterations

    ##################################################
    #
    # # GET OPTIMUM PHASES WITH GRADIENT DESCENT:
    #
    ##################################################

    # En gradient descent se le pasa un número de iteraciones, pero para mayor optimización 
    # podría pasarsele un umbral de cambio entre iteraciones (y un número de iteraciones máximas)
    def phase_gradient_descent_LOS(self,tx_positions, r_position,iterations,learning_rate,get_single_antenna_power_method): 
        umbral = 0.0001
        rng = np.random.default_rng(seed=0)
        tx_theta = rng.uniform(0, 2*np.pi, size=len(tx_positions))
        power_vector = []
        E_total = 0+0j
        for i in range(iterations):
            E_total = np.sum([get_single_antenna_power_method(tx, tx_phase, r_position,self.params) 
                        for tx, tx_phase in zip(tx_positions, tx_theta)])
            Pr = np.abs(E_total)**2
            if (i > 2 and abs(modelos_indoor.to_dBm(Pr)-modelos_indoor.to_dBm(power_vector[-1])) < umbral): break
            power_vector.append(Pr)
            for k, tx in enumerate(tx_positions):
                E_k = get_single_antenna_power_method(tx, tx_theta[k], r_position,self.params)
                Deriv_P = -2 * np.imag(np.conj(E_total) * E_k)
                tx_theta[k] = np.mod(tx_theta[k] + learning_rate * Deriv_P, 2*np.pi)

        return tx_theta, power_vector
