import numpy as np


class modelos_indoor:
    c = 3e8
       # FALTA TWO-PATH MODEL

    @staticmethod
    def to_dBm(x): return 10*np.log10(x+1e-30)+30
    @staticmethod
    def dBm_to_W(dBm): return 10**(dBm/10)/1000

    # Dipole angular dependency:
    @staticmethod
    def theta_r(_tx,_rx):
        R_path = np.linalg.norm(_rx - _tx)
        cos_theta = abs((_rx - _tx))[2] / (R_path + 1e-15)
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))
        return sin_theta**2
    
    # Algunos modelos dependen de la distancia. Revisar.
    @staticmethod
    def breakpoint_distance(h_bs,h_ut,h_e,params):
        freq = params["f"]
        lam = modelos_indoor.c / freq
        hprima_bs = h_bs - h_e
        hprima_ut = h_ut - h_e
        return 4*hprima_bs*hprima_ut/lam

    ##################################################
    #
    # FRIIS EQUATION MODEL: 
    #
    ##################################################

    @staticmethod
    def single_antenna_received_power(tx, tx_phase, r_position,params):
        freq = params["f"]
        lam = modelos_indoor.c / freq
        k = 2 * np.pi / lam
        G_t = params["G_t"]
        G_r = params["G_r"]
        Pt_i = params["P_total"] / params["n"]

        R_path = np.linalg.norm(np.array(r_position)  - np.array(tx))
        dir_gain = modelos_indoor.theta_r(tx,r_position)
        L_i = G_t*G_r*dir_gain*(lam / (4*np.pi*R_path))**2
        phase = tx_phase-k * R_path
        E = np.sqrt(Pt_i*L_i)*np.exp(1j*phase)

        return E

    @staticmethod
    def total_power_Friis(tx_positions,tx_theta,r_position,params):

        E_total = 0+0j
        for tx,tx_phase  in zip(tx_positions,tx_theta):
            
            E_total += modelos_indoor.single_antenna_received_power(tx, tx_phase, r_position,params)

        Pr_total = np.abs(E_total)**2  # potencia proporcional al |E|^2
        return Pr_total #Watts
    
    ##################################################
    #
    # TWO-PATH MODEL
    #
    ##################################################

    @staticmethod
    def two_path_model(tx_positions,tx_theta,r_position,params):
        Pt_i = params["P_total"] / params["n"]
        G_t = params["G_t"]
        G_r = params["G_r"]
        lam = modelos_indoor.c / params["f"]
        k = 2 * np.pi / lam
        E_total = 0+0j
        for tx,tx_phase  in zip(tx_positions,tx_theta):
            # Calcula 'break distance' a partir de la cual se puede usar este modelo. Si es una distancia menor, utilizamos la fórmula de Friis, sin reflexiones. 
            R_path = np.linalg.norm(np.array(r_position) - np.array(tx))
            x_distance = np.array(tx)[0]-np.array(r_position)[0]
            y_distance = np.array(tx)[1]-np.array(r_position)[1]
            Ground_path = np.sqrt(x_distance**2+y_distance**2)
        
            critical_distance = (4*np.array(tx)[2]*np.array(r_position)[2])/lam
            ground_reflection_coeficient = -1.0

            dir_gain_DIRECT = modelos_indoor.theta_r(tx,r_position)
            G_0 = G_r*G_t*dir_gain_DIRECT

            dir_gain_REFLECTED = modelos_indoor.theta_r([tx[0],tx[1],-tx[2]],r_position) # La dirección del rayo reflejado es virtualmente el rayo incidente desde -h_t
            G_1 = G_r*G_t*dir_gain_REFLECTED # dir_gain_REFLECTED el ángulo de entrada del rayo reflejado por el suelo.
            d1 = np.sqrt(x_distance**2+y_distance**2+(np.array(tx)[0]+np.array(r_position)[0])**2)

            # Cálculo la ganancia según la dirección a la que la antena receptora recibe la señal, ya que es un dipolo y la ganancia depende del ángulo de entrada:
            
            if Ground_path > critical_distance:
                L_i = G_t*G_r*dir_gain_DIRECT*(lam / (4*np.pi*R_path))**2
                phase = tx_phase#-k * R_path
                E = np.sqrt(Pt_i*L_i)*np.exp(1j*phase)
            else: 
                phase_0 = tx_phase-k * R_path 
                phase_1 = tx_phase-k * d1 # ESTA FASE NO SE SI ESTÁ BIEN
                E1 = (np.sqrt(Pt_i*G_0)/R_path)*np.exp(1j*phase_0)
                E2 = (ground_reflection_coeficient*np.sqrt(Pt_i*G_1)/d1)*np.exp(1j*phase_1)
                E = (lam / (4.0*np.pi))*(E1+E2)

            E_total += E

        Pr_total = np.abs(E_total)**2  
        return Pr_total # to_dBm(Pr_total) 

    ##################################################
    #
    # TOTAL POWER CALCULATIUON WITH SHADOWING 3GPP:
    #
    ##################################################

    # Los
    @staticmethod
    def LOS_PL(d,fc):
        fc_GHz = fc/10**9
        sigma = 3
        rng = np.random.default_rng(seed=0)
        PL_los = 32.4+17.3*np.log10(d+1e-30)+20*np.log10(fc_GHz+1e-30)+rng.normal(0, sigma)
        return PL_los

    @staticmethod
    def NLOS_PL(d,fc):
        fc_GHz = fc/10**9
        sigma = 8.03
        rng = np.random.default_rng(seed=0)
        PLprima = 38.3*np.log10(d+1e-30)+17.3+24.9*np.log10(fc_GHz+1e-30)+rng.normal(0, sigma)
        PL_nlos = max(modelos_indoor.LOS_PL(d,fc),PLprima)
        return PL_nlos

    # LOS probabilities:
    @staticmethod
    def Probability_indoor_mixed_office(tx, r_position):
        d_2d = np.sqrt((r_position[0]-tx[0])**2+(r_position[1]-tx[1])**2)
        if (d_2d<=1.2):
            return 1
        elif (1.2<d_2d<6.5):
            return np.exp(-(d_2d-1.2)/4.7)
        else:
            return np.exp(-(d_2d-6.5)/32.6)*0.32

    @staticmethod
    def Probability_indoor_open_office(tx, r_position):
        d_2d = np.sqrt((r_position[0]-tx[0])**2+(r_position[1]-tx[1])**2)
        if (d_2d<=5):
            return 1
        elif (5<d_2d<49):
            return np.exp(-(d_2d-5)/70.8)
        else:
            return np.exp(-(d_2d-49)/211.7)*0.54
        
    @staticmethod
    def single_antenna_received_power_shadowing(tx, tx_phase, r_position,params):
        R_path = np.linalg.norm(np.array(r_position)  - np.array(tx))
        dir_gain = modelos_indoor.theta_r(tx,r_position)
        Pt_i = params["P_total"] / params["n"]
        lam = modelos_indoor.c / params["f"]
        k = 2 * np.pi / lam

        # Según LOS probabilities sería LOS_PL o NLOS_PL. Dehecho, debería sumarse ambas aportaciones de Path loss según las probabilidades de una u otra. 
        # Ahora solo LOS in mixed Indoor: 
        #if R_path>=1 and R_path <= 150:
        prob = modelos_indoor.Probability_indoor_mixed_office(tx, r_position)
        PL_dbm = prob*modelos_indoor.LOS_PL(R_path,params["f"]) + (1-prob)*modelos_indoor.NLOS_PL(R_path,params["f"]) #dBm

        PL_total = modelos_indoor.to_dBm(Pt_i)+params["G_t"]+params["G_r"]*dir_gain-PL_dbm
        PTt_W = modelos_indoor.dBm_to_W(PL_total)
        phase = tx_phase-k * R_path
        E = np.sqrt(PTt_W)*np.exp(1j*phase)
      
        return E, prob

    @staticmethod
    def total_power_shadowing(tx_positions,tx_theta,r_position,params):
        #print(tx_positions,tx_theta,r_position,params)
        E_total = 0+0j
        prob = 0
        for tx,tx_phase  in zip(tx_positions,tx_theta):
            E, p = modelos_indoor.single_antenna_received_power_shadowing(tx, tx_phase, r_position,params)
            E_total += E
            prob += p

        Pr_total = np.abs(E_total)**2  # potencia proporcional al |E|^2
        prob_media = prob/len(tx_theta) if len(tx_theta) > 0 else 0.0
        return Pr_total, prob_media #Watts