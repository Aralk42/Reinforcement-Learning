import numpy as np

class antenna_distributions:
    ##################################################
    #
    # ALEATORIA: 
    #
    ##################################################

    @staticmethod
    def random_positions(num_ant,room,seed):
        rng = np.random.default_rng(seed=seed)
        tx_positions = [(round(rng.uniform(-room[0]/2,room[0]/2),2), 
                         round(rng.uniform(-room[1]/2,room[1]/2),2), 
                        room[2]) for _ in range(int(num_ant))]
        return tx_positions

    ##################################################
    #
    # GRID: 
    #
    ##################################################
    @staticmethod
    def grid_antenna_positions(num_ant,antenna_distances,height):
        tx_positions = [(-(np.sqrt(num_ant)-1)*antenna_distances/2+i * antenna_distances, 
                        -(np.sqrt(num_ant)-1)*antenna_distances/2+j * antenna_distances, 
                        height) for i in range(int(np.sqrt(num_ant))) 
                        for j in range(int(np.sqrt(num_ant)))]
        return tx_positions

    ##################################################
    #
    # LINEAR: 
    #
    ##################################################

    # Perpendicular to the receiver position.
    @staticmethod
    def perpendicular_linear_antenna_distribution(num_ant,antenna_distances,height,r_position):
        phi = np.radians(90.0)-np.atan(r_position[1]/(r_position[0]+1e-9))
        print(np.degrees(phi),np.cos(phi), np.sin(phi))
        r_max = antenna_distances*num_ant/2+antenna_distances/2
        tx_positions = [(-(r_max-i*antenna_distances)*np.cos(phi), 
                        -(r_max-i*antenna_distances)*np.sin(phi), 
                        height) for i in range(int(num_ant)) ]

        return tx_positions
    
    # Parallel to one of the walls.
    @staticmethod
    def y_linear_antenna_distribution(num_ant, antenna_distances, height):
        r_max = antenna_distances*num_ant/2+antenna_distances/2
        tx_positions = [(0, 
                        -(r_max-i*antenna_distances), 
                        height) for i in range(int(num_ant)) ]

        return tx_positions

    # Diagonal to both room walls.
    @staticmethod
    def diagonal_linear_antenna_distribution(num_ant, antenna_distances, height):
        r_max = antenna_distances*num_ant/2+antenna_distances/2
        tx_positions = [(-(r_max-i*antenna_distances), 
                        -(r_max-i*antenna_distances), 
                        height) for i in range(int(num_ant)) ]

        return tx_positions
    
    ##################################################
    #
    # CIRCULAR: 
    #
    ##################################################
    @staticmethod
    def get_circular_positions(num_ant,r,height):
        angle_antennas = np.radians(360/num_ant)
        tx_positions = [(r*np.cos(angle_antennas*i), 
                        r*np.sin(angle_antennas*i), 
                        height) for i in range(int(num_ant)) 
                        ]
        return tx_positions
    
    ##################################################
    #
    # CROSS: 
    #
    ##################################################
    @staticmethod
    def cross_antenna_distribution(num_antennas, antenna_distances,height):
        r_max = antenna_distances*num_antennas/2/2-antenna_distances/2
        tx_positions_1 = [(-(r_max-i*antenna_distances), 
                        0, 
                        height) for i in range(int(num_antennas/2))]
        tx_positions_2 = [(0, 
                        -(r_max-j*antenna_distances), 
                        height) 
                        for j in range(int(num_antennas/2))]
        tx_positions = tx_positions_1 + tx_positions_2
                        

        return tx_positions