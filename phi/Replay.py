import numpy as np
import pandas as pd
import pyphi

# Calculating Φ values for replay experiment in Gidon et al., 2025.
# This file was written by Albert Gidon 
#   17/01/2025

# Scientific Context
# 1.	System Overview:
# o	    The model consists of 5 cortical neurons (A,B,C,D,E) with a feedforward LGN input (LGN A, LGN B).
#       o	The LGN is activated by visual input (green light) and considered as a part of the system
#               but never as a part of the major complex after the calculation of Phi.
#       o   Initial state is identical to the LGN inputs, activated by the light  presented to the
#                    artificial subject. Also, after the network evolved one step, the LGN 
#                   caused neurons A or B to fire.
# 2.	Key Features:
#       o	Connectivity Matrix (cm) defines directed connections among neurons based on cortical interconnections.
#                   connectivity is like a ring, [a --> c,d,e],[b --> d,e,a],[c --> e,a,b], [d --> a,b,c], [e --> b,c,d]
#                   additionally, feedforward connection from the LGN [LNGA --> a] and [LGNB --> b]
#       o	Transition Probability Matrix (TPM) encodes how neurons transition between states based on 
#                   the neurons and LGN input
# 3.	Dynamic Rules (producing the TPM):
#       o	Nodes activate if the sum of their neighbors' states plus LGN input
#                   meets or exceeds a defined threshold (THRESHOLD).
# 4.	Analysis Goals:
#       o	Compute Big Phi (Φ) for the major complex to measure the system's integration under LGN input - i.e.
#           presenting the artificial subject blue light.
#           Note that we do here blue light because it maintains Φ > 0 ,and the simplified mode
#           is still close to the simulation in the paper.

pyphi.config.PROGRESS_BARS = True
pyphi.config.VALIDATE_SUBSYSTEM_STATES = True

def int2bits(n,N):
    return np.array([n >> int(i) & 1 for i in range(N)])  # Extract each bit, from least significant to most significant

def bits2int(s):
    return np.sum([int(bit) << i for i, bit in enumerate(s)])  # Combine bits into an integer

def artificial_subject_phi(light, threshold, replay,  **kwargs):

    N = 7
    N2 = 2**N 
    cm = np.array([
        # A    B    C    D    E    LGNA LGNB
        [0,   0,   1,   1,   1,   0,   0],  # Row 0: A -> C, D, E
        [1,   0,   0,   1,   1,   0,   0],  # Row 1: B -> A, D, E
        [1,   1,   0,   0,   1,   0,   0],  # Row 2: C -> A, B, E
        [1,   1,   1,   0,   0,   0,   0],  # Row 3: D -> A, B, C
        [0,   1,   1,   1,   0,   0,   0],  # Row 4: E -> B, C, D
        [1,   0,   0,   0,   0,   0,   0],  # Row 5: LGNA -> A
        [0,   1,   0,   0,   0,   0,   0],  # Row 6: LGNB -> B
    ])

    cm = cm[0:N,0:N]
    THRESHOLD = threshold  #the number of nodes that can cause a target node to fire
    WEIGHT = np.array((1,1,1,1,1,THRESHOLD,THRESHOLD))[0:N] # LGN nodes connected to nodes A and B always make them cross the threshold.
    
    #input to the LGN
    BLUE_LIGHT = np.array((0,0,0,0,0,1,0))[0:N]
    RED_LIGHT = np.array((0,0,0,0,0,0,1))[0:N]
    NO_LIGHT = np.array((0,0,0,0,0,0,0))[0:N]
    GREEN_LIGHT = BLUE_LIGHT + RED_LIGHT

    LGN_ON = BLUE_LIGHT if light == "blue"\
        else RED_LIGHT if light == "red"\
        else GREEN_LIGHT if light == "green"\
        else NO_LIGHT
    LGN_ON = LGN_ON[0:N]
    CURRENT_STATE = kwargs.get('current_state', LGN_ON)[0:N] #set the current state according to the evolution of the network activity
    REPLAY_STATE = kwargs.get('replay_state', np.array([0,0,0,0,0,0,0]))[0:N]  # give therecorded states for the voltage clamp manually


    def get_next_state(S, replay_type):
        if replay_type == "no replay":  # No replay, natural dynamics
            Sn = get_next_state_noreplay(S)
        elif replay_type == "ff replay":  # Feedforward replay, don't care about S
            Sn = get_next_state_ff()
        elif replay_type == "fb replay":  # Feedback replay
            Sn = get_next_state_fb(S)
        else:
            raise ValueError("Invalid replay_type. Must be 'no replay', 'ff replay', or 'fb replay'.")
        return Sn


    # Note: The connectivity matrix (cm) is transposed because we are interested in 
    # the connections TO a mechanism rather than FROM it (as specified in the cm).
    def get_next_state_noreplay(S):
        # Ensure the threshold is crossed when LGN is active by making the synaptic weight to the value of the threshold
        Sn = np.matmul(cm.T, S*WEIGHT)   
        # print("Current",S*WEIGHT)
        Sn = (Sn >= THRESHOLD).astype(int)
        # print("Next",Sn)
        return Sn

    # Feedforward Replay:
    # In this mode, the input is ignored, and the nodes are set directly based on the recorded state.

    def get_next_state_ff():
        # LGN is not determined by the voltage clamp, we set it always to zero here
        return np.append(REPLAY_STATE,(0,0))[0:N] #me to myself: weird code
    

    # Feedback Replay: This function simulates the operation of a feedback replay mechanism,
    # mimicking the behavior of a voltage clamp. The voltage clamp forces the system's state
    # to match pre-recorded values, effectively overriding the natural dynamics of the system.

    def get_next_state_fb(S):
        """
        Parameters:
        S: Current state of the system (e.g., a vector or matrix representing neuronal states).
        Returns:
        voltage_clamp: The next state of the system after applying the voltage clamp,
                    which matches the pre-recorded state values.
        """
        # Calculate the state that the system would transition to naturally without replay
        Sn = get_next_state_noreplay(S)  # Function to compute the next state without any feedback
        
        # Retrieve the pre-recorded state values for the current system context
        # Note that LGN nodes are free to do whatever they want.
        command_voltage = np.append(REPLAY_STATE,Sn[N-2:N])[0:N]  # Pre-determined states (assumes this function is defined elsewhere)
        
        # Compute the delta between the natural state and the recorded state
        # Delta represents the adjustment made by the voltage clamp to enforce the recorded state
        delta_voltage = Sn - command_voltage
        
        # Apply the voltage clamp adjustment
        # The final state is computed by subtracting the delta from the natural state
        voltage_clamp = Sn - delta_voltage
        
        # Note:
        # - The voltage clamp forces the state of the system to match the recorded values,
        #   effectively "replaying" a recorded  state.
        #   The next state for clamped neurons is the recorded pattern, no matter what.        
        # - This example demonstrates where IIT does not distinguish between feedforward replay
        #   (e.g., imposed inputs) and feedback replay (e.g., voltage clamp) for determining
        #   system dynamics or integrated information.

        # - IIT focuses on the causal structure, not the mechanism enforcing the replay.
        return voltage_clamp

    # Initialize a Transition Probability Matrix (TPM) for the system.
    tpm = np.array([[0] * N2 for _ in range(N2)])  # Create an NxN zero matrix, where N2 = 2^N (number of possible states)

    for current_state in range(N2):
        """
        For each possible state of the system:
        - Convert the current state (integer) into its binary representation.
        - Calculate the next state based on the dynamics defined by `get_next_state`.
        - Update the TPM to indicate a deterministic transition from the current state to the next state.
        """
        binary_state = int2bits(current_state, N)
        next_state = get_next_state(binary_state, replay)  # Get the next state based on replay type
        next_state = bits2int(next_state)
        tpm[current_state][next_state] += 1.0 # Set the transition probability to 1 for the deterministic case

    network = pyphi.Network(tpm, cm=cm, node_labels=['A', 'B', 'C', 'D', 'E', 'LGNA', 'LGNB'][0:N])
    # Compute the major complex of the network for a given current state.
    # The major complex represents the subset of the system with maximum integrated information (Φ).
    major_complex = pyphi.compute.major_complex(network, CURRENT_STATE)
    print( f"big Φ for {major_complex.subsystem} {replay} and with {light} light stimulus is {major_complex.phi} (thereshold={THRESHOLD})" )
    
    state_labels = ['"' + format(i, f"0{N+1}b")[::-1] +'"' for i in range(N2)]
    csvfilename = "tpm (" + light + "_" 
    csvfilename += "replay" if replay else "noreplay"
   
    pd.DataFrame(tpm, index=state_labels, columns=state_labels).to_csv(csvfilename + ".csv")
    return major_complex, 


####################################################################################################

####See results below

theta = 2
# print("Threshold set to", theta)
# No inputs
artificial_subject_phi(light = "no",
                       threshold=theta, 
                       replay = "no replay",
                       current_state = np.array((0,0,0,0,0,0,0)))
# input to the LGN - this is the first step in the simulation where LGN A was activated but cell A is not yet active 
artificial_subject_phi(light = "blue",    
                       threshold=theta, 
                       replay = "no replay",
                       current_state = np.array((0,0,0,0,0,1,0)))
# input to the LGN is still on, and the simulation evolved one step whereby cell A is active
artificial_subject_phi(light = "blue",    
                       threshold=theta, 
                       replay = "no replay",
                       current_state = np.array((1,0,0,0,0,1,0)))
print ("feedforward replay")
artificial_subject_phi(light = "blue",
                       threshold=theta,
                       replay = "ff replay",
                       current_state = np.array((1,0,0,0,0,1,0)),
                       replay_state = np.array((1,0,0,0,0)))

print ("congruent feedback replay - blue replayed during blue input")
artificial_subject_phi(light = "blue",
                       threshold=theta, 
                       replay = "fb replay",
                       current_state = np.array((1,0,0,0,0,1,0)),
                       replay_state = np.array((1,0,0,0,0)))

print ("incongruent feedback replay  - blue replayed during red input")
artificial_subject_phi(light = "blue",    
                       threshold=theta, 
                       replay = "fb replay",
                       current_state = np.array((0,1,0,0,0,0,0)),
                       replay_state = np.array((1,0,0,0,0)))


#RESULTS for congruent replay:
#################
#   blue=cortical cell A fires above the threshold
#   threshold = 2: number of inputs required to activate a mechanism.

# no replay
# big Φ for Subsystem(A, B, C, D, E) no replay and with no light stimulus is 10.729488 (thereshold=2)
# big Φ for Subsystem(A, C, D, E) no replay and with blue light stimulus is 0.366389 (thereshold=2, current state =0,0,0,0,0,0,1)
# big Φ for Subsystem(A, D) no replay and with blue light stimulus is 1.0 (thereshold=2, current state = 0,1,0,0,0,0,1)

# feedforward replay
# big Φ for Subsystem() ff replay and with the blue light stimulus is 0.0 (thereshold=2)

# congruent feedback replay - blue replayed during blue input
# big Φ for Subsystem() fb replay and with blue light stimulus is 0.0 (thereshold=2)
# incongruent feedback replay  - blue replayed during red input
# big Φ for Subsystem() fb replay and with blue light stimulus is 0.0 (thereshold=2)
