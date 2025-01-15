import numpy as np
import pandas as pd
import pyphi

# Scientific Context
# 1.	System Overview:
#       o	The models consists 5 cortical neurons (A,B,C,D,E) with a feedforward LGN input.
#       o	The LGN is activated by visual input (green light) and acts as a background constraint for 
#                   Nodes A and B, ensuring their activation during stimulation.
#       o Initial state is identical to the LGN inputs which activate by the light  presented to the
#                    artificial subject.
# 2.	Key Features:
#       o	Connectivity Matrix (cm) defines directed connections among neurons based on cortical interconnections.
#                   connectivity is like a ring, [a --> c,d,e],[b --> d,e,a],[c --> e,a,b], [d --> a,b,c], [e --> b,c,d]
#       o	Transition Probability Matrix (TPM) encodes how neurons transition between states based on 
#                   the current state and LGN input (set as background constaints to simplify the analisys)
# 3.	Dynamic Rules (producing the TPM):
#       o	Nodes activate if the sum of their neighbors' states (plus LGN input, if applicable) 
#                   meets or exceeds a defined threshold (THRESHOLD).
#       o	Receving LGN inputs endure consistent activation accross all states.
# 4.	Analysis Goals:
#       o	Compute Big Phi (Φ) for the major complex to measure the system's integration under LGN input - i.e.
#           presenting the artificial subject green light.

pyphi.config.PROGRESS_BARS = True
pyphi.config.VALIDATE_SUBSYSTEM_STATES = True

def int2bits(n,N):
    return np.array([n >> int(i) & 1 for i in range(N)])  # Extract each bit, from least significant to most significant

def bits2int(s):
    return np.sum([int(bit) << i for i, bit in enumerate(s)])  # Combine bits into an integer

def artificial_subject_phi(light, threshold, replay,  **kwargs):
    # Matrix here  matches the NEURON simulation
    # CM[i][j] = 1 means that node i influences node j.
    # CM[i][j] = 0 means that node i does not influence node j.
    cm =  np.array([
        [0, 0, 1, 1, 1], 
        [1, 0, 0, 1, 1], 
        [1, 1, 0, 0, 1], 
        [1, 1, 1, 0, 0], 
        [0, 1, 1, 1, 0]  
    ])
    N = 5 
    N2 = 2**N 

    THRESHOLD = threshold  #two LGN nodes are on cause any cell to fire.
    BLUE_LIGHT = np.array((1,0,0,0,0))
    RED_LIGHT = np.array((0,1,0,0,0))
    NO_LIGHT = np.array((0,0,0,0,0))
    GREEN_LIGHT = BLUE_LIGHT + RED_LIGHT

    LGN_ON = BLUE_LIGHT if light == "blue"\
        else RED_LIGHT if light == "red"\
        else GREEN_LIGHT if light == "green"\
        else NO_LIGHT
    INITIAL_STATE = LGN_ON if 'initial_state' not in kwargs else kwargs['initial_state'] # initial state should be like the LGN connectivity - makes sense

   # Replay as Background Constraints:
    # This function simulates the next state of the system under various replay conditions,
    # including no replay (natural dynamics), feedforward replay, and feedback replay.
    # - The replay corresponds to background constraints, particularly when all cortical
    #   neurons are active, representing the congruent green light output in the simulation.
    # - Note: When replay is active (feedforward or feedback), the replay type does not
    #   influence the outcome, as the resulting TPM is identical for both replays.

    def get_next_state(a, b, c, d, e, replay_type):
        """
        Compute the next state of the system based on the current state and replay type.
        
        Parameters:
        a, b, c, d, e: Current states of the neurons (binary: 0 or 1).
        replay_type: Type of replay mechanism ("no replay", "ff replay", "fb replay").
        
        Returns:
        The next state encoded as an integer.
        """
        # Combine inputs into a state vector
        S = np.array((a, b, c, d, e))
        
        # Determine the next state based on the replay type
        if replay_type == "no replay":  # No replay, natural dynamics
            Sn = get_next_state_noreplay(S)
        elif replay_type == "ff replay":  # Feedforward replay, don't care about S
            Sn = get_next_state_ff()
        elif replay_type == "fb replay":  # Feedback replay
            Sn = get_next_state_fb(S)
        else:
            raise ValueError("Invalid replay_type. Must be 'no replay', 'ff replay', or 'fb replay'.")
        
        # Convert the binary state vector to an integer representation
        return Sn

        # Note: The connectivity matrix (cm) is transposed because we are interested in 
        # the connections TO a mechanism rather than FROM it (as specified in the cm).

    def get_next_state_noreplay(S):
        """
        Compute the next state of the system without replay, relying on natural dynamics.
        
        Parameters:
        S: Current state of the system (vector of binary values).
        
        Returns:
        Sn: Next state of the system (vector of binary values).
        """
        # Calculate the weighted input to each neuron using the transposed connectivity matrix
        Sn = np.matmul(cm.T, S) + LGN_ON * THRESHOLD  # Ensure threshold is crossed when LGN is active
        
        # Apply the thresholding function to determine the next state (binary activation)
        Sn = (Sn >= THRESHOLD).astype(int)
        return Sn

    # Note: The recorded state is calculated here instead of being explicitly recorded,
    # which simplifies the implementation for feedforward replay.

    def recorded_state():
        """
        Compute the recorded state based on LGN inputs and threshold conditions.
        
        Returns:
        A binary vector representing the recorded state.
        """
        # If the sum of LGN_ON inputs crosses the threshold, all are active
        # Otherwise, the recorded state matches the LGN inputs
        return np.ones(LGN_ON.shape) if LGN_ON.sum() >= THRESHOLD else LGN_ON

    # Feedforward Replay:
    # In this mode, the input is ignored, and the nodes are set directly based on the recorded state.

    def get_next_state_ff():
        """
        Compute the next state of the system under feedforward replay.
        
        Returns:
        The recorded state (binary vector) as the next state.
        """
        return recorded_state()



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
        command_voltage = recorded_state()  # Pre-determined states (assumes this function is defined elsewhere)
        
        # Compute the delta between the natural state and the recorded state
        # Delta represents the adjustment made by the voltage clamp to enforce the recorded state
        delta_voltage = Sn - command_voltage
        
        # Apply the voltage clamp adjustment
        # The final state is computed by subtracting the delta from the natural state
        voltage_clamp = Sn - delta_voltage
        
        # Note:
        # - The voltage clamp forces the state of the system to match the recorded values,
        #   effectively "replaying" a previously observed state.
        # - This example demonstrates where IIT does not distinguish between feedforward replay
        #   (e.g., imposed inputs) and feedback replay (e.g., voltage clamp) for determining
        #   system dynamics or integrated information.
        # - IIT focuses on the causal structure, not the mechanism enforcing the replay.
        return voltage_clamp


    # Initialize a Transition Probability Matrix (TPM) for the system.
    # The TPM captures the transition probabilities between all possible states of the system.
    tpm = np.array([[0] * N2 for _ in range(N2)])  # Create an NxN zero matrix, where N2 = 2^N (number of possible states)

    # Populate the TPM based on the system's dynamics.
    for current_state in range(N2):
        """
        For each possible state of the system:
        - Convert the current state (integer) into its binary representation.
        - Calculate the next state based on the dynamics defined by `get_next_state`.
        - Update the TPM to indicate a deterministic transition from the current state to the next state.
        """
        binary_state = int2bits(current_state, N)
        next_state = get_next_state(*binary_state, replay)  # Get the next state based on replay type
        next_state = bits2int(next_state)
        tpm[current_state][next_state] += 1.0 # Set the transition probability to 1 for the deterministic case

    # Create a PyPhi network using the computed TPM and connectivity matrix (cm).
    network = pyphi.Network(tpm, cm=cm, node_labels=['A', 'B', 'C', 'D', 'E'])

    # Compute the major complex of the network for a given initial state.
    # The major complex represents the subset of the system with maximum integrated information (Φ).
    major_complex = pyphi.compute.major_complex(network, INITIAL_STATE)

    print( f"big Φ for {major_complex.subsystem} {replay} and with {light} light stimulus is {major_complex.phi} (thereshold={THRESHOLD})" )
    
    state_labels = ['"' + format(i, "06b")[::-1] +'"' for i in range(N2)]
    csvfilename = "tpm (" + light + "_" 
    csvfilename += "replay" if replay else "noreplay"
   
    pd.DataFrame(tpm, index=state_labels, columns=state_labels).to_csv(csvfilename + ".csv")
    return major_complex, 


####################################################################################################

####See results below


print("All the values below are generated for congruent replay")
theta = 3
print("Threshold set to", theta)
artificial_subject_phi(light = "no",     threshold=theta, replay = "no replay")
artificial_subject_phi(light = "red",    threshold=theta, replay = "no replay")
artificial_subject_phi(light = "blue",   threshold=theta, replay = "no replay")
artificial_subject_phi(light = "green",  threshold=theta, replay = "no replay")

#feedforward replay
artificial_subject_phi(light = "no",     threshold=theta, replay = "ff replay")
artificial_subject_phi(light = "red",    threshold=theta, replay = "ff replay")
artificial_subject_phi(light = "blue",   threshold=theta, replay = "ff replay")
artificial_subject_phi(light = "green",  threshold=theta, replay = "ff replay")

#feedback replay
artificial_subject_phi(light = "no",     threshold=theta, replay = "fb replay")
artificial_subject_phi(light = "red",    threshold=theta, replay = "fb replay")
artificial_subject_phi(light = "blue",   threshold=theta, replay = "fb replay")
artificial_subject_phi(light = "green",  threshold=theta, replay = "fb replay")


theta = 2
print("Threshold set to", theta)
artificial_subject_phi(light = "no",     threshold=theta, replay = "no replay")
artificial_subject_phi(light = "red",    threshold=theta, replay = "no replay")
artificial_subject_phi(light = "blue",   threshold=theta, replay = "no replay")
artificial_subject_phi(light = "green",  threshold=theta, replay = "no replay")


#feedforward replay
artificial_subject_phi(light = "no",     threshold=theta, replay = "ff replay")
artificial_subject_phi(light = "red",    threshold=theta, replay = "ff replay")
artificial_subject_phi(light = "blue",   threshold=theta, replay = "ff replay")
artificial_subject_phi(light = "green",  threshold=theta, replay = "ff replay")

#feedback replay
artificial_subject_phi(light = "no",     threshold=theta, replay = "fb replay")
artificial_subject_phi(light = "red",    threshold=theta, replay = "fb replay")
artificial_subject_phi(light = "blue",   threshold=theta, replay = "fb replay")
artificial_subject_phi(light = "green",  threshold=theta, replay = "fb replay")



#RESULTS for congruent replay:
#################
#   blue=cortical cell A fires
#   red= cortical cell B fires
#       for threshold = 2
#   green= cortical cells A,B,C,D,E fire which creates all or non 
#       network as a whole for green and therefore Φ = 0
#       for threshold = 3
#   no(light)=no cell recieves input
#   threshold=minimals number of inputs required to activate to activate a mechanism.

# without replay
#------------------
# big Φ for Subsystem(A, B, C, D, E) no replay and with no light stimulus is 0.056123 (thereshold=3)
# big Φ for Subsystem(A, C, D, E) no replay and with red light stimulus is 0.051021 (thereshold=3)
# big Φ for Subsystem(B, C, D, E) no replay and with blue light stimulus is 0.051021 (thereshold=3)
# big Φ for Subsystem(C, D, E) no replay and with green light stimulus is 0.215278 (thereshold=3)


# with feedforward replay
#------------------
# big Φ for Subsystem() ff replay and with no light stimulus is 0.0 (thereshold=3)
# big Φ for Subsystem() ff replay and with red light stimulus is 0.0 (thereshold=3)
# big Φ for Subsystem() ff replay and with blue light stimulus is 0.0 (thereshold=3)
# big Φ for Subsystem() ff replay and with green light stimulus is 0.0 (thereshold=3)

# with feedback replay
#------------------
# big Φ for Subsystem() fb replay and with no light stimulus is 0.0 (thereshold=3)
# big Φ for Subsystem() fb replay and with red light stimulus is 0.0 (thereshold=3)
# big Φ for Subsystem() fb replay and with blue light stimulus is 0.0 (thereshold=3)
# big Φ for Subsystem() fb replay and with green light stimulus is 0.0 (thereshold=3)



# without replay
#------------------
# big Φ for Subsystem(A, B, C, D, E) no replay and with no light stimulus is 10.729488 (thereshold=2)
# big Φ for Subsystem(A, D) no replay and with red light stimulus is 1.0 (thereshold=2)
# big Φ for Subsystem(C, E) no replay and with blue light stimulus is 1.0 (thereshold=2)
# big Φ for Subsystem() no replay and with green light stimulus is 0.0 (thereshold=2)

# with feedforward replay
#------------------
# big Φ for Subsystem() ff replay and with no light stimulus is 0.0 (thereshold=2)
# big Φ for Subsystem() ff replay and with red light stimulus is 0.0 (thereshold=2)
# big Φ for Subsystem() ff replay and with blue light stimulus is 0.0 (thereshold=2)
# big Φ for Subsystem() ff replay and with green light stimulus is 0.0 (thereshold=2)

# with feedback replay
#------------------
# big Φ for Subsystem() fb replay and with no light stimulus is 0.0 (thereshold=2)
# big Φ for Subsystem() fb replay and with red light stimulus is 0.0 (thereshold=2)
# big Φ for Subsystem() fb replay and with blue light stimulus is 0.0 (thereshold=2)
# big Φ for Subsystem() fb replay and with green light stimulus is 0.0 (thereshold=2)