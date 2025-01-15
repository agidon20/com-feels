
NEURON {
	:NOTE: since this is an interface, there is no POINT_PROCESS name
	RANGE e, i, gmax, g
	RANGE tau_r
	RANGE tau_d
	RANGE factor
	RANGE tsyn,block
	RANGE D,tau_stp,fD :for synaptic depression
	RANGE tag

	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	e = 0			(mV)	
	tau_r = 0	 	(ms) 	
	tau_d = 0		(ms) 
	tau_stp = 1000 	(ms)
	fD = 1 <0,1> :no depression 
	:fF = 1 <0,1> :no fascilitaiton by default
	factor
	gmax
	tsyn
	block = 0
	tag = 0
}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
}

STATE {
	A
	B
	D :simple mechanism for depression
}

FUNCTION get_factor(tau_fast,tau_slow){
	LOCAL tp
	tp = (tau_fast*tau_slow)/(tau_slow - tau_fast) * log(tau_slow/tau_fast)
	get_factor = -exp(-tp/tau_fast) + exp(-tp/tau_slow)
	get_factor = 1/get_factor
}

INITIAL {
	:start fresh
	A = 0
	B = 0
	g = 0
	D = 1
	tsyn = -1e80 			:last time the synapse was active is very long ego
	if(tau_r/tau_d>0.999){
		tau_r = tau_d*0.999
	}
	if(!tau_r || !tau_d){
		printf("User must set tau_r and tau_d (zero by default)")
	}
	factor = get_factor(tau_r,tau_d)
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = gmax * (B - A) * (!block)
	i = g*(v - e)
}


DERIVATIVE state {
	A' = -A/tau_r
	B' = -B/tau_d
	D' = (1 - D)/tau_stp
}


PROCEDURE EPSP(w){
	A = A + w*factor*D
	B = B + w*factor*D
	D = D*fD
	:D = D + fF*(1-D) :for fascilitation .
	tsyn = t
}
