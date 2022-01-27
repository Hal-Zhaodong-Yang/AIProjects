import sys
from networkx.generators.community import gaussian_random_partition_graph
from pgmpy import base

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
import random
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function    
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")
    BayesNet.add_edge("temperature", "faulty gauge")
    BayesNet.add_edge("temperature", "gauge")
    BayesNet.add_edge("faulty gauge", "gauge")
    BayesNet.add_edge("gauge", "alarm")
    BayesNet.add_edge("faulty alarm", "alarm")
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    
    cpd_a = TabularCPD("temperature", 2, values = [[0.8], [0.2]])
    cpd_fgt = TabularCPD("faulty gauge", 2, values = [[0.95, 0.2], [0.05, 0.8]], evidence = ["temperature"], evidence_card = [2])
    cpd_gfgt = TabularCPD("gauge", 2, values = [[0.95, 0.05, 0.2, 0.8],\
        [0.05, 0.95, 0.8, 0.2]], evidence = ["faulty gauge", "temperature"], evidence_card=[2, 2])
    cpd_afag = TabularCPD("alarm", 2, values = [[0.9, 0.1, 0.55, 0.45],\
        [0.1, 0.9, 0.45, 0.55]], evidence = ["faulty alarm", "gauge"], evidence_card = [2, 2])
    cpd_fa = TabularCPD("faulty alarm", 2, values = [[0.85], [0.15]])
    bayes_net.add_cpds(cpd_a, cpd_fgt, cpd_gfgt, cpd_afag, cpd_fa)
    return bayes_net

def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=["alarm"], joint = False)
    alarm_prob = (marginal_prob["alarm"].values)[1]
    return alarm_prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=["gauge"], joint = False)
    gauge_prob = (marginal_prob["gauge"].values)[1]
    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=["temperature"], evidence = {"alarm":1, "faulty alarm":0, "faulty gauge":0}, joint = False)
    temp_prob = (conditional_prob["temperature"].values)[1]
    return temp_prob

'''
def test_prob():
    power_plant = set_probability(make_power_plant_net())
    alarm = get_alarm_prob(power_plant)
    gauge = get_gauge_prob(power_plant)
    temp = get_temperature_prob(power_plant)
    print("alarm", alarm)
    print("gauge", gauge)
    print("temperature", temp)
'''

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")
    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("A", "CvA")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "CvA")
    BayesNet.add_edge("C", "BvC")
    cpd_a = TabularCPD("A", 4, values = [[0.15], [0.45], [0.3], [0.1]])
    cpd_b = TabularCPD("B", 4, values = [[0.15], [0.45], [0.3], [0.1]])
    cpd_c = TabularCPD("C", 4, values = [[0.15], [0.45], [0.3], [0.1]])
    cpd_avbab = TabularCPD("AvB", 3, values = [[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10],
                                            [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                            [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]],
                                             evidence = ["A", "B"], evidence_card = [4, 4])
    cpd_bvcbc = TabularCPD("BvC", 3, values = [[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10],
                                            [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                            [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]],
                                             evidence = ["B", "C"], evidence_card = [4, 4])
    cpd_cvaca = TabularCPD("CvA", 3, values = [[0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10],
                                            [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
                                            [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]],
                                             evidence = ["C", "A"], evidence_card = [4, 4])
    BayesNet.add_cpds(cpd_a, cpd_b, cpd_c, cpd_avbab, cpd_bvcbc, cpd_cvaca)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables = ["BvC"], evidence = {"AvB":0, "CvA":2},  joint = False)
    posterior = conditional_prob["BvC"].values
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)
    # TODO: finish this function
    initial_state = list(initial_state)
    skills = [0, 1, 2, 3]
    match = [0, 1, 2]
    if initial_state == []:
        initial_state = [random.choice(skills), random.choice(skills), random.choice(skills), 0, random.choice(match), 2]
        sample = tuple(initial_state)
    else:
        index = random.choice(["A", "B", "C", "BvC"])
        a_cpd = bayes_net.get_cpds("A").values
        b_cpd = bayes_net.get_cpds("B").values
        c_cpd = bayes_net.get_cpds("C").values
        avb_cpd = bayes_net.get_cpds("AvB").values
        bvc_cpd = bayes_net.get_cpds("BvC").values
        cva_cpd = bayes_net.get_cpds("CvA").values
        if index == "A":
            jp_a = [0,0,0,0]
            for i in range(4):
                jp_a[i] = a_cpd[i] * b_cpd[initial_state[1]] * c_cpd[initial_state[2]] * \
                    avb_cpd[initial_state[3], i, initial_state[1]] * bvc_cpd[initial_state[4], initial_state[1], initial_state[2]] * \
                        cva_cpd[initial_state[5], initial_state[2], i]
            cp_a = []
            for i in range(4):
                cp_a.append(jp_a[i] / sum(jp_a))
            sample_choice = random.random()
            choice = 0
            while sample_choice - cp_a[choice] > 0:
                sample_choice -= cp_a[choice]
                choice += 1
            initial_state[0] = choice
            sample = tuple(initial_state)
        if index == "B":
            jp_b = [0,0,0,0]
            for i in range(4):
                jp_b[i] = a_cpd[initial_state[0]] * b_cpd[i] * c_cpd[initial_state[2]] * \
                    avb_cpd[initial_state[3], initial_state[0], i] * bvc_cpd[initial_state[4], i, initial_state[2]] * \
                        cva_cpd[initial_state[5], initial_state[2], initial_state[0]]
            cp_b = []
            for i in range(4):
                cp_b.append(jp_b[i] / sum(jp_b))
            sample_choice = random.random()
            choice = 0
            while sample_choice - cp_b[choice] > 0:
                sample_choice -= cp_b[choice]
                choice += 1
            initial_state[1] = choice
            sample = tuple(initial_state)
        if index == "C":
            jp_c = [0,0,0,0]
            for i in range(4):
                jp_c[i] = a_cpd[initial_state[0]] * b_cpd[initial_state[1]] * c_cpd[i] * \
                    avb_cpd[initial_state[3], initial_state[0], initial_state[1]] * bvc_cpd[initial_state[4], initial_state[1], i] * \
                        cva_cpd[initial_state[5], i, initial_state[0]]
            cp_c = []
            for i in range(4):
                cp_c.append(jp_c[i] / sum(jp_c))
            sample_choice = random.random()
            choice = 0
            while sample_choice - cp_c[choice] > 0:
                sample_choice -= cp_c[choice]
                choice += 1
            initial_state[2] = choice
            sample = tuple(initial_state)
        if index == "BvC":
            jp_bvc = [0,0,0]
            for i in range(3):
                jp_bvc[i] = a_cpd[initial_state[0]] * b_cpd[initial_state[1]] * c_cpd[initial_state[2]] * \
                    avb_cpd[initial_state[3], initial_state[0], initial_state[1]] * bvc_cpd[i, initial_state[1], initial_state[2]] * \
                        cva_cpd[initial_state[5], initial_state[2], initial_state[0]]
            cp_bvc = []
            for i in range(3):
                cp_bvc.append(jp_bvc[i] / sum(jp_bvc))
            sample_choice = random.random()
            choice = 0
            while sample_choice - cp_bvc[choice] > 0:
                sample_choice -= cp_bvc[choice]
                choice += 1
            initial_state[4] = choice
            sample = tuple(initial_state)

    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = tuple(initial_state)    
    # TODO: finish this function
    initial_state = list(initial_state)
    a_cpd = bayes_net.get_cpds("A").values
    b_cpd = bayes_net.get_cpds("B").values
    c_cpd = bayes_net.get_cpds("C").values
    avb_cpd = bayes_net.get_cpds("AvB").values
    bvc_cpd = bayes_net.get_cpds("BvC").values
    cva_cpd = bayes_net.get_cpds("CvA").values
    skills = [0, 1, 2, 3]
    match = [0, 1, 2]
    uniform_pd = [ -1, 0, 1]
    if initial_state == []:
        initial_state = [random.choice(skills), random.choice(skills), random.choice(skills), 0, random.choice(match), 2]
        #print("init mhsample", initial_state)
        sample = tuple(initial_state)
    else:
        x_cand = list(initial_state)
        x_cand[0] = initial_state[0] + random.choice(uniform_pd)
        x_cand[1] = initial_state[1] + random.choice(uniform_pd)
        x_cand[2] = initial_state[2] + random.choice(uniform_pd)
        x_cand[4] = initial_state[4] + random.choice(uniform_pd)
        if x_cand[0] < 0 or x_cand[0] > 3 or x_cand[1] < 0 or x_cand[1] > 3 or x_cand[2] < 0 or x_cand[2] > 3 or x_cand[4] < 0 or x_cand[4] > 2:
            sample = tuple(initial_state)
            #print("did detect out of range",sample)
        else:
            jp_cand = a_cpd[x_cand[0]] * b_cpd[x_cand[1]] * c_cpd[x_cand[2]] * \
                    avb_cpd[x_cand[3], x_cand[0], x_cand[1]] * bvc_cpd[x_cand[4], x_cand[1], x_cand[2]] * \
                        cva_cpd[x_cand[5], x_cand[2], x_cand[0]]
            jp_init = a_cpd[initial_state[0]] * b_cpd[initial_state[1]] * c_cpd[initial_state[2]] * \
                    avb_cpd[initial_state[3], initial_state[0], initial_state[1]] * bvc_cpd[initial_state[4], initial_state[1], initial_state[2]] * \
                        cva_cpd[initial_state[5], initial_state[2], initial_state[0]]
            alpha = min(1, jp_cand / jp_init)
            u = random.random()
            if u < alpha:
                sample = tuple(x_cand)
            else:
                sample = tuple(initial_state)
        
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    Gibbs_statis = [0, 0, 0]
    MH_statis = [0, 0, 0]
    conv_count = 0
    burnin = 2000

    #initialize Gibbs
    conv_n = 1200
    delta = 0.000015
    gibbs_sample = Gibbs_sampler(bayes_net, initial_state)
    Gibbs_count += 1
    Gibbs_statis[gibbs_sample[4]] += 1
    for i in range(3):
        Gibbs_convergence[i] = Gibbs_statis[i] / sum(Gibbs_statis)
    while conv_count <= conv_n:
        prev_conv = list(Gibbs_convergence)
        gibbs_sample = Gibbs_sampler(bayes_net, gibbs_sample)
        Gibbs_count += 1
        if Gibbs_count < burnin:
            continue
        Gibbs_statis[gibbs_sample[4]] += 1
        for i in range(3):
            Gibbs_convergence[i] = Gibbs_statis[i] / sum(Gibbs_statis)
        if (abs(Gibbs_convergence[0] - prev_conv[0]) + abs(Gibbs_convergence[1] - prev_conv[1]) + abs(Gibbs_convergence[2] - prev_conv[2])) / 3 < delta:
            conv_count += 1
        else:
            conv_count = 0
    
    #initialize MH
    conv_count = 0
    conv_n = 1200
    delta = 0.000015
    flag = 0
    mh_sample = MH_sampler(bayes_net,initial_state)
    MH_count += 1
    MH_statis[mh_sample[4]] += 1
    for i in range(3):
        MH_convergence[i] = MH_statis[i] / sum(MH_statis)
    while conv_count <= conv_n:
        prev_conv = list(MH_convergence)
        prev_sample = tuple(mh_sample)
        mh_sample = MH_sampler(bayes_net, mh_sample)
        if flag == 0:
            flag = 1
            continue
        MH_count += 1
        if MH_count < burnin:
            continue
        if prev_sample == mh_sample:
            MH_rejection_count += 1
        else:
            #print("mh_sample", mh_sample)
            MH_statis[mh_sample[4]] += 1
            for i in range(3):
                MH_convergence[i] = MH_statis[i] / sum(MH_statis)
            if (abs(MH_convergence[0] - prev_conv[0]) + abs(MH_convergence[1] - prev_conv[1]) + abs(MH_convergence[2] - prev_conv[2])) / 3 < delta:
                conv_count += 1
            else:
                conv_count = 0


        


    return Gibbs_convergence, MH_convergence, Gibbs_count - burnin, MH_count - burnin, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor



    bayes = get_game_network()
    Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count = compare_sampling(bayes, initial_state = [])
    #print("Gibbs_convergence", Gibbs_convergence)
    #print("MH_convergence", MH_convergence)
    #print("Gibbs_count", Gibbs_count)
    #print("MH_count", MH_count)
    #print("MH_rejection_count", MH_rejection_count)



    
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = MH_count / Gibbs_count
    if factor > 1:
        choice = 0
    else:
        choice = 1
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Zhaodong Yang"

'''
def main(args):
    a, b = sampling_question()
    #print("ground truth", [0.25890074, 0.42796763, 0.31313163])
    #print(a, b)

    

if __name__ == '__main__':
    main(sys.argv)
'''
