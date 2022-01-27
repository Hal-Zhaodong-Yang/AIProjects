import numpy as np
import operator


def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous 
        distribution should be zero, instead of the number outputed here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x. 
        The proability of getting value "x" is zero bcause a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if para_tuple == (None, None):
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile 


def gp(x, para_tuple):
    if para_tuple == (None, None):
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile 

def part_1_a():
    """Provide probabilities for the word HMMs outlined below.

    Word BUY, CAR, and HOUSE.

    Review Udacity Lesson 8 - Video #29. HMM Training

    Returns:
        tuple() of
        (prior probabilities for all states for word BUY,
         transition probabilities between states for word BUY,
         emission parameters tuple(mean, std) for all states for word BUY,
         prior probabilities for all states for word CAR,
         transition probabilities between states for word CAR,
         emission parameters tuple(mean, std) for all states for word CAR,
         prior probabilities for all states for word HOUSE,
         transition probabilities between states for word HOUSE,
         emission parameters tuple(mean, std) for all states for word HOUSE,)


        Sample Format (not complete):
        (
            {'B1': prob_of_starting_in_B1, 'B2': prob_of_starting_in_B2, ...},
            {'B1': {'B1': prob_of_transition_from_B1_to_B1,
                    'B2': prob_of_transition_from_B1_to_B2,
                    'B3': prob_of_transition_from_B1_to_B3,
                    'Bend': prob_of_transition_from_B1_to_Bend},
             'B2': {...}, ...},
            {'B1': tuple(mean_of_B1, standard_deviation_of_B1),
             'B2': tuple(mean_of_B2, standard_deviation_of_B2), ...},
            {'C1': prob_of_starting_in_C1, 'C2': prob_of_starting_in_C2, ...},
            {'C1': {'C1': prob_of_transition_from_C1_to_C1,
                    'C2': prob_of_transition_from_C1_to_C2,
                    'C3': prob_of_transition_from_C1_to_C3,
                    'Cend': prob_of_transition_from_C1_to_Cend},
             'C2': {...}, ...}
            {'C1': tuple(mean_of_C1, standard_deviation_of_C1),
             'C2': tuple(mean_of_C2, standard_deviation_of_C2), ...}
            {'H1': prob_of_starting_in_H1, 'H2': prob_of_starting_in_H2, ...},
            {'H1': {'H1': prob_of_transition_from_H1_to_H1,
                    'H2': prob_of_transition_from_H1_to_H2,
                    'H3': prob_of_transition_from_H1_to_H3,
                    'Hend': prob_of_transition_from_H1_to_Hend},
             'H2': {...}, ...}
            {'H1': tuple(mean_of_H1, standard_deviation_of_H1),
             'H2': tuple(mean_of_H2, standard_deviation_of_H2), ...}
        )
    """
    # TODO: complete this function.
    

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.,
        'B2': 0.,
        'B3': 0.,
        'Bend': 0.,
    }
    b_transition_probs = {
        'B1': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 0.},
        'B2': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 0.},
        'B3': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 0.},
        'Bend': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 0.},
    }
    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (None, None),
        'B2': (None, None),
        'B3': (None, None),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 0.},
        'C2': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 0.},
        'C3': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 0.},
        'Cend': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 0.},
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (None, None),
        'C2': (None, None),
        'C3': (None, None),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 0.},
        'H2': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 0.},
        'H3': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 0.},
        'Hend': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 0.},
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (None, None),
        'H2': (None, None),
        'H3': (None, None),
        'Hend': (None, None)
    }

    """data"""
    data_label = [0,0,0,1,1,1,2,2,2]
    word_data = [{'B1':[36, 44],'B2':[52, 56],'B3':[49, 44]},
                 {'B1':[42, 46, 54],'B2':[62, 68, 65],'B3':[60, 56]},
                 {'B1':[42, 40, 41],'B2':[43, 52, 55],'B3':[59, 60, 55, 47]},
                 {'C1':[47, 39, 32],'C2':[34, 36, 42],'C3':[42, 42, 34, 25]},
                 {'C1':[35, 35, 43],'C2':[46, 52, 52],'C3':[56, 49, 45]},
                 {'C1':[28, 35, 46],'C2':[46, 48, 43],'C3':[43, 40]},
                 {'H1':[37, 36, 32, 26, 26],'H2':[25, 23, 22, 21, 39],'H3':[48, 60, 70, 74, 77]},
                 {'H1':[50, 50, 49, 47, 39],'H2':[39, 38, 38, 50, 56],'H3':[61, 67, 67, 67, 67]},
                 {'H1':[45, 43, 44, 43, 40],'H2':[35, 36, 37, 39, 45],'H3':[60, 68, 66, 72, 72, 75]}]
    boundary_b = [['B1','B2'],['B2','B3']]
    boundary_c = [['C1','C2'],['C2','C3']]
    boundary_h = [['H1','H2'],['H2','H3']]
    
    
    """word buy initialization"""
    b_prior_probs['B1'] = np.round(3.0 / 9.0,3)
    for key in b_emission_paras:
        if key == 'Bend':
            continue
        e_mean = np.mean(word_data[0][key] + word_data[1][key] + word_data[2][key])
        e_std = np.std(word_data[0][key] + word_data[1][key] + word_data[2][key])
        b_emission_paras[key] = (e_mean,e_std)

    converged = False
    while not converged:
        converged = True
        for boundary in boundary_b:
            for i in range(3):
                left_left_ratio = abs(word_data[i][boundary[0]][-1] - b_emission_paras[boundary[0]][0]) / b_emission_paras[boundary[0]][1]
                left_right_ratio = abs(word_data[i][boundary[0]][-1] - b_emission_paras[boundary[1]][0]) / b_emission_paras[boundary[1]][1]
                right_left_ratio = abs(word_data[i][boundary[1]][0] - b_emission_paras[boundary[0]][0]) / b_emission_paras[boundary[0]][1]
                right_right_ratio = abs(word_data[i][boundary[1]][0] - b_emission_paras[boundary[1]][0]) / b_emission_paras[boundary[1]][1]
                if left_left_ratio > left_right_ratio:
                    if right_left_ratio < right_right_ratio:
                        if right_left_ratio < left_right_ratio:
                            if len(word_data[i][boundary[1]]) > 1:
                                word_data[i][boundary[0]].append(word_data[i][boundary[1]][0])
                                del word_data[i][boundary[1]][0]
                                converged = False
                        else:
                            if len(word_data[i][boundary[0]]) > 1:
                                word_data[i][boundary[1]].insert(0,word_data[i][boundary[0]][-1])
                                del word_data[i][boundary[0]][-1]
                                converged = False
                    else:
                        if len(word_data[i][boundary[0]]) > 1:
                            word_data[i][boundary[1]].insert(0,word_data[i][boundary[0]][-1])
                            del word_data[i][boundary[0]][-1]
                            converged = False
                elif right_left_ratio < right_right_ratio:
                    if len(word_data[i][boundary[1]]) > 1:
                        word_data[i][boundary[0]].append(word_data[i][boundary[1]][0])
                        del word_data[i][boundary[1]][0]
                        converged = False
        for key in b_emission_paras:
            if key == 'Bend':
                continue
            e_mean = np.mean(word_data[0][key] + word_data[1][key] + word_data[2][key])
            e_std = np.std(word_data[0][key] + word_data[1][key] + word_data[2][key])
            b_emission_paras[key] = (e_mean,e_std)
    
    for key in b_emission_paras:
        if key == 'Bend':
            continue
        e_mean = np.round(np.mean(word_data[0][key] + word_data[1][key] + word_data[2][key]),3)
        e_std = np.round(np.std(word_data[0][key] + word_data[1][key] + word_data[2][key]),3)
        b_emission_paras[key] = (e_mean,e_std)
    
    b_transition_probs['B1']['B1'] = np.round((len(word_data[0]['B1']) + len(word_data[1]['B1']) + len(word_data[2]['B1']) - 3) / (len(word_data[0]['B1']) + len(word_data[1]['B1']) + len(word_data[2]['B1'])),3)
    b_transition_probs['B1']['B2'] = 1 - b_transition_probs['B1']['B1']
    b_transition_probs['B2']['B2'] = np.round((len(word_data[0]['B2']) + len(word_data[1]['B2']) + len(word_data[2]['B2']) - 3) / (len(word_data[0]['B2']) + len(word_data[1]['B2']) + len(word_data[2]['B2'])),3)
    b_transition_probs['B2']['B3'] = 1 - b_transition_probs['B2']['B2']
    b_transition_probs['B3']['B3'] = np.round((len(word_data[0]['B3']) + len(word_data[1]['B3']) + len(word_data[2]['B3']) - 3) / (len(word_data[0]['B3']) + len(word_data[1]['B3']) + len(word_data[2]['B3'])),3)
    b_transition_probs['B3']['Bend'] = 1 - b_transition_probs['B3']['B3']
    b_transition_probs['Bend']['Bend'] = 1


    '''Word Car Initialization'''
    c_prior_probs['C1'] = np.round(3.0 / 9.0,3)
    for key in c_emission_paras:
        if key == 'Cend':
            continue
        e_mean = np.mean(word_data[3][key] + word_data[4][key] + word_data[5][key])
        e_std = np.std(word_data[3][key] + word_data[4][key] + word_data[5][key])
        c_emission_paras[key] = (e_mean,e_std)

    converged = False
    while not converged:
        converged = True
        for boundary in boundary_c:
            for i in range(3,6):
                left_left_ratio = abs(word_data[i][boundary[0]][-1] - c_emission_paras[boundary[0]][0]) / c_emission_paras[boundary[0]][1]
                left_right_ratio = abs(word_data[i][boundary[0]][-1] - c_emission_paras[boundary[1]][0]) / c_emission_paras[boundary[1]][1]
                right_left_ratio = abs(word_data[i][boundary[1]][0] - c_emission_paras[boundary[0]][0]) / c_emission_paras[boundary[0]][1]
                right_right_ratio = abs(word_data[i][boundary[1]][0] - c_emission_paras[boundary[1]][0]) / c_emission_paras[boundary[1]][1]
                if left_left_ratio > left_right_ratio:
                    if right_left_ratio < right_right_ratio:
                        if right_left_ratio < left_right_ratio:
                            if len(word_data[i][boundary[1]]) > 1:
                                word_data[i][boundary[0]].append(word_data[i][boundary[1]][0])
                                del word_data[i][boundary[1]][0]
                                converged = False
                        else:
                            if len(word_data[i][boundary[0]]) > 1:
                                word_data[i][boundary[1]].insert(0,word_data[i][boundary[0]][-1])
                                del word_data[i][boundary[0]][-1]
                                converged = False
                    else:
                        if len(word_data[i][boundary[0]]) > 1:
                            word_data[i][boundary[1]].insert(0,word_data[i][boundary[0]][-1])
                            del word_data[i][boundary[0]][-1]
                            converged = False
                elif right_left_ratio < right_right_ratio:
                    if len(word_data[i][boundary[1]]) > 1:
                        word_data[i][boundary[0]].append(word_data[i][boundary[1]][0])
                        del word_data[i][boundary[1]][0]
                        converged = False
        for key in c_emission_paras:
            if key == 'Cend':
                continue
            e_mean = np.mean(word_data[3][key] + word_data[4][key] + word_data[5][key])
            e_std = np.std(word_data[3][key] + word_data[4][key] + word_data[5][key])
            c_emission_paras[key] = (e_mean,e_std)
        
    for key in c_emission_paras:
        if key == 'Cend':
            continue
        e_mean = np.round(np.mean(word_data[3][key] + word_data[4][key] + word_data[5][key]),3)
        e_std = np.round(np.std(word_data[3][key] + word_data[4][key] + word_data[5][key]),3)
        c_emission_paras[key] = (e_mean,e_std)
    
    c_transition_probs['C1']['C1'] = np.round((len(word_data[3]['C1']) + len(word_data[4]['C1']) + len(word_data[5]['C1']) - 3) / (len(word_data[3]['C1']) + len(word_data[4]['C1']) + len(word_data[5]['C1'])),3)
    c_transition_probs['C1']['C2'] = 1 - c_transition_probs['C1']['C1']
    c_transition_probs['C2']['C2'] = np.round((len(word_data[3]['C2']) + len(word_data[4]['C2']) + len(word_data[5]['C2']) - 3) / (len(word_data[3]['C2']) + len(word_data[4]['C2']) + len(word_data[5]['C2'])),3)
    c_transition_probs['C2']['C3'] = 1 - c_transition_probs['C2']['C2']
    c_transition_probs['C3']['C3'] = np.round((len(word_data[3]['C3']) + len(word_data[4]['C3']) + len(word_data[5]['C3']) - 3) / (len(word_data[3]['C3']) + len(word_data[4]['C3']) + len(word_data[5]['C3'])),3)
    c_transition_probs['C3']['Cend'] = 1 - c_transition_probs['C3']['C3']
    c_transition_probs['Cend']['Cend'] = 1

    """Word House Initialization"""
    h_prior_probs['H1'] = np.round(3.0 / 9.0,3)
    for key in h_emission_paras:
        if key == 'Hend':
            continue
        e_mean = np.mean(word_data[6][key] + word_data[7][key] + word_data[8][key])
        e_std = np.std(word_data[6][key] + word_data[7][key] + word_data[8][key])
        h_emission_paras[key] = (e_mean,e_std)

    converged = False
    while not converged:
        converged = True
        for boundary in boundary_h:
            for i in range(6,9):
                left_left_ratio = abs(word_data[i][boundary[0]][-1] - h_emission_paras[boundary[0]][0]) / h_emission_paras[boundary[0]][1]
                left_right_ratio = abs(word_data[i][boundary[0]][-1] - h_emission_paras[boundary[1]][0]) / h_emission_paras[boundary[1]][1]
                right_left_ratio = abs(word_data[i][boundary[1]][0] - h_emission_paras[boundary[0]][0]) / h_emission_paras[boundary[0]][1]
                right_right_ratio = abs(word_data[i][boundary[1]][0] - h_emission_paras[boundary[1]][0]) / h_emission_paras[boundary[1]][1]
                if left_left_ratio > left_right_ratio:
                    if right_left_ratio < right_right_ratio:
                        if right_left_ratio < left_right_ratio:
                            if len(word_data[i][boundary[1]]) > 1:
                                word_data[i][boundary[0]].append(word_data[i][boundary[1]][0])
                                del word_data[i][boundary[1]][0]
                                converged = False
                        else:
                            if len(word_data[i][boundary[0]]) > 1:
                                word_data[i][boundary[1]].insert(0,word_data[i][boundary[0]][-1])
                                del word_data[i][boundary[0]][-1]
                                converged = False
                    else:
                        if len(word_data[i][boundary[0]]) > 1:
                            word_data[i][boundary[1]].insert(0,word_data[i][boundary[0]][-1])
                            del word_data[i][boundary[0]][-1]
                            converged = False
                elif right_left_ratio < right_right_ratio:
                    if len(word_data[i][boundary[1]]) > 1:
                        word_data[i][boundary[0]].append(word_data[i][boundary[1]][0])
                        del word_data[i][boundary[1]][0]
                        converged = False
        for key in h_emission_paras:
            if key == 'Hend':
                continue
            e_mean = np.mean(word_data[6][key] + word_data[7][key] + word_data[8][key])
            e_std = np.std(word_data[6][key] + word_data[7][key] + word_data[8][key])
            h_emission_paras[key] = (e_mean,e_std)

    for key in h_emission_paras:
        if key == 'Hend':
            continue
        e_mean = np.round(np.mean(word_data[6][key] + word_data[7][key] + word_data[8][key]),3)
        e_std = np.round(np.std(word_data[6][key] + word_data[7][key] + word_data[8][key]),3)
        h_emission_paras[key] = (e_mean,e_std)
    
    h_transition_probs['H1']['H1'] = np.round((len(word_data[6]['H1']) + len(word_data[7]['H1']) + len(word_data[8]['H1']) - 3) / (len(word_data[6]['H1']) + len(word_data[7]['H1']) + len(word_data[8]['H1'])),3)
    h_transition_probs['H1']['H2'] = 1 - h_transition_probs['H1']['H1']
    h_transition_probs['H2']['H2'] = np.round((len(word_data[6]['H2']) + len(word_data[7]['H2']) + len(word_data[8]['H2']) - 3) / (len(word_data[6]['H2']) + len(word_data[7]['H2']) + len(word_data[8]['H2'])),3)
    h_transition_probs['H2']['H3'] = 1 - h_transition_probs['H2']['H2']
    h_transition_probs['H3']['H3'] = np.round((len(word_data[6]['H3']) + len(word_data[7]['H3']) + len(word_data[8]['H3']) - 3) / (len(word_data[6]['H3']) + len(word_data[7]['H3']) + len(word_data[8]['H3'])),3)
    h_transition_probs['H3']['Hend'] = 1 - h_transition_probs['H3']['H3']
    h_transition_probs['Hend']['Hend'] = 1


    




    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.

    Args:
        evidence_vector (list): List of right hand Y-axis positions (interger).

        states (list): List of all states in a word. No transition between words.
                       example: ['B1', 'B2', 'B3', 'Bend']

        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}

        transition_probs (dict): dictionary representing transitions from each
                                 state to every other state.

        emission_paras (dict): parameters of Gaussian distribution 
                                from each state.

    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )

    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.

    """
    
    # TODO: complete this function.
    
    if evidence_vector == []:
        return [], 0

    sequence = []
    probability = 0.0
    viterbi_trellis = dict(prior_probs)
    
    for key in viterbi_trellis:
        viterbi_trellis[key] = []
    for i in range(len(evidence_vector)):
        if i == 0:
            for key in viterbi_trellis:
                viterbi_trellis[key].append([prior_probs[key] * gaussian_prob(evidence_vector[0],emission_paras[key]), "start"])
            continue
        for key in viterbi_trellis:
            viterbi_trellis[key].append([])
        for key in viterbi_trellis:
            for child in transition_probs[key]:
                viterbi_trellis[child][i].append([viterbi_trellis[key][i - 1][0] * transition_probs[key][child] * gaussian_prob(evidence_vector[i],emission_paras[child]),key])
        for key in viterbi_trellis:
            max_prob_node = viterbi_trellis[key][i][0]
            for node in viterbi_trellis[key][i]:
                if node[0] > max_prob_node[0]:
                    max_prob_node = node
            viterbi_trellis[key][i] = max_prob_node
    
    max_prob = 0
    for key in viterbi_trellis:
        print(viterbi_trellis)
        if viterbi_trellis[key][-1][0] > max_prob:
            max_prob_key = key
            max_prob = viterbi_trellis[key][-1][0]

    if max_prob == 0:
        return max_prob_node, max_prob

    probability = max_prob
    sequence.append(max_prob_key)
    for i in range(len(evidence_vector) - 1, 0, -1):
        sequence.append(viterbi_trellis[sequence[-1]][i][1])
    sequence.reverse()


    return sequence, probability


def part_2_a():
    """Provide probabilities for the word HMMs outlined below.

    Now, at each time frame you are given with 2 observations (right hand Y
    position & left hand Y position). Use the result you derived in
    part_1_a, accompany with the provided probability for left hand, create
    a tuple of (right-y, left-y) to represent high-dimention transition & 
    emission probabilities.
    """

    # TODO: complete this function.
    

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.,
        'B2': 0.,
        'B3': 0.,
        'Bend': 0.,
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0., 0.7), 'B2': (0., 0.3), 'B3': (0., 0.), 'Bend': (0., 0.)},
        'B2': {'B1': (0., 0.), 'B2': (0., 0.05), 'B3': (0., 0.95), 'Bend': (0., 0.)},
        'B3': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.727), 'Bend': (0., 0.273)},
        'Bend': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (0., 1.)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(None, None), (108.200, 17.314)],
        'B2': [(None, None), (78.670, 1.886)],
        'B3': [(None, None), (64.182, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': (0., 0.7), 'C2': (0., 0.3), 'C3': (0., 0.), 'Cend': (0., 0.)},
        'C2': {'C1': (0., 0.), 'C2': (0., 0.625), 'C3': (0., 0.375), 'Cend': (0., 0.)},
        'C3': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.625), 'Cend': (0., 0.375)},
        'Cend': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (0., 1.)},
    }
    c_emission_paras = {
        'C1': [(None, None), (56.300, 10.659)],
        'C2': [(None, None), (37.110, 4.306)],
        'C3': [(None, None), (50.000, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    h_transition_probs = {
        'H1': {'H1': (0., 0.7), 'H2': (0., 0.3), 'H3': (0., 0.), 'Hend': (0., 0.)},
        'H2': {'H1': (0., 0.), 'H2': (0., 0.842), 'H3': (0., 0.158), 'Hend': (0., 0.)},
        'H3': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.824), 'Hend': (0., 0.176)},
        'Hend': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (0., 1.)},
    }
    h_emission_paras = {
        'H1': [(None, None), (53.600, 7.392)],
        'H2': [(None, None), (37.168, 8.875)],
        'H3': [(None, None), (74.176, 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    b_prior_probs, right_b_transition_probs, right_b_emission_paras,\
        c_prior_probs, right_c_transition_probs, right_c_emission_paras,\
            h_prior_probs, right_h_transition_probs, right_h_emission_paras = part_1_a()

    for key in b_transition_probs:
        for sub_key in b_transition_probs[key]:
            b_transition_probs[key][sub_key] = (right_b_transition_probs[key][sub_key], b_transition_probs[key][sub_key][1])
    
    b_transition_probs['B3'].update({'C1':(np.round(b_transition_probs['B3']['Bend'][0] / 3,3), np.round(b_transition_probs['B3']['Bend'][1] / 3,3)),
    'H1':(np.round(b_transition_probs['B3']['Bend'][0] / 3,3), np.round(b_transition_probs['B3']['Bend'][1] / 3,3))})
    b_transition_probs['B3']['Bend'] = (np.round(b_transition_probs['B3']['Bend'][0] / 3,3), np.round(b_transition_probs['B3']['Bend'][1] / 3,3))

    for key in b_emission_paras:
        b_emission_paras[key][0] = right_b_emission_paras[key]

    

    for key in c_transition_probs:
        for sub_key in c_transition_probs[key]:
            c_transition_probs[key][sub_key] = (right_c_transition_probs[key][sub_key], c_transition_probs[key][sub_key][1])
    
    c_transition_probs['C3'].update({'B1':(np.round(c_transition_probs['C3']['Cend'][0] / 3,3), np.round(c_transition_probs['C3']['Cend'][1] / 3,3)),
    'H1':(np.round(c_transition_probs['C3']['Cend'][0] / 3,3), np.round(c_transition_probs['C3']['Cend'][1] / 3,3))})
    c_transition_probs['C3']['Cend'] = (np.round(c_transition_probs['C3']['Cend'][0] / 3,3), np.round(c_transition_probs['C3']['Cend'][1] / 3,3))
    
    for key in c_emission_paras:
        c_emission_paras[key][0] = right_c_emission_paras[key]
    


    for key in h_transition_probs:
        for sub_key in h_transition_probs[key]:
            h_transition_probs[key][sub_key] = (right_h_transition_probs[key][sub_key], h_transition_probs[key][sub_key][1])
    
    h_transition_probs['H3'].update({'C1':(np.round(h_transition_probs['H3']['Hend'][0] / 3,3), np.round(h_transition_probs['H3']['Hend'][1] / 3,3)),
    'B1':(np.round(h_transition_probs['H3']['Hend'][0] / 3,3), np.round(h_transition_probs['H3']['Hend'][1] / 3,3))})
    h_transition_probs['H3']['Hend'] = (np.round(h_transition_probs['H3']['Hend'][0] / 3,3), np.round(h_transition_probs['H3']['Hend'][1] / 3,3))
    
    for key in h_emission_paras:
        h_emission_paras[key][0] = right_h_emission_paras[key]
    
    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.

    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    """
    # TODO: complete this function.
    if evidence_vector == []:
        return [], 0

    sequence = []
    probability = 0.0
    viterbi_trellis = dict(prior_probs)
    
    for key in viterbi_trellis:
        viterbi_trellis[key] = []
    for i in range(len(evidence_vector)):
        if i == 0:
            for key in viterbi_trellis:
                viterbi_trellis[key].append([prior_probs[key] * gaussian_prob(evidence_vector[0][0],emission_paras[key][0]) * gaussian_prob(evidence_vector[0][1], emission_paras[key][1]), "start"])
            continue
        for key in viterbi_trellis:
            viterbi_trellis[key].append([])
        for key in viterbi_trellis:
            for child in transition_probs[key]:
                viterbi_trellis[child][i].append([viterbi_trellis[key][i - 1][0] * transition_probs[key][child][0] * transition_probs[key][child][1] * gaussian_prob(evidence_vector[i][0],emission_paras[child][0]) * gaussian_prob(evidence_vector[i][1],emission_paras[child][1]),key])
        for key in viterbi_trellis:
            max_prob_node = viterbi_trellis[key][i][0]
            for node in viterbi_trellis[key][i]:
                if node[0] > max_prob_node[0]:
                    max_prob_node = node
            viterbi_trellis[key][i] = max_prob_node
    
    max_prob = 0
    for key in viterbi_trellis:
        print(viterbi_trellis)
        if viterbi_trellis[key][-1][0] > max_prob:
            max_prob_key = key
            max_prob = viterbi_trellis[key][-1][0]

    if max_prob == 0:
        return max_prob_node, max_prob

    probability = max_prob
    sequence.append(max_prob_key)
    for i in range(len(evidence_vector) - 1, 0, -1):
        sequence.append(viterbi_trellis[sequence[-1]][i][1])
    sequence.reverse()

    return sequence, probability


def return_your_name():
    """Return your name
    """
    # TODO: finish this
    return "Zhaodong Yang"
