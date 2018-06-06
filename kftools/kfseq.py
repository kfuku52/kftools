import re

def codon2nuc_freqs(codon_freqs=dict(), model=''):
    if ('F1X4' in model):
        nuc_freqs = [{'A':0,'T':0,'C':0,'G':0,}]
        for nuc in nuc_freqs[0].keys():
            for codon in codon_freqs.keys():
                nuc_count = sum([ nuc==c for c in codon ])
                nuc_freqs[0][nuc] += codon_freqs[codon] * nuc_count / 3
    elif ('F3X4' in model):
        nuc_freqs = list()
        for codon_pos in [0,1,2]:
            nuc_freqs.append({'A':0,'T':0,'C':0,'G':0,})
            for nuc in nuc_freqs[codon_pos].keys():
                for codon in codon_freqs.keys():
                    if nuc==codon[codon_pos]:
                        nuc_freqs[codon_pos][nuc] += codon_freqs[codon] / 3
    for i in range(len(nuc_freqs)):
        scale_factor = sum(nuc_freqs[i].values())
        for nuc in nuc_freqs[i].keys():
            nuc_freqs[i][nuc] = nuc_freqs[i][nuc] / scale_factor
    return nuc_freqs

def nuc_freq2theta(nuc_freqs=list()):
    thetas = list()
    for i in range(len(nuc_freqs)):
        theta = nuc_freqs[i]['G'] + nuc_freqs[i]['C']
        theta1 = nuc_freqs[i]['A'] / (nuc_freqs[i]['A']+nuc_freqs[i]['T'])
        theta2 = nuc_freqs[i]['G'] / (nuc_freqs[i]['G']+nuc_freqs[i]['C'])
        thetas.append({'theta':theta, 'theta1':theta1, 'theta2':theta2,})
    return thetas

def get_mapnh_thetas(model, thetas):
    model_frequency = model
    model_frequency = re.sub(r'X4\+.*', 'X4', model_frequency)
    model_frequency = re.sub(r'.*\+F', 'F', model_frequency)
    param = model_frequency+'('
    for i in range(len(thetas)):
        if len(thetas)==1:
            param = param+'Full.theta='+str(thetas[i]['theta'])+','
            param = param+'Full.theta1='+str(thetas[i]['theta1'])+','
            param = param+'Full.theta2='+str(thetas[i]['theta2'])+','
        else:
            param = param+str(i+1)+'_Full.theta='+str(thetas[i]['theta'])+','
            param = param+str(i+1)+'_Full.theta1='+str(thetas[i]['theta1'])+','
            param = param+str(i+1)+'_Full.theta2='+str(thetas[i]['theta2'])+','
    param = re.sub(r',$', ')', param)
    return param

def alignment2nuc_freqs(leaf_name, alignment_file, model):
    entries = open(alignment_file).read().split('>')
    for e in entries:
        if (e.startswith(leaf_name)):
            seq = e[len(leaf_name):]
            seq = re.sub('\n', '', seq)
            break
    if ('F1X4' in model):
        print('F1X4 is not yet implemented')
    elif ('F3X4' in model):
        seq_codons = list()
        for start in [0,1,2]:
            seq_codons.append(seq[start::3])
        nuc_freqs = list()
        for i in range(len(seq_codons)):
            nuc_freqs.append(dict())
            for nuc in ['A','C','G','T']:
                nuc_freqs[i][nuc] = seq_codons[i].count(nuc)/len(seq_codons[i])
    return(nuc_freqs)

def weighted_mean_root_thetas(subroot_thetas, tree, model):
    subroot_branch_lengths = dict()
    for sn in tree.get_children():
        assert sn.name in subroot_thetas.keys()
        subroot_branch_lengths[sn.name] = sn.dist
    total_subroot_branch_length = sum(subroot_branch_lengths.values())
    print('total subroot branch length:', total_subroot_branch_length)
    if ('F1X4' in model):
        print('F1X4 is not yet implemented')
    elif ('F3X4' in model):
        root_thetas = list()
        codon_positions = [0,1,2]
        params = list(subroot_thetas.values())[0][0].keys()
        params = list(params)
        for cp in codon_positions:
            root_thetas.append(dict())
            for param in params:
                min_value = min([ t[cp][param] for t in subroot_thetas.values() ])
                max_value = max([ t[cp][param] for t in subroot_thetas.values() ])
                min_branch_length = None
                max_branch_length = None
                if (min_value == max_value):
                    weighted_mean = min_value
                else:
                    for srk in subroot_thetas.keys():
                        if (subroot_thetas[srk][cp][param]==min_value):
                            min_branch_length = subroot_branch_lengths[srk]
                        elif (subroot_thetas[srk][cp][param]==max_value):
                            max_branch_length = subroot_branch_lengths[srk]
                    weighted_mean = min_value + ((max_value-min_value)*(min_branch_length/(min_branch_length+max_branch_length)))
                print('param =', param, 'codon_position =', cp, 'min =', min_value, 'max =', max_value,
                      'min_bl =', min_branch_length, 'max_bl =', max_branch_length, 'weighted_mean =', weighted_mean)
                root_thetas[cp][param] = weighted_mean
    return(root_thetas)
