def add_dict_key_prefix(d, prefix):
    out = dict()
    for key in d.keys():
        out[prefix + '_' + key] = d[key]
    return out
