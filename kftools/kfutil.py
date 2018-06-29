def add_dict_key_prefix(d, prefix):
    out = dict()
    for key in d.keys():
        out[prefix + '_' + key] = d[key]
    return out

def rgb_to_hex(r,g,b):
    rgb = [r,g,b]
    for i in range(len(rgb)):
        assert rgb[i]<=1
        rgb[i] = int(numpy.round(rgb[i]*255, decimals=0))
    hex_color = '#%02X%02X%02X' % (rgb[0],rgb[1],rgb[2])
    return hex_color
