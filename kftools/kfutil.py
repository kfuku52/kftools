import numpy

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

def get_rgb_gradient(ncol, col1, col2, colm=[0.5,0.5,0.5]):
    nmid = ncol/2
    cols = list()
    for i in range(ncol):
        if i<nmid:
            contrib_col1 = (nmid-i)/nmid
            contrib_colm = 1 - contrib_col1
            colnew = [0,0,0]
            for j in range(len(colnew)):
                colnew[j] = (col1[j]*contrib_col1)+(colm[j]*contrib_colm)
        elif i>nmid:
            contrib_col2 = (i-nmid)/nmid
            contrib_colm = 1 - contrib_col2
            colnew = [0,0,0]
            for j in range(len(colnew)):
                colnew[j] = (col2[j]*contrib_col2)+(colm[j]*contrib_colm)
        elif i==nmid:
            colnew = colm
        cols.append(colnew)
    return cols
