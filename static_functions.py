import numpy as np      # Matrizenrechnung
import pandas as pd     # Für die Plotdarstellung benötigt

from PIL import Image, ImageFont, ImageDraw     # benötigt Zeichnen der Ausgabe
from string import ascii_uppercase, digits      # Liste der Großbuchstaben, Ziffern

import seaborn as sns               # für schöneres Zeichnen der Plots




def verify_input(L, EI, P, C_N, C_M):
    """Check input validity"""
    if not(len(L) == len(EI) == len(P)):
        raise Exception("L, EI, P must all be of same length")
    elif len(C_N) != len(C_M):
        raise Exception("C_N and C_M must all be of same length")
    elif len(C_N) != len(L)+1:
        raise Exception("C_N must be one entry longer than L")
    else: 
        print("Input successfully passed checks")



# Weggrößenverteilung
def load_vectors(P, L):
    """Lastvektoren P_quer, P_0 und s_0"""
    P_quer = np.zeros(len(L)*2 + 2)

    P_0 = np.zeros(len(L)*2 + 2)
    for i in range(len(L) + 1):
        if i == 0:
            P_0[1+i*2] = L[i]*P[i]/2
        elif i == len(L):   
            P_0[1+i*2] = L[i-1]*P[i-1]/2
        else: 
            P_0[1+i*2] = L[i-1]*P[i-1]/2 + L[i]*P[i]/2

    s_0 = np.zeros(4*len(L) + 2)    # 2*n(L) für Stabendkraftgrößen + 2*(n(L) + 1) für Knotenfedern
    for i, l in enumerate(L):
        s_0[0+2*i] = P[i]*l*l/12
        s_0[1+2*i] = -P[i]*l*l/12
    
    return P_quer, P_0, s_0



def kinematic_transformation_matrix(L): 
    """kinematische Transformationsmatrix a"""
    a = np.zeros([4*len(L) + 2, 2*len(L) + 2])

    # Werte der Transformationsmatrix für die Federn infolge Verformungseinheitszuständen
    for i in range(len(L) + 1):         
        a[len(L)*2+2*i][2*i]     = 1    # infolge Verdrehung
        a[len(L)*2+1+2*i][2*i+1] = -1   # infolge Verschiebung

    # Werte der Transformationsmatrix für die Stabendweggrößen infolge Verformungseinheitszuständen
    a[0][0] = 1
    a[2*len(L)-1][-2] = 1
    for i in range(len(L) - 1):
        a[1+2*i][2+2*i] = 1
        a[2+2*i][2+2*i] = 1
    
    # Werte der Transformationsmatrix für die Stabendweggrößen infolge Verformungseinheitszuständen
    a[0][1] = a[1][1] = -1/L[0]
    a[2*len(L)-2][-1] = a[2*len(L)-1][-1] = 1/L[-1]
    for i in range(len(L) - 1):
        a[0+2*i][3+2*i] = a[1+2*i][3+2*i] = 1/L[i]
        a[2+2*i][3+2*i] = a[3+2*i][3+2*i] = -1/L[i+1]

    return a



def stiffness_matrix(L, EI, C_N, C_M):
    """Steifigkeitsmatrix k"""
    k = np.zeros([4*len(L) + 2, 4*len(L) + 2])
    for i, ei in  enumerate(EI):
        k[0+2*i][0+2*i] = k[1+2*i][1+2*i] =  4*ei/L[i]
        k[1+2*i][0+2*i] = k[0+2*i][1+2*i] = 2*ei/L[i]

    for i in range(len(C_N)):
        k[len(L)*2+2*i][len(L)*2+2*i] = C_M[i]
        k[len(L)*2+1+2*i][len(L)*2+1+2*i] = C_N[i]

    return k



def displacement_method(L, EI, P, C_N, C_M):
    """Zusammenstellung des Weggrößenverfahrens"""
    P_quer, P_0, s_0 = load_vectors(P, L)
    a = kinematic_transformation_matrix(L)
    k = stiffness_matrix(L, EI, C_N, C_M)

    aT_k = a.transpose().dot(k)
    K = aT_k.dot(a)

    P_res = P_quer + P_0 - a.transpose().dot(s_0)

    V = np.linalg.inv(K).dot(P_res)
    v = a.dot(V)

    s_I = k.dot(v) + s_0
    s_II = np.copy(s_I)
    for i in range(len(L)):
        s_II[0+2*i] = -1*s_II[0+2*i]

    return s_II



# Aufteilung des Schnittgrößenvektors
def support_loadings(s_II, L):
    """Rückgabe von Auflagerkräften und -momenten"""
    forces = [-s_II[2*len(L)+1+2*i] for i in range(len(L)+1)]
    moments = [-s_II[2*len(L)+2*i] if s_II[2*len(L)+2*i] != 0 else s_II[2*len(L)+2*i] for i in range(len(L)+1)]
    
    return forces, moments



def nodal_moments(s_II, L):
    """Stabmomente an den Knoten"""
    return s_II[0:2*len(L)]




# Plot der Schnittgrößen
def x_axis_points(L, number_of_divides):
    """Liste von x-Koordinaten für gewisse Stabteilung"""
    sprung = number_of_divides*50
    x_axis = [-0.01]

    for i in range(len(L)):
        if i != 0:
            x_values = [x_axis[-1] + (j+1)*L[i]/number_of_divides for j in range(number_of_divides)]
            x_axis.append(x_axis[-1] + L[i]/sprung)
        else:
            x_axis.append(0)
            x_values = [x_axis[-1] + (j+1)*L[i]/number_of_divides for j in range(number_of_divides)]
    
        x_axis.extend(x_values)
    x_axis.append(sum(L)+0.01) 
    
    return x_axis



def get_nodal_shear_forces(force_vector, L, P):
    """Berechnung der Querkräfte im Stab an den Auflagern"""
    shear_forces = [-force_vector[2*len(L)+1]]

    for i in range(len(L)):
        shear_forces.append(shear_forces[-1] - P[i]*L[i])
        if i != len(L) - 1:
            shear_forces.append(shear_forces[-1] - force_vector[len(L)*2+3+i*2])

    return shear_forces



def moment_distribution(force_vector, L, P, number_of_divides):
    """Liste der Momentenverteilung passend zu den X-Koordinaten"""
    x_axis = x_axis_points(L, number_of_divides)
    nodal_shear_forces = get_nodal_shear_forces(force_vector, L, P)
    moments = [0]

    for i in range(number_of_divides+1):
        moments.append(force_vector[len(L)*2] - force_vector[len(L)*2+1]*x_axis[i+1] - P[0]*x_axis[i+1]**2/2)

    l_vorh = 0
    for j in range(len(L)-1):
        l_vorh += L[j]
        for i in range(number_of_divides+1):
            x_element = (x_axis[i+2+j+number_of_divides*(j+1)]-l_vorh)
            moments.append(force_vector[2+2*j] + nodal_shear_forces[2+2*j]*x_element - P[1+j]*x_element**2/2)

    moments.append(0)
    
    return x_axis, moments



def shear_force_distribution(force_vector, L, P, number_of_divides):
    """Liste der Querkraftverteilung passend zu den X-Koordinaten"""
    x_axis = x_axis_points(L, number_of_divides)
    nodal_shear_forces = get_nodal_shear_forces(force_vector, L, P)
    shear_forces = [0]

    for i in range(number_of_divides+1):
        shear_forces.append(-force_vector[len(L)*2+1] - P[0]*x_axis[i+1])

    l_vorh = 0
    for j in range(len(L)-1):
        l_vorh += L[j]
        for i in range(number_of_divides+1):
            shear_forces.append(nodal_shear_forces[2+2*j] - P[1+j]*(x_axis[i+2+j+number_of_divides*(j+1)] - l_vorh))

    shear_forces.append(0)
    
    return x_axis, shear_forces




def plot_moments(force_vector, L, P, number_of_divides=20):
    """Darstellung der Momentenlinie"""
    x, y = moment_distribution(force_vector, L, P, number_of_divides)

    df = pd.DataFrame({'Länge': x, 'Moment': y})
    mom = sns.lineplot(x='Länge', y='Moment', data=df, sort=False)
    mom.invert_yaxis()

    return mom



def plot_shear_forces(force_vector, L, P, number_of_divides=20):
    """Darstellung der Querkraftlinie"""
    x, y = shear_force_distribution(force_vector, L, P, number_of_divides)

    df = pd.DataFrame({'Länge': x, 'Querkraft': y})
    shear = sns.lineplot(x='Länge', y='Querkraft', data=df, sort=False)
    shear.invert_yaxis()

    return shear



# Hilfsfunktion
def max_moments(force_vector, L, P):
    """Maximalwerte der Momente in den jeweiligen Feldern"""
    left_shear_force = get_nodal_shear_forces(force_vector, L, P)[::2]
    left_node_moments = force_vector[0:2*len(L):2]

    max_mom = []
    for i in range(len(L)):
        if left_shear_force[i]/P[i] <= L[i]:
            max_mom.append(left_node_moments[i] + left_shear_force[i]**2/P[i]/2)
        else:
            max_mom.append(force_vector[1+2*i])
    
    return max_mom



# Zusammenstellung der Ausgabe
def draw_text(text_list, img): 
    """Zeichnet Text auf Bild"""
    pen = ImageDraw.Draw(img)
       
    for el in text_list:
        pen.text(el["coord"], el["text"], font=el["font"], fill='black')
        if el["underline"]:
            el_size = el["font"].getsize(el["text"])
            pen.line([el["coord"][0], el["coord"][1] + el_size[1] + 10, el["coord"][0] + el_size[0], el["coord"][1] + el_size[1] + 10], fill="black", width=2)

    return img



def draw_summary(result, L, EI, P, C_N, C_M, syst_pic_path, mom_buf=None, shear_buf=None):
    """Zusammenfassung zeichnen"""
    heading = ImageFont.truetype("arial.ttf", 70)
    subheading = ImageFont.truetype("arial.ttf", 55)
    text = ImageFont.truetype("arial.ttf", 40)
    footnote = ImageFont.truetype("arial.ttf", 30)

    size = (2480, 3508)
    img = Image.new("RGB", (int(size[0]), int(size[1])), "white")
    pencil = ImageDraw.Draw(img)

    # System-Zeichnung einfügen
    system_pic = Image.open(syst_pic_path)
    system_pic = system_pic.resize((int(system_pic.size[0]/1.5), int(system_pic.size[1]/1.5)))
    img.paste(system_pic, (int((size[0] - system_pic.size[0])/2) - 100, 200))

    height =  system_pic.size[1] + (len(L)+1)*100 + 500

    # Überschriften
    text_list = [#
    {"text": f"Berechnung eines Trägers mit {len(L)} Feldern", "font": heading, "coord": ((size[0]-heading.getsize(f"Berechnung eines Trägers mit {len(L)} Feldern")[0])/2,100), "underline": True},
    {"text": "(beispielhafte Darstellung für 3 Felder)", "font": footnote, "coord": (int((size[0] + system_pic.size[0])/2)-200, 100+system_pic.size[1]), "underline": False},
    {"text": "Eingabe", "font": subheading, "coord": (250, 200+system_pic.size[1]), "underline": True},
    {"text": "Knoten", "font": text, "coord": (250, 300+system_pic.size[1]), "underline": True},
    {"text": "Stäbe", "font": text, "coord": (size[0]/2-50, 300+system_pic.size[1]), "underline": True},
    {"text": "Ergebnis", "font": subheading, "coord": (250, height), "underline": True},
    {"text": "Knoten", "font": text, "coord": (250, 100+height), "underline": True},
    {"text": "Stäbe", "font": text, "coord": (size[0]/2-50, 100+height), "underline": True},
    ]

    img = draw_text(text_list, img)

    # Eingabedaten
    node_in = ["Nr.", "c_N [kN/m]", "c_M [kNm/rad]"]
    node_in_coord = (250, 420+system_pic.size[1])
    beam_in = ["Nr.", "L [m]", "EI [kNm²]", "p [kN/m]"]
    beam_in_coord = (size[0]/2-50, 420+system_pic.size[1])

    for i, el in enumerate(node_in):
        pencil.text((node_in_coord[0]+270*i, node_in_coord[1]-40), el, font=text, fill='black')

    for i, el in enumerate(zip(ascii_uppercase, C_N, C_M)):
        for j in range(len(node_in)):
            pencil.text((node_in_coord[0]+270*j, node_in_coord[1]-50+(i+1)*100), str(el[j]), font=text, fill='black')

    for i, el in enumerate(beam_in):
        pencil.text((beam_in_coord[0]+270*i, beam_in_coord[1]-40), el, font=text, fill='black')

    for i, el in enumerate(zip(digits[1:], L, EI, P)):
        for j in range(len(beam_in)):
            pencil.text((beam_in_coord[0]+270*j, beam_in_coord[1]+(i+1)*100), str(el[j]), font=text, fill='black')

    pencil.line([node_in_coord[0], node_in_coord[1]+20, beam_in_coord[0]+4*270, node_in_coord[1]+20], fill="black", width=2)

    # Ausgabedaten
    node_out = ["Nr.", "C [kN]", "M [kNm]"]
    node_out_coord = (250, height+220)
    beam_out = ["Nr.", "M_l [kNm]", "M_max [kNm]", "M_r [kNm]"]
    beam_out_coord = (size[0]/2-50, height+220)

    for i, el in enumerate(node_out):
        pencil.text((node_out_coord[0]+270*i, node_out_coord[1]-40), el, font=text, fill='black')

    support = support_loadings(result, L)
    for i, el in enumerate(zip(ascii_uppercase, support[0], support[1])):
        for j in range(len(node_out)):
            if j == 0:
                pencil.text((node_out_coord[0]+270*j, node_out_coord[1]-50+(i+1)*100), str(el[j]), font=text, fill='black')
            else:
                pencil.text((node_out_coord[0]+270*j, node_out_coord[1]-50+(i+1)*100), str(round(el[j], 3)), font=text, fill='black')

    for i, el in enumerate(beam_out):
        pencil.text((beam_out_coord[0]+270*i, beam_out_coord[1]-40), el, font=text, fill='black')

    node_moment = nodal_moments(result, L)
    max_mom = max_moments(result, L, P)
    for i, el in enumerate(zip(digits[1:], node_moment[::2], max_mom, node_moment[1::2])):
        pencil.text((beam_out_coord[0]+270*0, beam_out_coord[1]+(i+1)*100), str(el[0]), font=text, fill='black')
        pencil.text((beam_out_coord[0]+270*1, beam_out_coord[1]+(i+1)*100), str(round(el[1], 3)), font=text, fill='black')
        pencil.text((beam_out_coord[0]+270*2, beam_out_coord[1]+(i+1)*100), str(round(el[2], 3)), font=text, fill='black')
        pencil.text((beam_out_coord[0]+270*3, beam_out_coord[1]+(i+1)*100), str(round(el[3], 3)), font=text, fill='black')

    pencil.line([node_out_coord[0], node_out_coord[1]+20, beam_out_coord[0]+4*270, node_out_coord[1]+20], fill="black", width=2)

    # Momenten-, Querkraftlinie
    if shear_buf:
        shear_pic = Image.open(shear_buf)
        shear_pic = shear_pic.resize((int(shear_pic.size[0]*1.4), int(shear_pic.size[1]*1.4)))
        img.paste(shear_pic, (int((size[0]-shear_pic.size[0])/2), size[1] - 100 - shear_pic.size[1]))

    if mom_buf:
        mom_pic = Image.open(mom_buf)
        mom_pic = mom_pic.resize((int(mom_pic.size[0]*1.4), int(mom_pic.size[1]*1.4)))
        if shear_buf:
            img.paste(mom_pic, (int((size[0]-mom_pic.size[0])/2), size[1] - 150 - mom_pic.size[1] - shear_pic.size[1]))
        else:
            img.paste(mom_pic, (int((size[0]-mom_pic.size[0])/2), size[1] - 150 - mom_pic.size[1]))

    return img