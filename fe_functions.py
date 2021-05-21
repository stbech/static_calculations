from typing import Tuple
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")



# Hilfsfunktionen

def get_parameter_coords(x_elem, l_elem):
    """Umrechnung in Parameterraum"""
    return 2*x_elem/l_elem - 1 

 
    
def get_local_coords(xi_elem, l_elem):
    """Rückrechnung aus Parameterraum"""
    return l_elem/2*(xi_elem + 1)



# FE-Berechnung

def first_load_element(length: float, x_load: float, n_elem: int) -> int:
    """erstes Element mit Belastung"""
    return round(x_load/length*n_elem)



def element_lengths(length: float, x_load: float, n_elem: int) -> list:
    """Längen der einzelnen finiten Elemente"""
    break_point = first_load_element(length, x_load, n_elem)

    len_list = [x_load/break_point for i in range(break_point)]
    len_list.extend([(length - x_load)/(n_elem - break_point) for i in range(n_elem - break_point)])

    return len_list



def element_stiffness_matrix(EI: float, length: float) -> np.array:
    """Steifigkeitsmatrix eines finiten Elements"""
    matrix = EI/length**3*np.array([
        [       12,   -6*length,      -12,   -6*length],
        [-6*length, 4*length**2, 6*length, 2*length**2],
        [      -12,    6*length,       12,    6*length],
        [-6*length, 2*length**2, 6*length, 4*length**2] ,  
    ])

    return matrix



def stiffness_matrix(EI: list, length_vector: list) -> np.array:
    """Gesamtsteifigkeitsmatrix aller finiten Elemente"""
    size = 2*len(length_vector) + 2
    matrix = np.zeros([size, size])     # Erzeugen einer quadratischen leeren Matrix

    for i, length in enumerate(length_vector):
        elem_matrix = element_stiffness_matrix(EI, length)
        for row in range(len(elem_matrix)):
            for col in range(len(elem_matrix[0])):
                tot_row = row + 2*i
                tot_col = col + 2*i
                matrix[tot_row][tot_col] = matrix[tot_row][tot_col] + elem_matrix[row][col]

    return matrix



def element_load_vector(length: float, p: float) -> np.array:
    """konsistenter Lastvektor eines mit einer Gleichstreckenlast belasteten Elements"""
    shear = p*length/2
    moment = p*length**2/12

    return np.array([shear, -moment, shear, moment])



def load_vector(length_vector: list, x_load: float, n_elem: int, p: float) -> np.array:
    """konsistenter Gesamtlastvektor"""
    vector = np.zeros([2*len(length_vector)+2, 1])                              # Erzeugen eines leeren Vektors
    break_point = first_load_element(sum(length_vector), x_load, n_elem)
    
    for i, length in enumerate(length_vector[break_point:]):                    # für alle belasteten Elemente:
        elem_vector = element_load_vector(length, p)
        tot_row = (break_point + i)*2
        for row in range(len(elem_vector)):
            vector[tot_row + row] = vector[tot_row + row] + elem_vector[row]    # Addition neuer Werte zu bestehenden

    return vector



def reduce_matrix(matrix: np.array, bearing_situation: Tuple[bool, bool], start: bool) -> np.array:
    """Streichen der durch Auflagerbedingungen vorgegebenen Zeilen und Spalten"""
    count = 0
    for i, boolean in enumerate(bearing_situation):
        if boolean:                                     # falls Auflager vorhanden:
            if start:                                   # falls Auflager am linken Rand:
                index = -count                          # falls erste Zeile schon gestrichen
            else:
                index = len(matrix) - 2                 # Auflager rechts: beginne von hinten

            matrix = np.delete(matrix, index + i, 0)    # Lösche Zeile
            matrix = np.delete(matrix, index + i, 1)    # Lösche Spalte
            count += 1

    return matrix



def reduce_vector(vector: np.array, bearing_situation: Tuple[bool, bool], start: bool) -> np.array:
    """Streichen der durch Auflagerbedingungen vorgegebenen Zeilen"""
    count = 0
    for i, boolean in enumerate(bearing_situation):
        if boolean:
            if start:
                index = -count
            else:
                index = len(vector) - 2

            vector = np.delete(vector, index + i, 0)
            count += 1

    return vector



def extend_vector(vector: np.array, bearing_situation: Tuple[bool, bool], start: bool) -> np.array:
    """Vektor der Weggrößen um die durch die Auflagerbedingungen gegebenen Werte zu ergänzen"""
    if start:
        for i, boolean in enumerate(bearing_situation):
            if boolean:
                vector = np.insert(vector, i, 0)
    else:
        for i, boolean in enumerate(bearing_situation[::-1]):
            if boolean:
                vector = np.insert(vector, len(vector) - i, 0)
  
    return vector



def calculate_displace_vector(tot_stiffness_matrix: np.array, load_vector: np.array, bearing_A: Tuple[bool, bool], bearing_B: Tuple[bool, bool]) -> np.array:
    """Bestimme Weggrößenvektor aus der Gesamtsteifigkeitsmatrix"""
    # entferne durch Auflager vorgegebene Werte aus Gesamtsteifigkeitsmatrix
    tot_stiffness_matrix = reduce_matrix(tot_stiffness_matrix, bearing_A, True)    
    tot_stiffness_matrix = reduce_matrix(tot_stiffness_matrix, bearing_B, False)

    # entferne durch Auflager vorgegebene Werte aus konsistenten Gesamtlastvektor
    load_vector = reduce_vector(load_vector, bearing_A, True)
    load_vector = reduce_vector(load_vector, bearing_B, False)

    # Prüfe ob Gesamtsteifigkeitsmatrix invertierbar -> wenn nicht, ist System kinematisch
    try:
        K_inv = np.linalg.inv(tot_stiffness_matrix)     # invertiere Gesamtsteifigkeitsmatrix
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            print("Matrix nicht invertierbar\nPrüfe ob System statisch unbestimmt")
            raise
        else:
            raise err

    v_red = K_inv.dot(load_vector)              # berechne reduzierten Weggrößenvektor

    # Erweiterung des Weggrößenvektors mit durch Auflager vorgegebene Werte
    v = extend_vector(v_red, bearing_A, True)   
    v = extend_vector(v, bearing_B, False)
    
    return v



def elem_displace_vectors(deform_vector: np.array) -> list:
    """Zerlege Gesamtweggrößenvektor in Elementweggrößenvektor"""
    deform_list = []
    
    for i in range(int((len(deform_vector) - 2)/2)):
        start = i*2
        elem_vector = deform_vector[start:start+4]
        deform_list.append(elem_vector)

    return deform_list



def single_span_girder(length: float, x_load: float, n_elem: int, EI: float, p: float, bearing_A: Tuple[bool, bool], bearing_B: Tuple[bool, bool]) -> Tuple[list, list]:
    lengths = element_lengths(length, x_load, n_elem)       # Berechne Längen der finiten Elemente
    K = stiffness_matrix(EI, lengths)                       # Berechne Gesamtsteifigkeitsmatrix
    s_0 = load_vector(lengths, x_load, n_elem, p)           # Berechne konsistenter Lastvektor
    displacement_vector = calculate_displace_vector(K, s_0, bearing_A, bearing_B)   # Berechne Gesamtweggrößenvektor

    return elem_displace_vectors(displacement_vector), lengths  # Aufspalten in Elementweggrößenvektoren



# Bestimmung der Verformungs- und Momentenlinie

def trial_functions(xi: float, l_elem: float) -> np.array:
    """Ansatzfunktionen"""
    return np.array([
         1/4*xi**3 - 3/4*xi + 1/2,
        -l_elem/8*xi**3 + l_elem/8*xi**2 + l_elem/8*xi - l_elem/8,
        -1/4*xi**3 + 3/4*xi + 1/2,
        -l_elem/8*xi**3 - l_elem/8*xi**2 + l_elem/8*xi + l_elem/8,
    ])



def operator_functions(xi: float, l_elem: float) -> np.array:
    """Operatorfunktionen"""
    return -4/l_elem**2*np.array([
         3/2*xi,
        -3/4*l_elem*xi + l_elem/4,
        -3/2*xi,
        -3/4*l_elem*xi - l_elem/4,
    ])
    


def deformation(xi: float, elem_deforms: list, l_elem: float) -> float:
    """Berechne Verformung eines Punktes"""
    return trial_functions(xi, l_elem).dot(elem_deforms)



def moment(xi: float, elem_deforms: list, l_elem: float, EI: float) -> float:
    """Berechne Moment eines Punktes"""
    return EI*operator_functions(xi, l_elem).dot(elem_deforms)



def deform_line(elem_deform_list: list, l_elem_list: list, division: int = 20) -> Tuple[list, list]:
    """Verformungslinie"""
    x_coords = []
    deform = []
    x_start = 0
    parts = division + 1
    for i, length in enumerate(l_elem_list):
        for j in range(parts):
            x_coords.append(x_start + j*length/division)                    # x-Koordinate speichern
            xi = get_parameter_coords(j*length/division, length)            # Umrechnung in Parameterraum
            deform.append(deformation(xi, elem_deform_list[i], length))     # Berechnung der Verformung

        x_start += length

    return x_coords, deform



def moment_line(elem_deform_list: list, l_elem_list: list, EI: list, division: int = 20) -> Tuple[list, list]:
    """Momentenlinie"""
    x_coords = []
    moments = []
    x_start = 0
    parts = division + 1
    for i, length in enumerate(l_elem_list):
        for j in range(parts):
            x_coords.append(x_start + j*length/division)
            xi = get_parameter_coords(j*length/division, length)
            moments.append(moment(xi, elem_deform_list[i], length, EI))

        x_start += length

    return x_coords, moments
