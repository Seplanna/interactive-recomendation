import numpy as np
import math

class Vector(object):
    def __init___(self):
        self.parallel = np.empty()
        self.ortogonal = np.empty()
        self.parallel_norm = 0
        self.ortogonal_norm = 0

class Point(object):
    def __init__(self):
        self.x = 0
        self.y = 0

def GetDistanse(x, y):
    return math.sqrt(np.dot((x-y), x-y))

def GetFarthersPoint(points, x):
    d = 0
    point = 0
    for i in range(points):
        p = points[i]
        new_d = GetDistanse(p.ortogonal, x)
        if (d < new_d):
            d = new_d
            point = i
    return point, d

def GetDiametrOfSet(points):
    result = Point()
    diametr = 0
    for i in range(points):
        point, current_diametr = GetFarthersPoint(points[i+1,:], points[i].ortogonal)
        if (diametr < current_diametr):
            diametr = current_diametr
            result.x = i
            result.y = point
    return result

def GetOrtogonalBasis(oldBasis, old_basis_norm, new_v):
    new_el_in_basis = new_v
    for i in range(len(oldBasis)):
        b = oldBasis[i]
        b_norm = old_basis_norm[i]
        new_el_in_basis -= b * (np.dot(b, new_v) / b_norm)
    oldBasis.append(new_el_in_basis)
    old_basis_norm.append(np.dot(new_el_in_basis, new_el_in_basis))

def GetOrtogonalComponent(vectors, new_element_in_bas, new_element_norm):
    max_norm = 0
    min_norm = 100000
    for v in len(vectors):
        new_parallel = np.dot(v.ortogonal, new_element_in_bas) / new_element_norm
        v.parallel.append(new_parallel)
        v.parallel_norm += new_parallel * new_parallel
        v.ortogonal_norm -= new_parallel * new_parallel
        v.ortogonal -= new_parallel * new_element_in_bas
        if (min_norm > v.parallel_norm):
            min_norm = v.parallel_norm
        if (max_norm < v.parallel_norm):
            max_norm = v.parallel_norm
    return (min_norm, max_norm)

def GetSetOfPoints(vectors, min_norm, max_norm):
    result = []
    threshold = min_norm + (max_norm - min_norm) / math.sqrt(len(vectors))
    for i in range(len(vectors)):
        if (vectors[i].parallel_norm < threshold):
            result.append(i)
    return  result

def FirstAlgorithm(n_iterations, items):
    items_by_basis = []
    for i in range(len(items)):
        el = Vector()
        el.ortogonal = items[i]
        el.ortogonal_norm = np.dot(items[i], items[i].T)
        
    basis = []
    basis_norm = []
    questions = []
    arr = np.arange(len(items))
    pool_of_points = np.random.shuffle(arr)[:math.sqrt(len(items))]
    for i in range(n_iterations):
        question = GetDiametrOfSet(items_by_basis[pool_of_points])
        new_vector = items[question.x] - items[question.y]
        questions.append(new_vector)
        GetOrtogonalBasis(basis, basis_norm, new_vector)
        min_norm, max_norm = GetOrtogonalComponent(items_by_basis, basis[-1], basis_norm[-1])
        pool_of_points = GetSetOfPoints(items_by_basis, min_norm, max_norm)
    np.savetxt(np.array("data/questions", questions))






