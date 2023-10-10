# number2-image-edge-detection
from os import lstat
from sys import displayhook
from qiskit import *
from qiskit import IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use('bmh')
# A 8x8 binary image represented as a numpy array
image = np.array([[0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                  [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
                  [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                  [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                  [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                  [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0]])           
# Function for plotting the image using matplotlib
def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(img.shape[1]))
    plt.yticks(range(img.shape[0]))
    plt.imshow (img,extent=[0,img.shape[1],img.shape[0],0] ,cmap='viridis')
    plt.show()
#plot_image(image, 'Original Image')
def amplitude_encode(img_data):
        # Calculate the RMS value
    rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))
        # Create normalized image
    image_norm =[]
    for arr in img_data:
        for ele in arr:
            image_norm.append(ele / rms)
            # Return the normalized image as a numpy array
    return np.array(image_norm)
# Get the amplitude ancoded pixel values
# Horizontal: Original image
image_norm_h = amplitude_encode(image)
#print(amplitude_encode(0))
# Vertical: Transpose of Original image
image_norm_v = amplitude_encode(image.T)
# Initialize some global variable for number of qubits
data_qb = 8
anc_qb = 1
total_qb = data_qb + anc_qb
# Initialize the amplitude permutation unitary
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
# Create the circuit for horizontal scan
qc_h = QuantumCircuit(total_qb)
qc_h.initialize(image_norm_h, range(1, total_qb))
qc_h.h(0)
qc_h.unitary(D2n_1, range(total_qb))
qc_h.h(0)
#print(qc_h.draw('mpl', fold=-1))
# Create the circuit for vertical scan
qc_v = QuantumCircuit(total_qb)
qc_v.initialize(image_norm_v, range(1, total_qb))
qc_v.h(0)
qc_v.unitary(D2n_1, range(total_qb))
qc_v.h(0)
#displayhook 
# (qc_v.draw('mpl', fold=-1))
# Combine both circuits into a single list
circ_list = [qc_h, qc_v]
# Simulating the cirucits
back = Aer.get_backend('statevector_simulator')
results = execute(circ_list, backend=back).result()
sv_h = results.get_statevector(qc_h)
sv_v = results.get_statevector(qc_v)
from qiskit.visualization import array_to_latex
print(sv_h)
# print('Horizontal scan statevector:')
#print(array_to_latex(sv_h[:30], max_size=30))
#print()
#print('Vertical scan statevector:')
#displayhook(array_to_latex(sv_v[:30], max_size=30))
# Classical postprocessing for plotting the output
# Defining a lambda function for
# thresholding to binary values
threshold = lambda amp: ( amp < -1e-15)
#threshold1 = lambda amp1: (amp1 > 1e-15)
# Selecting odd states from the raw statevector and
# reshaping column vector of size 64 to an 8x8 matrix
edge_scan_h1=np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for i in range (2**data_qb) ])).reshape(16, 16)
#edge_scan_h2=np.abs(np.array([1 if threshold1(sv_h[2*i+1].real) else 0 for i in range (2**data_qb) ]))
#edge_scan_h3=np.roll(edge_scan_h2,1)
#edge_scan_h4=edge_scan_h3.reshape(16, 16)
#edge_scan_h=edge_scan_h1|edge_scan_h4
edge_scan_v1 = np.abs(np.array([1 if threshold(sv_v[2*i+1].real)  else 0 for i in range(2**data_qb)])).reshape(16, 16).T
#dge_scan_v2=np.abs(np.array([1 if threshold1(sv_v[2*i+1].real) else 0 for i in range (2**data_qb) ]))
#edge_scan_v3=np.roll(dge_scan_v2,1)
#edge_scan_v4=edge_scan_v3.reshape(16, 16).T
#edge_scan_v=edge_scan_v1|edge_scan_v4
# Plotting the Horizontal and vertical
# cal scans
#plot_image(edge_scan_h, 'Horizontal scan output')
#plot_image(edge_scan_v, 'Vertical scan output')
# Combining the horizontal and vertical component of the result
edge_scan_sim = edge_scan_h1 | edge_scan_v1
# Plotting the original and edge-detected images
#plot_image(image, 'Original image')
plot_image(edge_scan_sim, 'Edge Detected image')
