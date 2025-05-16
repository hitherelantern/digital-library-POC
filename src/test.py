import os
import numpy as np
"""This is just a random python file for test purpose!"""

# Given path
path = r"D:\digital library\Business,finance and economics\The_Role_of_Public_Relations_in_Branding.pdf"

# Extract the last part with .pdf extension
filename = os.path.basename(path)
print(filename)

relative_path = r'data/'
p = os.path.abspath(relative_path)
print(os.listdir(relative_path),os.path.basename(relative_path))





# import os

# index_dir = "faiss_index"
# abs_path = os.path.abspath(index_dir)
# print(f"FAISS index is stored at: {abs_path}")
