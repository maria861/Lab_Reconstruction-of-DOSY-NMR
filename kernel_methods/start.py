# Math packages
import numpy as np

# Progress bar
from tqdm import tqdm

# Import different tools functions
from src.Dropout import *
from src.Transform import *
from src.ImportData import *
from src.Normalisation import *
from src.Preprocessing import *
from src.Predictions import *

# Import functions for the Arraysation 
from src.arraysation import *
from src.Array import *

# Import functions for the embedding
from src.EmbeddingDefault import *
from src.FiguresEmbedding import *
from src.SpectrumEmbedding import *


# Import functions for the selection of the model
from src.CrossValidation import *
from src.GridSearch import *

# Import functions for the kernels
from src.LinearKernel import *
from src.PolyKernel import *
from src.GaussianKernel import *
from src.KernelDefault import *

# Import function of model
from src.KernelLogisticRegression import *
from src.KernelSVM import *


if __name__ == '__main__':

    # Extraction of the dataset
    df_mat_dict = ImportData("./Data/plus/", "./Data/", suffix="_mat100")
    df_dict = ImportData("./Data/", "./Data/", header=0, sep=",")

    # Definition of the data augmentation function
    data_array = Array(Arraysation, {})

    # Defintion of the embedding
    embedding = EmbeddingDefault(SpectrumEmbedding, {"d_l": [5, 7, 12]})
    #

    # Definition of the kernel
    kernel = KernelDefault(PolyKernel, {"k":2 })

    # Definition of the model
    model = KernelLogisticRegression(kernel, informations=False,  lamda=1,
                                     max_iter=15, preprocessing=None)

    # Defintion of best parameters values
    best_parameters_values = {"Data Array": {"Function": data_array},
                              "Embedding": {"Function": embedding},
                              "Kernel": {"Function": kernel},
                              "Model": {"Function": model}}

    # Computation of the predicition
    predictions = Prediction(best_parameters_values, df_dict)

    # Save the Predicitons
    np.savetxt("Yte.csv", predictions,
               fmt='%i', delimiter=",", header="Id,Bound", comments='')
