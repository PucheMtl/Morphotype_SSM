 import numpy as np
import open3d as o3d
import math
import copy

MEAN_MODEL_PATH = 'C:/Users/Phd/Documents/Morphotype_SSM/Morphotypes/exemple_pca/data/PJ116_humerus_A_avg.ply'
PCA_EIGEN_VECTORS = 'C:/Users/Phd/Documents/Morphotype_SSM/Morphotypes/exemple_pca/data/PCA_eigen_vectors_humerus.csv'
PCA_EIGEN_VALUES = 'C:/Users/Phd/Documents/Morphotype_SSM/Morphotypes/exemple_pca/data/PCA_eigen_values_humerus.csv'
NEW_MODEL_SAVE_PATH = './new_model.ply'


# Construction d'un nouveau modele 3D a partir du modele moyen + les paramÃ¨tres de la PCA
# new_model = mean_model + P * b, P : matrice contenant les vecteurs propres, b parametres
# Ã  initialiser pour dÃ©finir le nouveau modele
def build_new_model(mean_model_path=MEAN_MODEL_PATH,
                    pca_eigen_vectors=PCA_EIGEN_VECTORS,
                    pca_eigen_values=PCA_EIGEN_VALUES):
    # Lecture du fichier contenant le modele 3D moyen
    mean_model = o3d.io.read_triangle_mesh(mean_model_path)

    # RÃ©cuperation des sommets et des faces du modele 3D moyen
    mean_model_vertices = np.asarray(mean_model.vertices)
    mean_model_faces = np.asarray(mean_model.triangles)
    print('Vertices - shape:')
    print(mean_model_vertices.shape)
    print('Triangles - shape:')
    print(mean_model_faces.shape)

    # Lecture du fichier csv contenant les vecteurs propres estimÃ©s par PCA
    pca_eigen_vectors = np.loadtxt(pca_eigen_vectors, delimiter=";", dtype=float)
    print('Eigen vectors - shape:')
    print(pca_eigen_vectors.shape)

    # Verification entre le nombre de sommets du modele moyen et le nombre de composantes qui forment un vecteur propre
    if mean_model_vertices.shape[0]*3 != pca_eigen_vectors.shape[0]:
        print("Attention, erreur entre le model moyen et les vecteurs propres")

    # Lecture du fichier csv contenant les valeurs propres estimÃ©es par PCA
    pca_eigen_values = np.loadtxt(pca_eigen_values, delimiter=";", dtype=float)
    print('Eigen values - shape:')
    print(pca_eigen_values.shape)

    # Generation d'un nouveau modele : new_model = mean_model + P * b
    # Indiques les modes qui doivent participer Ã  la construction de la nouvelle forme
    mode = [1,2] #plusieurs modes possibles [0,1], GTA: CP2 et 3 d'après Leen
    # b is a vector of floats to apply to each mode of variation
    full_b = np.zeros(pca_eigen_values.shape)
    for i in mode:
        # Attention, ici je donne une valeur random entre les bornes statistiquement acceptable +/-3 * sqrt(eigen_value)
        #new_value = np.random.uniform(-3 * math.sqrt(pca_eigen_values[i]), 3 * math.sqrt(pca_eigen_values[i]))
        new_value = 800
        full_b[i] = new_value
    # Calcul de P * b
    pca_applied = pca_eigen_vectors.dot(full_b)

    # Creation d'un nouveau modele : mean_model + pca_applied
    example = mean_model_vertices.flatten().T + pca_applied

    # Creation graphique pour le nouveau modele modele
    new_model = copy.deepcopy(mean_model)
    new_model_vertices = np.reshape(example, (mean_model_vertices.shape[0],mean_model_vertices.shape[1]))
    new_model.vertices = o3d.utility.Vector3dVector(new_model_vertices)
    new_model.compute_vertex_normals()
    new_model.paint_uniform_color([0.1, 0.5, 0.5])
    new_model.translate((100, 0, 0))
    mean_model.compute_vertex_normals()

    # Sauvegarde du modele dans un fichier .ply
    o3d.io.write_triangle_mesh(NEW_MODEL_SAVE_PATH, new_model)

    # Affichage du modele moyen et du nouveau modele
    o3d.visualization.draw_geometries([mean_model, new_model], width=1000, height=700)


if __name__ == '__main__':
    build_new_model()





