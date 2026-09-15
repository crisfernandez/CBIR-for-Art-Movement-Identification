import time
import torch
import faiss
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import os
import time
import cv2

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import io, color

import streamlit as st
from streamlit_cropper import st_cropper    

from tensorflow import keras
from tensorflow.keras import Model

from sklearn.feature_extraction.image import extract_patches_2d
from scipy.cluster.vq import kmeans, vq

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

st.set_page_config(layout="wide")

device = torch.device('cpu')

FILES_PATH = str(pathlib.Path().resolve())

# Path in which the images should be located
IMAGES_PATH = os.path.join(FILES_PATH, 'DatasetArteTrainTest/Train')
# Path in which the database should be located
DB_PATH = os.path.join(FILES_PATH, 'database')
DB_FILE = 'db.csv' # name of the database

def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    # Convertir la imagen en un array de numpy y normalizar (escalar píxeles a 0-1)
    img_array = np.array(img)
    return img_array

def get_image_list():
    
    df = pd.read_csv(os.path.join(DB_PATH, DB_FILE))
    image_list = list(df.image.values)

    return image_list

def calculate_histograms(image, bins=32):
    red = cv2.calcHist([image], [2], None, [bins], [0, 256])
    green = cv2.calcHist([image], [1], None, [bins], [0, 256])
    blue = cv2.calcHist([image], [0], None, [bins], [0, 256])
    vector = np.concatenate([red, green, blue], axis=0)
    vector = vector.reshape(-1)
    vector = np.array(vector)
    vector = vector.reshape(vector.shape[0], -1)
    vector = vector.reshape(1, -1)
    return vector

def calcular_histograma_textura(imagen, distancias=[5], angulos=[45]):
    """
    Calcula un histograma basado en propiedades de textura (GLCM).
    """
    imagen_gris = rgb2gray(imagen)
    imagen_gris = (imagen_gris * 255).astype(np.uint8)

    # Calcular GLCM
    glcm = graycomatrix(imagen_gris, distances=distancias, angles=angulos, symmetric=True, normed=True)
    
    # Propiedades de textura
    propiedades = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    valores_textura = {prop: graycoprops(glcm, prop).flatten() for prop in propiedades}
    
    # Crear un histograma concatenando los valores de textura
    histograma = np.concatenate([valores_textura[prop] for prop in propiedades])
    histograma = histograma.reshape(histograma.shape[0], -1)
    histograma = histograma.reshape(1, -1)
    return histograma

def extract_image_patches(image, random_state, patch_size=(30, 30), n_patches=250):
    patches = extract_patches_2d(
        image, 
        patch_size=patch_size, 
        max_patches=n_patches, 
        random_state=random_state
    )
    return patches.reshape((n_patches, -1))

def bag_of_words_extractor(img_query):
    codebook = np.load('Bag_Of_Words.npy')
    orb = cv2.ORB_create()
    image = np.asarray(img_query)
    patch_size=(30, 30)
    n_patches=250
    random_seed=0
    random_state = np.random.RandomState(random_seed)
    patches = []
    patches.append(extract_image_patches(image, random_state, patch_size, n_patches))

    keypoints = []
    descriptors = []
    for patch in patches:
        patch_keypoints, patch_descriptors = orb.detectAndCompute(patch, None)
        keypoints.append(patch_keypoints)
        descriptors.append(patch_descriptors)

    visual_words = [vq(desc, codebook)[0] for desc in descriptors]
    frequency_values = [np.bincount(word, minlength=250) for word in visual_words]
    frequency_values = np.float32(frequency_values)
    faiss.normalize_L2(frequency_values)

    return frequency_values

def compute_deep_features(image):
    # Carga el modelo preentrenado VGG19
    base_model = VGG19(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    # Ajusta la imagen al tamaño esperado por el modelo
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)
    image_preprocessed = preprocess_input(image_array)
    # Extrae características
    features = model.predict(image_preprocessed)
    features = features.flatten()
    return features.reshape(1, -1)

def autoencoder_extractor(image: np.array, encoder):
    # batch dimension
    image = np.expand_dims(image, axis=0)
    feature_vector = encoder.predict(image)
    feature_vector = feature_vector.reshape(1, -1)
    return feature_vector

def retrieve_image(img_query, feature_extractor, n_imgs=11):
    img_query = preprocess_image(img_query)

    if (feature_extractor == 'Extractor 1: Histograma Color'):
        model_feature_extractor = calculate_histograms(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH,  'feat_extract_1.index'))
        if model_feature_extractor is None:
            return None

    elif (feature_extractor == 'Extractor 2: Texturas'):
        model_feature_extractor = calcular_histograma_textura(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH,  'feat_extract_2.index'))
        if model_feature_extractor is None:
            return None

    elif (feature_extractor == 'Extractor 3: Bag Of Words'):
        model_feature_extractor = bag_of_words_extractor(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH,  'feat_extract_3.index'))
        if model_feature_extractor is None:
            return None

    elif (feature_extractor == 'Extractor 4: CNN-VGG19'):
        img_query = np.array(img_query) / 255.0
        model_feature_extractor = compute_deep_features(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH,  'feat_extract_4.index'))
        if model_feature_extractor is None:
            return None
        
    elif (feature_extractor == 'Extractor 5: Autoencoder'):
        img_query = np.array(img_query) / 255.0
        filename = 'autoencoder.keras'
        autoencoder = keras.models.load_model(filename)
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dec_conv0').output)
        model_feature_extractor = autoencoder_extractor(img_query, encoder)
        indexer = faiss.read_index(os.path.join(DB_PATH,  'feat_extract_5.index'))
        if model_feature_extractor is None:
            return None
    
    else:
        model_feature_extractor = calculate_histograms(img_query)
        indexer = faiss.read_index(os.path.join(DB_PATH,  'feat_extract_1.index'))
        if model_feature_extractor is None:
            return None

    # TODO: Modify accordingly

    _, indices = indexer.search(model_feature_extractor, k=n_imgs)

    return indices[0]

def metric_precision(img_query, retriev, k_values=[1, 3, 5, 7, 11]):
    img_query_name = img_query.split('-')[1].split('.')[0]   
    image_list = get_image_list()
    precision_values = {}

    for k in k_values:
        if k > len(retriev):
            continue  # Skip if k is not possible

        retrieved_image_names = [image_list[retriev[i]].split('-')[1].split('.')[0] for i in range(k)]
        # Count number of images with same label. 
        relevant_count = sum(1 for img in retrieved_image_names if img == img_query_name)
        precision = relevant_count / k
        precision_values[k] = round(precision, 2)
        
    return precision_values

def main():
    st.title('CBIR IMAGE SEARCH')
    
    col1, col2 = st.columns(2)

    with col1:
        st.header('QUERY')

        st.subheader('Choose Feature Extractor')
        # TODO: Adapt to the type of feature extraction methods used.
        option = st.selectbox('.', ('Extractor 1: Histograma Color', 'Extractor 2: Texturas', 'Extractor 3: Bag Of Words', 'Extractor 4: CNN-VGG19', 'Extractor 5: Autoencoder'))

        st.subheader('Upload Image')
        img_file = st.file_uploader(label='.', type=['png', 'jpg'])

        if img_file:
            img = Image.open(img_file).convert('RGB')
            cropped_img = img
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0004')
            
            # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)

    with col2:
        st.header('RESULT')
        if img_file:
            st.markdown('*Retrieving .......*')
            start = time.time()

            retriev = retrieve_image(cropped_img, option, n_imgs=11)
            image_list = get_image_list()

            end = time.time()
            st.markdown('*Finish in ' + str(end - start) + ' seconds*')

            # Evaluation metrics
            st.markdown('## PRECISION')
            precision_results = metric_precision(img_file.name, retriev)
            # Crear una lista de elementos en formato HTML para mostrarlos en línea
            precision_html = " | ".join([f"Precision-{k} = {precision}" for k, precision in precision_results.items()])
            # Mostrar los resultados en línea
            st.markdown(precision_html, unsafe_allow_html=True)

            col3, col4 = st.columns(2)

            with col3:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[0]]))
                st.image(image, use_container_width = 'always')

            with col4:
                image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[1]]))
                st.image(image, use_container_width = 'always')

            col5, col6, col7 = st.columns(3)

            with col5:
                for u in range(2, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_container_width = 'always')

            with col6:
                for u in range(3, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_container_width = 'always')

            with col7:
            
                for u in range(4, 11, 3):
                    image = Image.open(os.path.join(IMAGES_PATH, image_list[retriev[u]]))
                    st.image(image, use_container_width = 'always')

if __name__ == '__main__':
    main()