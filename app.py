import streamlit as st
from PIL import Image,ImageEnhance
import glob
import os
import numpy as np
import pickle
from feature_extractor import FeatureExtractor


def load_image(img):
    """Load image and resize it"""

    im = Image.open(img)
    im = im.resize((448,448))

    return im

def extract_feature():
    """Extract features from image database"""

    pass


def load_feature():
    """Load extraced faetures for similarity check with uploaded image"""

    features = []
    img_paths = []
    for feature_path in glob.glob("static/feature/*"):
        features.append(pickle.load(open(feature_path, 'rb')))
        img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')

    return features, img_paths

def check_image_similarity(extractor,features,img_paths,image_file):
    """Find similar images from image database"""

    img = Image.open(image_file)  
    query = extractor.extract(img)
    dists = np.linalg.norm(features - query, axis=1)  # Do search
    ids = np.argsort(dists)[:3] # Top 3 results
    scores = [(dists[id], img_paths[id]) for id in ids]

    return scores



def main():
    """Geological Image Similarity App"""

    html_title = """
    <div style="background-color:dodgerblue;padding:15px;">
    <h1 style="text-align:center"> Geological Image Similarity App</h1>
    </div>
    """
    st.markdown(html_title,unsafe_allow_html=True)
    
    image_file = st.file_uploader("Upload image to find similar images",type=['jpg','png','jpeg'])

    if image_file is not None:
        uploaded_image = load_image(image_file)
        st.markdown("## **_Original Image_**")
        st.image(uploaded_image)

        
        extractor = FeatureExtractor()
        features, img_paths = load_feature()
        scores = check_image_similarity(extractor,features,img_paths,image_file)   

        
        if st.checkbox("Show Similar Images"):
            st.markdown("## **_Top 3 Similar Images_**")
            for i in range(len(scores)):
                similar_image = load_image(scores[i][1])
                st.image(similar_image)

    if st.checkbox("Show Test Images(Right click -> Save -> Upload)"):
        for test_path in glob.glob("static/test/*"):
            st.image(test_path)

    st.markdown("#### _Link to GitHub Repo_")
    st.markdown("[Click here](https://github.com/aamibot/geosimilarityapp)")
    
            
if __name__ == "__main__":
    main()