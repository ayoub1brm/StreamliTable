import os
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image
import layoutparser as lp
import streamlit as st
import torch
import pdfplumber
import numpy
import io
import zipfile

model_tb = lp.Detectron2LayoutModel(config_path ="config.yaml",
            model_path ="model_final.pth",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: "Table"})

docs = st.file_uploader("File upload", accept_multiple_files=True, type="pdf")
bouton_action = st.button("Lancer")


if bouton_action :
    
    def table_detection(image_path,model_tb):
        #image = cv2.imread(image_path)
        image = image_path[..., ::-1] # load images
        return model_tb.detect(image)
    
    
    # Set the path to the Tesseract executable if it's not in your PATH
    def draw_points_on_text_blocks(image, layout):
        """
        Draw red points on the image at each coordinate specified by the TextBlock objects in the layout.
        
        Parameters:
        - image: The input image as a numpy array.
        - layout: The detected layout from the layout parser model.
        
        Returns:
        - image_with_points: The image with red points on the text block coordinates.
        """
        # Convert image to BGR (OpenCV format)
        #image = cv2.imread(image_path)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
        # Draw red points on each text block coordinate
        for txblock in layout:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = map(int, txblock.coordinates)
            # Draw red points at each corner of the text block
            image_bgr = cv2.circle(image_bgr, (x1, y1), radius=20, color=(0, 0, 255), thickness=-1)
            image_bgr = cv2.circle(image_bgr, (x2, y1), radius=20, color=(0, 0, 255), thickness=-1)
            image_bgr = cv2.circle(image_bgr, (x1, y2), radius=20, color=(0, 0, 255), thickness=-1)
            image_bgr = cv2.circle(image_bgr, (x2, y2), radius=20, color=(0, 0, 255), thickness=-1)
            # Optionally, draw a point in the center of the text block
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            image_bgr = cv2.circle(image_bgr, (cx, cy), radius=20, color=(0, 0, 255), thickness=-1)
    
        # Convert the image with points back to RGB for display (if needed)
        image_with_points_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_with_points_rgb
    
    
    
    # Function to convert documents to images and detect tables
    def process_documents(file_path,detection_model):
        # Iterate through each document in the folder
        #print('hello')
        #for root, _, files in os.walk(folder_path):
    
        #for file in files:
        pdf_path = os.getcwd()
        pdf_path = os.path.join(pdf_path,'file.pdf')
        with open(pdf_path,'wb') as f:
            f.write(file_path)
        # Assuming your documents are PDFs
        images = convert_from_path(pdf_path,dpi=200) # Convert document to images
        annotated_images= []
        for i, image in enumerate(images):
            # Detect tables in the image and annotate
            layouts = table_detection(numpy.array(image),detection_model)  # Call table detection function
            annotated_image = draw_points_on_text_blocks(numpy.array(image), layouts)
            annotated_images.append(Image.fromarray(annotated_image))

        return images, annotated_images
    buf = io.BytesIO()
    for uploaded_file in docs:
        starting_image, final_image = process_documents(uploaded_file.getvalue(),model_tb)
        st.text('Pdf to image')
        st.image(starting_image)
        st.text('Visualization of the table detection')
        st.image(final_image)
        j = 0
        for image in final_image:
            buf2 = io.BytesIO()
            image.save(buf2,'png')
            byte_im =buf2.getvalue()            
            with zipfile.ZipFile('result.zip', 'w') as img_zip:
                img_zip.writestr(f"annotated{uploaded_file.name}/{j}.png", byte_im)
            j +=1

    byte_zip = buf.getvalue()
    with open('result.zip') as fp:
        st.download_button(
                label="Download zip",
                data=byte_zip,
                file_name="result.zip",
                mime="applciation/zip"
        )
        
