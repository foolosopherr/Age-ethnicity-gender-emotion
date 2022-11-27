import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import cv2
from aegclass import MultilabelCNN


st.title('Age, ethnicity, gender and emotion classifiaction')
st.write('\n')
st.header('You can upload an image of one or more people')

image = st.file_uploader('Upload')

model_age = torch.load('model age.pt', map_location=torch.device('cpu'))
model_emotion = torch.load('model emotion.pt', map_location=torch.device('cpu'))
model_ethnicity = torch.load('model ethnicity.pt', map_location=torch.device('cpu'))
model_gender = torch.load('model gender.pt', map_location=torch.device('cpu'))


emotion_transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((48, 48)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                    ])

aeg_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((48, 48)),
                transforms.ToTensor()
                ])

ethnicity_labels = {0:'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Hispanic'}

gender_labels = {0: 'Male', 1: 'Female'}

emotion_labels = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprised'}


all_data = {'age': {'model': model_age, 'transform': aeg_transform}, 
            'emotion': {'model': model_emotion, 'transform': emotion_transform, 'labels': emotion_labels}, 
            'ethnicity': {'model': model_ethnicity, 'transform': aeg_transform, 'labels': ethnicity_labels}, 
            'gender': {'model': model_gender, 'transform': aeg_transform, 'labels': gender_labels}
            }


def check_image(image, all_data, category):
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    tr_image = all_data[category]['transform'](image)
    y_pred = all_data[category]['model'](tr_image)

    if category != 'age':        
        lst = torch.softmax(y_pred, dim=-1).detach().numpy().flatten()
        lst = list(map(lambda x: np.round(100*x, 2), lst))

        class_names = all_data[category]['labels'].values()
        preds = pd.DataFrame({'category':class_names, 'probability': lst}).sort_values(by='probability', ascending=False)
        preds['probability'] = preds['probability'].apply(lambda x: f"{x} %")
        return preds
    else:
        y_pred = int(torch.round(y_pred).item())
        min_age = max(0, y_pred-2)
        s = f"{min_age}-{y_pred+2}"
        return s

def cv2_face_extractor(image):
    # image = cv2.imread(image)
    pil_image = Image.open(image).convert('RGB') 
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    # open_cv_image = open_cv_image[:, :, ::-1].copy() 
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(open_cv_image, 1.1, 4)
    if len(faces) == 1:
        st.write(f"Found {len(faces)} face")
    else:
        st.write(f"Found {len(faces)} faces")

    for (x, y, w, h) in faces:
        FaceImg = open_cv_image[y:y+h,x:x+w]
        age, gender = check_image(FaceImg, all_data, 'age'), check_image(FaceImg, all_data, 'gender')
        ethnicity, emotion = check_image(FaceImg, all_data, 'ethnicity'), check_image(FaceImg, all_data, 'emotion')
        col1, col2 = st.columns(2)
        with col1:
            st.image(FaceImg)
        with col2:
            result = f"Age: {age}  \n \
                       Ethnicity: {ethnicity.iloc[0, 0]} ({ethnicity.iloc[0, 1]})  \n  \
                       Gender: {gender.iloc[0, 0]} ({gender.iloc[0, 1]})  \n \
                       Emotion: {emotion.iloc[0, 0]} ({emotion.iloc[0, 1]})"
            st.write(result)


if image:
    st.header('Original image')
    st.image(image)
    st.write('\n')
    st.write('\n')
    
    cv2_face_extractor(image)

else:
    st.write('I am waiting for image')
