import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import time
import threading


import pandas as pd
import matplotlib.pyplot as plt


# load model


import asyncio


import threading
import time


st.set_page_config(layout="wide")
import streamlit as st
import speech_recognition as sr
from time import sleep
import time
import os 
import subprocess

# Core 
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA
from altair_saver import save

import os 
import subprocess
import pandas as pd 
import numpy as np 
from datetime import datetime
import speech_recognition as sr

#For Pdf Generator 
import pdfkit
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
from datetime import date
import streamlit as st
from streamlit.components.v1 import iframe


import cv2
import streamlit as st

# EDA Pkgs
import joblib 
pipe_lr = joblib.load(open("models/emotion_classifier_pipe_lr_03_june_2021.pkl","rb"))




m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #2D2D2D;
    color:#D7F3D4;
    height:250px;
    width:435px;
    font-size:25px;
    border-radius:15px;
}
div.stButton > button:hover {
    background-color:#341e46;
    color:#D7F3D4;
    }
</style>""", unsafe_allow_html=True)






# font-size: 15px;



# name1 = "Spoorthy VV"
# section1 = "CSE"
# def function_for_name(name_parameter):
#     name1 = name_parameter
#     return name1
# def function_for_section(section_parameter):
#     section1 = section_parameter
#     return section1
    












# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table

def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results




emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


import os 
import subprocess
import pandas as pd 
import numpy as np 
from datetime import datetime
import speech_recognition as sr


import speech_recognition as sr
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}


# load json and create model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()

json_file.close()
classifier = model_from_json(loaded_model_json)


# load weights into new model
classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("INTRO-SPECT DETECTION SOFTWARE")
    activiteis = ["Home", "Face Emotion Detection","Interview","End Results","Text Analysis","About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.write(" ")
    
####################################################################################################################################################################
    if choice == "Home":
        st.success("Welcome to the Intro Spect Detection software 😊")
        html_temp_home1 = """




        <div style="border-radius: 10px; background:#; padding: 10px;font-size:28px; ">
        Kindly use the Side Pane For Navigation

        # <p style="background-image: url('b.jpg');">
        </div>
           
        """
        st.markdown(html_temp_home1, unsafe_allow_html=True)



        # mn = st.markdown("""
        # <style>
        # div.stButton > button:first-child {
        #     background-color: #2D2D2D;
        #     color:#D7F3D4;
        #     height:50px;
        #     width:435px;
        #     font-size:25px;
        #     border-radius:15px;
        # }
        # div.stButton > button:hover {
        #     background-color:#341e46;
        #     color:#D7F3D4;
        #     }
        # </style>""", unsafe_allow_html=True)


        # if submit_text:
        #     function_for_name(rname)
        #     function_for_section(rsection)








        

 
####################################################################################################################################################################
    elif choice == "Face Emotion Detection":
 


        html_temp_home1 = """


        <div style="border-radius: 10px; background:#b7b094; padding: 10px;font-size:20px; ">
        Click on start to use webcam to detect your face emotion


        .
        </div>
        <br></br>            
        """
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
                        
####################################################################################################################################################################

    elif choice == "Interview":


        # html_temp_home1 = """


        # <div style="border-radius: 10px; background:#b7b094; padding: 10px;font-size:20px; ">
        # Click on start to use webcam to detect your face emotion


        # .
        # </div>
        # <br></br>            
        # """
        # st.markdown(html_temp_home1, unsafe_allow_html=True)
        # webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
        #                 video_processor_factory=Faceemotion)





       
        mnm = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #2D2D2D;
            color:#D7F3D4;
            height:250px;
            width:655px;
            font-size:25px;
            border-radius:10px;
        }
        div.stButton > button:hover {
            background-color:#341e46;
            color:#D7F3D4;
            }
        </style>""", unsafe_allow_html=True)

        html_temp_home1 = """
         <div style="border-radius: 10px; background:#A29794; padding: 10px;font-size:25px;font-weight:bold ">
           ⚠️CLICK EVERY BUTTON ONLY ONCE⚠️
            
            </div>   
            """
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        

        st.write(" ")







        html_temp_home1 = """
         <div style="border-radius: 10px; background:#A29794; padding: 10px;font-size:25px;font-weight:bold ">
            INTRODUCE YOURESELF IN 60 SEC CHAMP 😊
            
            </div>   
            """
        st.markdown(html_temp_home1, unsafe_allow_html=True)


        col11,col22  = st.columns(2)

        with col11:
            # st.warning("Kindly Click on this to Start The interview")
         

            html_temp_home1 = """
             <div style="font-size:30px;font-weight:bold;text-align: center">
                Start                
                </div>                           
                """
            st.markdown(html_temp_home1, unsafe_allow_html=True)



            with st.spinner('Yeah Yeah Go on'):
                if st.button('Step 1: Record '):
                    subprocess.call(['sh', './test.sh'])
                    st.success('Done! You Have Done a Splendid Job Champ!!!!!')
            



        with col22:
            # st.warning("Kindly Click on this To Process Youre Data")
            html_temp_home1 = """
                



                <div style="font-size:30px;font-weight:bold;text-align: center">
                Process 
                </div>
                           
                """
            st.markdown(html_temp_home1, unsafe_allow_html=True)
            
            
            
            
            
            
            with st.spinner('Hang on a little while we process the data'):
                if st.button('Step 2: Process '):
                    subprocess.call(['sh', './test2.sh'])
                    subprocess.call(['sh', './test3.sh'])
                    st.success('YOUR REPORT IS READY . KINDLY COLLECT IT IN THE NEXT DIVISION')

            # reached2 =555
            # status = reached2
            # if reached2:
            #     st.success("YOUR REPORT IS READY . KINDLY COLLECT IT IN THE NEXT DIVISION")



        # html_temp_home1 = """
        #  <div style="border-radius: 10px; background:#A29794; padding: 10px;font-size:25px;font-weight:bold ">
        #     YOUR REPORT IS READY . KINDLY COLLECT IT IN THE NEXT DIVISION
            
        #     </div>   
        #     """
        # st.markdown(html_temp_home1, unsafe_allow_html=True)
        


        # with col33:
        #     html_temp_home1 = """
                



        #         <div style="font-size:30px;font-weight:bold;text-align: center;">
        #         View Result


                
        #         </div>
                          
        #         """
        #     st.markdown(html_temp_home1, unsafe_allow_html=True)


            # st.warning("Kindly Click on this To View Youre Result")

            # with open('converted.txt','r') as file:
            #     final1 = file.read()
            # with open('enhanced.txt','r') as file:
            #     final2 = file.read()
            # #Here for pdf Generator 
            # env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
            # template = env.get_template("template.html")





            # submit = st.button("Step 3: View Results")

            # if submit:
            #     html = template.render(
            #         name = name1,
            #         section = section1,
            #         student=final1,
            #         course=final2,
            #     )

            #     pdf = pdfkit.from_string(html, False)
            #     st.balloons()
            #     st.success("🎉 Youre pdf is Generated")



            #     st.download_button(
            #         "⬇️ Download PDF",
            #         data=pdf,
            #         file_name="report.pdf",
            #         mime="application/octet-stream",
            #         )
            #         #
            #         #
            #         #
            #         #
            #         #

####################################################################################################################################################################
    elif choice == "End Results":
        
        mn = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #2D2D2D;
            color:#D7F3D4;
            height:70px;
            width:630px;
            font-size:25px;
            border-radius:3px;
        }
        div.stButton > button:hover {
            background-color:#341e46;
            color:#D7F3D4;
            }
        </style>""", unsafe_allow_html=True)
        col111,col222  = st.columns(2)
        with col111:
            with st.form("Name"):

                rname = st.text_area("Kindly Type Your Name here ")
                rsection = st.text_area("Kindly Type Your Section here ")
                submit_text = st.form_submit_button()

                if submit_text:
                    st.success("Name and Section are  Recorded")
            
        with col222:
            with open('converted.txt','r') as file:
                final1 = file.read()
            with open('enhanced.txt','r') as file:
                final2 = file.read()
            #Here for pdf Generator 
            env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
            template = env.get_template("template.html")


            submit = st.button("Step 3: View Results")

            if submit:
                html = template.render(
                    name = rname,
                    section = rsection,
                    student=final1,
                    course=final2,
                )

                pdf = pdfkit.from_string(html, False)
                st.balloons()
                st.success("🎉 Youre pdf is Generated")



                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf,
                    file_name="report.pdf",
                    mime="application/octet-stream",
                    )
                    #
                    #
                    #
                    #
                    #
            st.info("Greet the interviewer with a handshake and a smile. Remember to maintain eye contact ")
            st.info("Never slight a teacher, friend, employer, or your university. ")
            st.info("No interview is complete until you follow up with a thank-you note.")
            st.info("Close on a positive, enthusiastic note.")
        










































    elif choice == "Text Analysis":



        proceed = 'Reached'
        if (proceed):
            col1,col2  = st.columns(2)
            with open('converted.txt','r') as file:
                raw_text = file.read()
            
            
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))



            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')


                st.altair_chart(fig,use_container_width=True)

  ####################################################################################################################################################################
    elif choice == "About":
        st.subheader("About this app")
        
####################################################################################################################################################################
    else:
        pass


if __name__ == "__main__":
    main()
