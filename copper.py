#Importing the required Libraries

import base64
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox,inv_boxcox
from streamlit_option_menu import option_menu

# Instantiation

le = LabelEncoder()

# Setting Page configuration and Background

st.set_page_config(page_title = 'Copper Modeling',layout='wide') 

st.markdown('<h1 style="text-align: center;color:black;">Industrial Copper Modeling</h1>', unsafe_allow_html=True)
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    

set_background('img1.png')
selected = option_menu("Main Menu", ['Status Prediction','Selling Price Prediction'],menu_icon="menu-up", default_index=0,orientation ="horizontal")

# Transforming and Inverse Transforming the Date column for streramlit application

df1 = pd.read_csv('Copper.csv')
df1['item_date'] = le.fit_transform(df1['item_date'])
encoded_item_date = list(df1['item_date'].unique())
original_item_date = list(le.inverse_transform(df1['item_date'].unique()))

def item_date_dic():
    item_date = {}
    for key in original_item_date:
        for value in encoded_item_date:
            item_date[key] = value
            encoded_item_date.remove(value)
            break
    return item_date


#  STATUS PREDICTION
if selected == 'Status Prediction' :
        
        c1,c2 = st.columns(2)
        with c1:
            date = st.selectbox('Transaction Date (yyyy-mm-dd)',options= list(original_item_date))
            user_quant = st.number_input("Item Quantity in Tons(0.00 - 99999.99)")
            code = st.selectbox('Country code',options= list(df1['country'].unique()))
            use_type = st.selectbox('Item Type',options=('W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'))
            application = st.selectbox('Application',options=list(df1['application'].unique()))
        with c2:
            thick = st.number_input('Thickness(0.20 to 2500.00)')
            width = st.number_input('Width(1.0 to 3000.00)')
            prod_ref = st.selectbox('Product Reference Code:',options= list(df1['product_ref'].unique()))
            del_date = st.selectbox('Delivery Date (yyyy-mm-dd)',options= list(df1['delivery date'].unique()))
            sel_pr = st.number_input("Selling Price(0.10 to 82000.00)")
         
        button = st.button('Predict Status')
        

        if button:
            item = item_date_dic()
            item_date = item[date]

            quantity_tons = boxcox(user_quant,0.005921481390060094).round(2)

            item_type = {'W':5,'S':3,'PL':2,'WI':6,'others':1,'IPL':0,'SLAWR':4}
            item_typ = item_type[use_type]

            thickness = boxcox(thick,-0.1792730255842548).round(2)

            delivery_date = int(del_date[0:4])

            selling_price = boxcox(sel_pr,0.09343054475928997).round(2)

            ip = [[item_date,quantity_tons,code,item_typ,application,thickness,width,prod_ref,delivery_date,selling_price]]
            
            with open('rf_model.pkl','rb') as file:
                rf_model = pickle.load(file)

            status_predict = rf_model.predict(np.array(ip))
            if status_predict:
                st.markdown('<h2 style="text-align: center;color:black;">Transaction or Item Status : WON !</h2>', unsafe_allow_html=True)
            else:
                st.markdown('<h2 style="text-align: center;color:black;">Transaction or Item Status: LOST</h2>', unsafe_allow_html=True)


# SELLING PRICE PREDICTION
                
if selected == 'Selling Price Prediction'  :
        a1,a2 = st.columns(2)

        with a1:
            date = st.selectbox('Transaction Date (yyyy-mm-dd)',options= list(original_item_date))
            user_quant = st.number_input("Item Quantity in Tons(0.00 - 99999.99)")
            code = st.selectbox('Country code',options= list(df1['country'].unique()))
            use_type = st.selectbox('Item Type',options=('W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'))
            application = st.selectbox('Application',options=list(df1['application'].unique()))
        with a2:
            thick = st.number_input('Thickness(0.20 to 2500.00)')
            width = st.number_input('Width(1.0 to 3000.00)')
            prod_ref = st.selectbox('Product Reference Code:',options= list(df1['product_ref'].unique()))
            del_date = st.selectbox('Delivery Date (yyyy-mm-dd)',options= list(df1['delivery date'].unique()))
            status = st.selectbox('Status',options= ('Won','Lost'))
        
        button1 = st.button('Predict the Selling Price')
             
        if button1:
            item = item_date_dic()
            item_date = item[date]

            quantity_tons = boxcox(user_quant,0.005921481390060094).round(2)

            item_type = {'W':5,'S':3,'PL':2,'WI':6,'others':1,'IPL':0,'SLAWR':4}
            item_typ = item_type[use_type]

            thickness = boxcox(thick,-0.1792730255842548).round(2)

            delivery_date = int(del_date[0:4])
            
            def stat(status):
                if status == 'Won':
                    return 1
                else:
                    return 0
                

            ip1 = [[item_date,quantity_tons,code,item_typ,application,thickness,width,prod_ref,delivery_date,stat(status)]]

            with open('rf_reg.pkl','rb') as file:
                reg_model = pickle.load(file)
            
            price_predict = reg_model.predict(np.array(ip1))
            s_price = inv_boxcox(price_predict[0].round(2),0.09343054475928997)
            st.subheader(f"Predicted Selling Price: {s_price}")
            

