#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# reviews_raw = pickle.load(open('reviews_raw.pkl','rb'))
reviews_raw = pd.read_csv('reviews_raw.csv')
model_test = pickle.load(open('xgbmodel.pkl','rb'))
# common_user_predicted_ratings = pickle.load(open('common_user_predicted_ratings.pkl','rb'))
user_final_rating = pickle.load(open('user_final_rating.pkl','rb'))
reviews_raw_subset = pickle.load(open('reviews_raw_subset.pkl','rb'))
tfidfconverter = pickle.load(open('tfidfconverter.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    user_name = request.form.get("UserName")
    print("UserName: ", user_name)
    
    # Retrive user_id from from 
 
    
    # if str(user_name).lower() in str(reviews_raw['reviews_username'][:]).lower():
    if reviews_raw['reviews_username'].str.contains(user_name).any():  
        print("User Exist")
        user_ID = reviews_raw.loc[reviews_raw['reviews_username'] == user_name]['user_id'].values[0]
        print("user_ID: ", user_ID)
        if user_ID in user_final_rating.index[:]:
            
            user_final_rating.loc[user_ID].sort_values(ascending=False)[0:25]
            final_rec_ratings_based = user_final_rating.loc[user_ID].sort_values(ascending=False)[0:25]
            final_rec_ratings_based = final_rec_ratings_based.to_frame()
            
            
            product_mapping = reviews_raw[["product_id","name","brand","categories"]]
            # remove duplicates
            product_mapping.drop_duplicates(subset="product_id",keep="first",inplace=True)
            
            # Merge final_rec_ratings_based with reviews_raw_subset dataframe which would have the reviews on which text pre=processing is done
        
            prod_reviews = pd.merge(final_rec_ratings_based,reviews_raw_subset,left_on='product_id',right_on='product_id', how = 'left')
            # prod_reviews.head()
            
            UniqueProductValues = prod_reviews['product_id'].unique()
            
            # Create a dataframe to compute positive reviews percentage
            prod_reviews_counter = prod_reviews["product_id"].value_counts().rename_axis('product_id').reset_index(name='Review_cnt')
            prod_reviews_counter["Pos_rev_cnt"]=0
            prod_reviews_counter["Neg_rev_cnt"]=0
            prod_reviews_counter["Pos_rev_per"]=0
            
            final_rec_ratings_sentiment_based=pd.DataFrame(columns=['product_id'])
            
            # Iterate through unique products and their review to apply sentiments model to predict positive / negative sentiment
            for prod_id in UniqueProductValues:
                print ("For Prod id ",prod_id )
                if prod_id in prod_reviews_counter.product_id.values:
                    prod_rev_cntr = prod_reviews_counter.loc[prod_reviews_counter["product_id"]==prod_id,"Review_cnt"]
            
                    for i in range (0, int(prod_rev_cntr)):
                        input1 = [prod_reviews.loc[prod_reviews["product_id"]==prod_id,"Review_Comments_Cleaned"].iloc[i]]
                        print(input1)
                        input_data = tfidfconverter.transform(input1).toarray()
                        input_pred = model_test.predict(input_data)
            
                        if input_pred[0] == 1:
                            print("Review is Positive for " ,i)
                            Pos_rev_cnt = prod_reviews_counter.loc[prod_reviews_counter["product_id"]==prod_id,"Pos_rev_cnt"].iloc[0]
                            Pos_rev_cnt = Pos_rev_cnt + 1
                            prod_reviews_counter.loc[prod_reviews_counter["product_id"]==prod_id,"Pos_rev_cnt"]=Pos_rev_cnt

                        else:
                            print("Review is Negative for " ,i)
                            Neg_rev_cnt = prod_reviews_counter.loc[prod_reviews_counter["product_id"]==prod_id,"Neg_rev_cnt"].iloc[0]
                            Neg_rev_cnt = Neg_rev_cnt + 1
                            prod_reviews_counter.loc[prod_reviews_counter["product_id"]==prod_id,"Neg_rev_cnt"]=Neg_rev_cnt
            
            # Compute positve reviews percentage
            prod_reviews_counter["Pos_rev_per"] = (abs((prod_reviews_counter.Pos_rev_cnt - prod_reviews_counter.Neg_rev_cnt) / prod_reviews_counter.Pos_rev_cnt) * 100).replace(np.inf, 0)
            
            # Arrange in decending order or positive percentages and assign it to a dataframe
            prod_reviews_counter_sorted=prod_reviews_counter.sort_values(["Pos_rev_per"], ascending=False)
            
            
            # Get top 20 Products to be recommended
            final_rec_ratings_sentiment_based["product_id"] = prod_reviews_counter_sorted["product_id"][0:20]

            
            # final_rec_ratings_sentiment_based
            
            prod_rec = pd.merge(final_rec_ratings_sentiment_based,product_mapping,left_on='product_id',right_on='product_id', how = 'left')
            prod_rec.drop_duplicates(subset="product_id",keep="first",inplace=True)
            prod_rec = prod_rec.drop("categories",axis=1)
        
           
            print("prod_rec")
            return render_template('prediction.html',  tables=[prod_rec.to_html(classes='data')], titles=prod_rec.columns.values)
        else:
            error="Cannot get predictions for this user"
            return render_template('index.html', prediction_text=error)
            
    else:
        print("User Doesn't Exist")
        error ="User Doesn't Exist"
        return render_template('index.html', prediction_text=error)
        
    
if __name__=='__main__':
    app.run(debug=True)

