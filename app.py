from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import preprocessor as p
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)	
	
@app.route('/predict',methods=['POST'])
def predict():
	with open('myword2vecModel.pkl', 'rb') as pickle_file:
		content = pickle.load(pickle_file)
	w2v=content
	clf = pickle.load(open('KNN.pkl', 'rb')) 


	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vector_record=[]
		#new_Sentence='I always play fair when I make the rules'

		sentence = data[0]
		try:
			sentence = p.clean(sentence)
		except:
			sentence = data[0]
		print(sentence)
		print(len(sentence.split()))
		
		wordSum = 0
		errorSum = 0
		vectorList = []
		sentLength = 0
		
		if len(sentence)!=0 and len(sentence.split())>5:
			#sentence_vect = sum([w2v[w] for w in sentence.lower().split()])/(len(sentence.split())+0.001)
			

			print(sentence)
			
			#alternative : 
			#words = filter(lambda x: x in model.vocab, doc.words)
			for w in sentence.lower().split():
				print(w)
				try:
					vectorList.append(w2v[w])
					sentLength+=1
				except:
					errorSum+=1
					vectorList.append(0)
					sentLength+=1
			
			print('creating vector')
			sentence_vect = sum(vectorList)/(sentLength+0.001)
			print('created vector')
			
			
		else:  
			sentence_vect = np.ones((100,)) #shouldn't come here ideally
			print('created vector')
			print(sentence_vect)
		
		try:		
			vector_record.append(sentence_vect) 
			X = pd.DataFrame(vector_record, columns=range(100))
			print(X.head())
			print(X.shape)
			
			#vect = cv.transform(data).toarray()
			

			my_prediction = clf.predict(vector_record)
		except:
			my_prediction = 0
		
		
		#### A block to check if input is not valid or just way too small.
		
		if((len(sentence.split())<=5) or (errorSum>=0.7 * sentLength) ):
			my_prediction=0
			print('Could not detect. Defaulting to 0.')
		print('Sentence Length is '+str(sentLength))
		print('Error Length is '+str(errorSum))
		print('Prediction is '+str(my_prediction))
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
