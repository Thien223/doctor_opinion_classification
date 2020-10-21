from flask import Flask, render_template, request, json, redirect
from main import detect
from waitress import serve
import os
## set current working directory to source code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("\nCurrent working directory: ---'{}'---\n".format(os.getcwd()))

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analysis', methods=['GET','POST'])
def analysis():
	print(request.form)
	opinions = request.form['opinions']
	index_words, name_words,code_words, opinions_sub_opinions, sub_opinions_keywords = detect(opinions)
	res = {'index_words':index_words, 'name_words':name_words, 'code_words':code_words, 'opinions_sub_opinions':opinions_sub_opinions, 'sub_opinions_keywords':sub_opinions_keywords}
	# return app.response_class(response=json.dumps(res), mimetype='application/json')
	return json.dumps(res)







if __name__=='__main__':
	serve(app, host='0.0.0.0', port=1000)