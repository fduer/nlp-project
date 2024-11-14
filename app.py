# encoding=utf-8
from idlelib.rpc import response_queue

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from test import MODEL
import os
from mydataset import Dataset
app = Flask(__name__)
CORS(app)

# 实例化ChatbotModel
# chatbot_model = model()
tables = Dataset()
table_name = 'Food Names'
tables.load_data()
model = MODEL()
model.load_model()
print("-------start----------")
@app.route('/multi_question', methods=['POST'])
def api_multi_question():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    ans = model.run_model(question, tables.get_data(table_name)['info'])
    response = tables.run_query(table_name, ans)

    return jsonify({"answer": str(response)})


@app.route('/', methods=['GET'])
def index(): #
    return send_file('./demo/user_input.html')


if __name__ == '__main__':
    # before_init()
    app.run(port=5000, debug=True)