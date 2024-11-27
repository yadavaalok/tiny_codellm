from flask import Flask, jsonify, request
import os
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace
from flask_cors import CORS


app = Flask(__name__)

CORS(app)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

#Write a python code to print fibonacci series.

@app.route('/')
def index():
    return "Welcome to the Flask API!"


@app.route('/codellm', methods=['POST'])
def code_llm():
    try:
        data = request.get_json()
        question = data['question']
        model = data['model']
        llm = HuggingFaceHub(
            repo_id=model,
            task="text-generation",
        )

        system_prompt = """
        Strictly Code output only, please refrain from any further explanation or commentary.
    
        This is the user question : {}
        """

        chat_model = ChatHuggingFace(llm=llm)

        ai_msg = chat_model.invoke(system_prompt.format(question))

        print(ai_msg.content)

        return jsonify({"Response": ai_msg.content})
    except Exception as e:
        print("Error: ", e)
        return jsonify({"Error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
