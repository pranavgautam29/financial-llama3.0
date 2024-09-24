import streamlit as st
st.title("Welcome to Financial Llama.")
st.write("Ask all your financial doubts. :money_with_wings: :v:")
from llama_cpp import Llama

model = Llama(
    model_path = "/content/unsloth.Q4_K_M.gguf",  
)

g_kwargs = {
    "max_tokens":2000,
    "temperature":0.3,
    "echo":False,
    "top_k":1
}

def get_response(question: str):
  """ Returns an answer from the llm """

  # define the correct prompt format
  prompt_format = """Below is a question. Write a response that accurately answers that question.

    ### Question:
    {}

    ### Response:
    {}"""

  # format the input according to the format

  prompt = prompt_format.format(
      question,
      ""
  )

  # create a dictionary of generational arguments
  g_kwargs = {
      "max_tokens":2000,
      "temperature":0.3,
      "echo":False,
      "top_k":1
  }

  # input the prompt and arguments to the model

  response = model(
      prompt,
      **g_kwargs
  )

  # return the output string

  return response["choices"][0]["text"]


input = st.text_input("Enter your Question")

response = get_response(input)

if st.button("Submit"):
  st.write(response)
