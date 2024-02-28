#! /usr/bin/python3 
import inquirer
import os
import argparse
from llama_cpp import Llama

#set globals
global user_prompt
user_prompt = ''
global token_ceiling
token_ceiling = ''
global output_type
output_type = 0
global context_range
context_range = 512 
chat_format = None
personality_prompt = "You are a pirate from Florida who is unassumingly soulful and loves dogs."

#get arguments
parser = argparse.ArgumentParser('mbp_ai')
parser.add_argument("--model", dest="model", type=str, help='The model file you want to load from ./models')
parser.add_argument("--prompt", dest="prompt", type=str, help='A string form for the prompt')
parser.add_argument("--context", dest="context", type=int, help='Context range set - int')
parser.add_argument("--token", dest="token", type=int, help='Max tokens for prompt')
parser.add_argument("--persona", dest="set_personality", type=str, help='Configure personality if type is set to conversation mode')
parser.add_argument("--type", dest="completion_type", type=int, help="Conversation type (Default 0) - set to 1 for chat-completion type...")
arg = parser.parse_args()

def triage_args():
    if arg.prompt:
        get_prompt(arg.prompt)
        print("prompt set via argument")
    else:
        get_prompt()
    if arg.token:
        get_max_tokens(arg.token)
        print("Token ceiling set -", token_ceiling)
    else:
        get_max_tokens()
    if arg.completion_type:
        print("completion type set to ", arg.completion_type)
        if arg.completion_type > 0 and type(arg.completion_type) == int:
            chat_format = "llama-2"
            global output_type
            output_type = arg.completion_type 
    if arg.context:
        if arg.context < 2049:
            context_range = arg.context
        else:
            print('**CONTEXT** is higher than recommended... Defaulting to 2048')
            context_range = 2048
        print("Context range set -", context_range)
    if arg.set_personality:
            global personality_prompt
            personality_prompt = arg.set_personality
       # else:
       #     personality_greetings_list = [
       #         'Who would you like to speak to?',
       #         'May I ask who you are calling for?'
       #             ]
       #     dice_throw = random.randint()
       #     if dice_throw > 0.5:
       #         personality_greeting = personality_greetings_list[0]
       #     else:
       #         personality_greeting = personality_greetings_list[1]
       #     personality_prompt = input(personality_greeting)

# Define user_prompt
def get_prompt(prompt=''):
    global user_prompt
    if prompt == '':
        user_prompt = input('Present your matters to The Oracle \n\t>> ')
        if user_prompt == '':
            user_prompt = 'Name the planets in the solar system'
    else: 
        user_prompt = prompt


# Set token ammount
def get_max_tokens(token=''):
    global token_ceiling
    if token == '':
        token_ceiling = input('Max tokens? (default 32) \n\t>>')
        if token_ceiling == '':
            token_ceiling = 32
        else:
            token_ceiling = int(token_ceiling)
    else:
        token_ceiling = token


#begin triaging arguments
triage_args()

# Set model directory
mbp_ai_directory = os.path.dirname(os.path.realpath(__file__))
model_directory = mbp_ai_directory + '/models'
local_model_files = os.listdir(model_directory)

# List models:
local_models = [
    inquirer.List(
        "available",
        message="What local model to run?",
        choices=local_model_files,
    ),
]

selected_model = inquirer.prompt(local_models)
selected_model = model_directory + '/' + selected_model["available"]

# start llama
llm = Llama(
      model_path=selected_model,
      n_threads = 8, #set cpu threads to use
      n_gpu_layers=35, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=context_range, # Uncomment to increase the context window
      chat_format=chat_format,
)

def simple_llama_output():
    output = llm(
          "Q: " + user_prompt + " A: ", # Prompt
          max_tokens=token_ceiling, # Generate up to 32 tokens, set to None to generate up to the end of the context window
          stop=["Q:"], # Stop generating just before the model would generate a new question
          echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    return output

def chat_completion_llama():
    print(' -- chat completion mode --')
    output = llm.create_chat_completion(
          messages = [
              {
                  "role": "assistant", 
                  # "content": "You are a valley girl from California who is unassumingly smart and enjoys dogs."},
                  "content": personality_prompt},
              {
                  "role": "user",
                  "content": user_prompt
              }
          ],
    )
    return output

# Handle output data
if output_type > 0:
    output = chat_completion_llama()
    simple_output = output['choices'][0]['message']['content']
else:
    output = simple_llama_output()
    simple_output = output['choices'][0]['text']


print(output)
print('\n' + ('-' * 15) + '\n')
print(simple_output)
