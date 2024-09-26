# task planner for llava

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import ast

def get_model_response(user_prompt,system_prompt):
    template = """
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {user_prompt}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

    prompt = PromptTemplate(
    input_variables=["system_prompt","user_prompt"],
    template = template
    )

    response = llm(prompt.format(system_prompt = system_prompt, user_prompt=user_prompt))

    return response

def generate_task_plans_prompt(scene_graph, user_input):
    template = f"""
    Mission : 
    You're tasked with generating plans to follow user's instruction.

    Input : 
    1. You'll receive a user's instruction which is a task you have to conduct.
    2. You'll receive a scene graph represented as a Python dictionary. Each key represents a workspace, and the corresponding value is a list of objects currently associated with that workspace. 

    Rules :
    1. You should make excellent plans to conduct user's instruction.
    2. You must only include objects and workspaces in the given scene graph in your plans.
    3. The plans can include moving objects to proper workspace.
    4. If the target location in the user's instruction is not in the input scene graph, aim for the location in the scene graph that is most similar to the target location.
    5. When moving objects, refrain from moving objects that are too heavy or large.

    Output :
    1. You'll output a plan to conduct user's instruction in form of sentences.
    2. You have to make a sufficient number of plans to follow user's instruction.

    ---------------------------------------------------------------------------------
    Example :
    [Input] : 
    [USER INSTRUCTION]
    Move the study materials on the desk and move all the objects away from couch.

    [Scene Graph]
    {{'counter' : ['book','pencil'],
    'couch' : ['remote controller','apple'],
    'couch_1' : ['toy']
    'table' : ['potted plant','bowl','spoon'],
    'desk' : ['laptop','pillow']}}
    
    [Process]
    Information analysis :
    1. Workspace identification : I can see workspaces 'counter', 'couch', 'table', 'desk' in the scene graph.
    2. Objects identification :
    (1) 'book', 'pencil' in workspace 'counter'
    (2) 'remote controller', 'apple' in workspace 'couch'
    (3) 'toy' in workspace' 'couch_1'
    (3) 'potted plant', 'bowl', 'spoon' in workspace 'table'
    (4) 'laptop', 'pillow' in workspace 'desk'

    Reasoning :
    There are two missions to solve. First is to move the study materials on the desk, and second is to move all the objects away from couch.
    I'll divide these two missions and make plans to solve it.
    
    First mission : move the study materials on the desk
    This is how i think to solve the mission.
    'book', 'pencil', 'laptop' are study materials in the scene. -> 'book' and 'pencil' are in 'counter' currently. -> I should move them to 'desk'.
    -> 'laptop' is a study material too. -> It is currently on 'desk' so it is in the right place. -> It doesn't have to me moved.

    Second mission : move all the objects away from couch.
    This is how i think to solve the mission.
    There are 'remote controller' and 'apple' in couch. -> They should be moved to other workspaces -> 'remote controller' should be moved to 'table' since someone would use it in the future.
    -> 'apple' should be moved to 'counter' since it's a food ingredient. -> 'toy' is in couch_1 and 'toy' should also be moved to other workspaces -> 'toy' should be moved to 'table' that someone would use it in the future. 

    [Output] :
    Step1. Move 'book' to the desk since it is a study material.
    Step2. Move 'pencil' to the desk since it is a study material.
    Step3. Move 'remote controller' to 'table' away from 'couch'
    Step4. Move 'apple' to 'counter' away from 'couch'
    Step5. Move 'toy' to 'table' from 'couch_1'
    
    ---------------------------------------------------------------------------------
    
    [User] 
    [Input] : 
    [USER INSTRUCTION]
    {user_input}

    [Scene Graph]  
    {scene_graph}

    [Process]

    [Output] : * Output format should follow Step1. Step2. Step3. etc.
    Step1.
    Step2.
    Step3.
    ...
    """
    return template

def preprocess_task_plans_prompt(task_plan):
    template = f"""
    Trim the given text below and write the step by step task plan in the given format as output. 
    Output should contain only the step-by-step sequence such as Step1, Step2, Step3, etc., without any other modifiers.

    
    [Text]
    {task_plan}
    
    [Format]
    Step1.
    Step2.
    Step3.
    ...
    """
    return template

def generate_steps_code_prompt(plans): 
    template = f"""
    Mission : 
    You're tasked with converting the given task plan to sequences of steps in form of python dictionary form with the given action functions you can use to conduct a task given by user.

    Input : 
    1. You'll receive a step-by-step task plan to conduct the task.

    Action function :
    1. GoTo(object) : 'object' parameter can be either the name of 'place' or 'object'. The robot agent will go in front of the 'place' or 'object' utilizing this function.
    2. Pickup_Object(object) : 'object' parameter is the name of 'object'. You'll pick up the object utilizing this function.
    3. Put_Object(object) : 'object' parameter is the name of 'object'. You should go to 'place' or 'object' first, then you'll put the holding object at the 'place' or near the 'object'.

    Output :
    1. You'll generate a python dictionary where each key of the dictionary represents each step, and each value represents a list of action functions to perform each step.
    2. Make sure that you only include the Action functions mentioned.
    3. You should print only the final dictionary, without any other modifiers.
    
    Example :
    [Input] : 
    Step1. Move 'book' to the desk since it is a study material.
    Step2. Move 'pencil' to the desk since it is a study material.
    Step3. Move 'remote controller' to 'table' away from 'couch'
    Step4. Move 'apple' to 'counter' away from 'couch'
    Step5. Move 'toy' to 'table' from 'couch_1'

    [Output] :
    {{"Step1": [GoTo(book), Pickup_Object(book), GoTo(desk), Put_Object(book)]
    "Step2" : [GoTo(pencil), Pickup_Object(pencil), GoTo(desk), Put_Object(pencil)], 
    "Step3": [GoTo(remote controller), Pickup_Object(remote controller),GoTo(table),Put_Object(remote controller)],
    "Step4": [GoTo(apple), Pickup_Object(apple),GoTo(counter),Put_Object(apple)],
    "Step5": [GoTo(toy), Pickup_Object(toy),GoTo(table),Put_Object(toy)]}}

    [User] 
    [Input] : 
    {plans}

    [Output] *give me only the final dictionary :
    """
    return template    

def generate_final_codes_prompt(task_plan):
    template = f"""
    Trim the given text below and output the final dictionary as a string. 
    Output should contain only the dictionary with the given [Format] without any other modifiers.

    [Text]
    {task_plan}
    
    [Format]
    {{"Step1":[~~],"Step2":[~~]}}

    [Output]
    """
    return template

def task_plan_generator(scene_graph, user_input):
    # step1. generate task plan
    user_prompt_1 = generate_task_plans_prompt(scene_graph, user_input)
    response_1 = get_model_response(user_prompt_1,"주어진 한국어 명령과 scene graph를 이용해서, task planning을 해줘.")
    print("############################################")
    print(f"[Step1. Generated Task Plan] : {response_1}")
    # step2. preprocess task plan : erase modifiers and only leave step-by-step task plan
    user_prompt_2 = preprocess_task_plans_prompt(response_1)
    response_2 = get_model_response(user_prompt_2,"Please trim the given text following the user prompt")
    print("############################################")
    print(f"[Step2. Preprocessed Task Plan] : {response_2}")
    # step3. convert task plan to sequences of action functions
    user_prompt_3 = generate_steps_code_prompt(response_2)
    response_3 = get_model_response(user_prompt_3,"Please preprocess the given task plan following the user prompt")
    print("############################################")
    print(f"[Step3. Conversion Process to Codes] : {response_3}")
    # step4. preprocess sequences of action functions 
    user_prompt_4 = generate_final_codes_prompt(response_3)
    response_4 = get_model_response(user_prompt_4,"Please preprocess the given task plan following the user prompt")
    print("############################################")
    print(f"[Step4. Preprocess Codes] : {response_4}")
    # step5. convert to python dictionary form
    dict_start = response_4.find('{')
    dict_end = response_4.rfind('}') + 1
    dict_str = response_4[dict_start:dict_end]
    task_plan = ast.literal_eval(dict_str)

    # 결과 출력
    print("############################################")
    print(f"[Final Task Plan] : {task_plan}]")
    
    return task_plan


if __name__ == "__main__":
    llm = Ollama(model="llama3.1:8b", stop=["<|eot_id|>"])
    print("[Llama Loaded]")

    scene_graph = {"countertop": ["bowl", "stove burner", "coffee machine", "pot", "pan","dish sponge"], "cabinet": ["microwave"], "table": ["house plant"], "floor": ["chair", "fridge"], "countertop_1": ["potato", "soap bottle", "lettuce"], "stove burner": ["salt shaker"], "countertop_3": ["plate"], "table_1": ["bread", "statue"]}

    user_input = "모든 요리도구들을 같은 곳에 모아줘."
    
    task_plan = task_plan_generator(scene_graph, user_input)
    





