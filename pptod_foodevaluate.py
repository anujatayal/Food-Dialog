import torch
from transformers import T5Tokenizer
# model_path = r'./ckpt/small/epoch_1_dev_e2e_evaluation_inform_100.0_success_82.35_bleu_31.29_combine_score_122.47/'
model_path = r'./ckpt/small/epoch_1_dev_e2e_evaluation_inform_100.0_success_82.35_bleu_31.29_combine_score_122.47/'
tokenizer = T5Tokenizer.from_pretrained(model_path)
from modelling.T5Model import T5Gen_Model
from ontology import sos_eos_tokens
from dataclass import MultiWozData
from config import Config
import json
import random
import re
from collections import OrderedDict

special_tokens = sos_eos_tokens
model = T5Gen_Model(model_path,tokenizer,special_tokens,dropout=0.0,add_special_decoder_token=True,is_training=False)
sos_context_token = tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
eos_context_token = tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]
pad_token,sos_btoken,eos_btoken,sos_atoken,eos_atoken,sos_rtoken,eos_rtoken,sos_ictoken,eos_ictoken = \
        tokenizer.convert_tokens_to_ids(['<_PAD_>','<sos_b>','<eos_b>','<sos_a>','<eos_a>','<sos_r>','<eos_r>','<sos_d>','<eos_d>'])
bs_prefix_text = 'translate dialogue to belief state:'
da_prefix_text = 'translate dialogue to dialogue action:'
response_prefix_text = 'translate dialogue to system response:'
intent_prefix_text = 'translate dialogue to user intent:'
bs_prefix_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(bs_prefix_text))
da_prefix_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(da_prefix_text))
response_prefix_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_prefix_text))
intent_prefix_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(intent_prefix_text))
# dialogue_context = "<sos_u> I want to eat pork chop <eos_u>"
# dialogue_context = "<sos_u> I want to eat pork chop <eos_u><sos_r> what type of food are you eating? <eos_r><sos_u> it is braised <eos_u><sos_r> what is the quantity of pork chops? <eos_r><sos_u> 100 gms <eos_u>"

data_path_prefix = "./../data/multiwoz/data/"
cfg = Config(data_path_prefix)
data = MultiWozData("t5-small", tokenizer, cfg, data_path_prefix, shuffle_mode='unshuffle', 
        data_mode='train', use_db_as_input=True, add_special_decoder_token=True, 
        train_data_ratio=0.01)
# print("data",data)

with open("rules.json","r") as input_file:
    rules = json.load(input_file)

def rule_value(bspan,constraint_dict,output):
    if 'foodweight' in constraint_dict['food']:
        if 'value' in constraint_dict['food']:
            old_value = constraint_dict["food"]["value"] 
            del constraint_dict["food"]["value"] #remove value
            # print("value removed",constraint_dict)
        constraint_dict["food"] = rule_weights_change(constraint_dict["food"])
        foodweight = constraint_dict["food"]["foodweight"]            
        try:
            float(foodweight)
        except ValueError:
            # print("not float")
            if '/' in foodweight:
                numerator, denominator = foodweight.split('/')
                try:
                    foodweight = float(numerator) / float(denominator)
                except ValueError:
                    foodweight = "100"
                    constraint_dict["food"]["metric"] = "g"
            else:
                foodweight = "100"
                constraint_dict["food"]["metric"] = "g"

        del constraint_dict["food"]["foodweight"] #remove foodweight
        old_venues = data.reader.db.queryJsons("food", constraint_dict["food"])
        # print("check if foodweight constraint is good",old_venues)
        if len(old_venues)==0:
            constraint_dict["food"]["foodweight"] = foodweight
            constraint_dict["food"] = rule_weights(constraint_dict["food"])
            # print("No value for constraint dict",constraint_dict["food"] )
            foodweight = constraint_dict["food"]["foodweight"]
            constraint_dict["food"]["foodweight"] = '100'
            constraint_dict["food"]["metric"] = 'g'
            # print(constraint_dict)
            venues = data.reader.db.queryJsons("food", constraint_dict["food"], return_name=True)
            print("new venues with 100 g food",venues)
            if len(venues)>=1:
                constraint_dict["food"]["foodweight"] = foodweight
                constraint_dict["food"]["metric"] = "g"
                value = (float(venues[0])*float(foodweight))/100
                constraint_dict["food"]['value']=str(round(value,2))
            else:
                print("output",output)
                constraint_dict["food"]["foodweight"] = foodweight
                constraint_dict["food"]["metric"] = "g"
                value = (float(output['value'])*float(foodweight))/100
                constraint_dict["food"]['value']=str(round(value,2))
        else:
            # print("old_venues[0]",old_venues[0])
            #check if value can be converted to float 
            weight = float(old_venues[0]['foodweight'])
            constraint_dict["food"]["foodweight"] = foodweight
            constraint_dict["food"]["metric"] = old_venues[0]["metric"]
            value = (float(old_venues[0]['value'])*float(foodweight))/weight
            constraint_dict["food"]['value']=str(round(value,2))
        # print("foodweight constraint",constraint_dict["food"])
    elif 'value' in constraint_dict["food"]:
        old_venues = data.reader.db.queryJsons("food", constraint_dict["food"], return_name=True)
        # print("check if all constraints match in db along with value",old_venues)  #check if all constraints match in db along with value
        if len(old_venues)==0: #not in db
            old_value = constraint_dict["food"]["value"] 
            del constraint_dict["food"]["value"] #remove value
            # print("value removed",constraint_dict)
            venues = data.reader.db.queryJsons("food", constraint_dict["food"], return_name=True)
            # print("check whether constraint are good",venues)
            if len(venues)>=1:
                constraint_dict["food"]['value']=venues[0] #if yes take 1st value
            elif len(output)>0:
                constraint_dict["food"]['value']=old_value
    bspan = data.reader.constraint_dict_to_bspan(constraint_dict["food"])
    bspan = "[food]" +bspan
    return bspan,constraint_dict

def rule_weights(dict):
    if 'foodweight' in dict and 'metric' in dict and dict['metric'] not in ['g','gm','gms'] and dict['metric']+'-gm' in rules:
        # print("here",dict,rules[dict['metric']+'-gm'])
        foodweight = dict["foodweight"]
        oz_val = str(float(dict['foodweight'])*rules[dict['metric']+'-gm'])
        dict['foodweight']=oz_val
        print("foodweight changed",oz_val)
    else:
        print("default foodweight changed")
        dict['foodweight']='100'
    dict['metric']='gm'
    return dict

def rule_weights_change(constraint_dict):
    if 'foodweight' in constraint_dict:
        foodweight = constraint_dict['foodweight']
        pattern = r'(\d+(\.\d+)?)\s+(.+)'
        match = re.match(pattern, foodweight)
    if match:
        constraint_dict['foodweight'] = str(match.group(1))
        constraint_dict['metric'] = match.group(3)
    else:    
        if constraint_dict['foodweight'] == 'half a pound':
            constraint_dict['foodweight'] = "250"
            constraint_dict['metric'] = "g"
        if constraint_dict['foodweight'] == 'pound':
            constraint_dict['foodweight'] = "500"
            constraint_dict['metric'] = "g"
        if constraint_dict['foodweight'] == 'full pot':
            constraint_dict['foodweight'] = "250"
            constraint_dict['metric'] = "g"
        # if constraint_dict['foodweight'] == '1/2':
        #     constraint_dict['foodweight'] = ".5"
    # print("new cons weights",constraint_dict)
    return (constraint_dict)

         


def belief_rule(bspan,output):
    constraint_dict = data.reader.bspan_to_constraint_dict(bspan)
    print("belief_rule const",constraint_dict) #get dict
    if 'food' in constraint_dict:
        constraint_dict['food'] = clean_description(constraint_dict['food'])
    # constraint_dict['food'] = rule_weights(constraint_dict['food']) #change weights if needed
        bspan, constraint_dict = rule_value(bspan,constraint_dict,output) #change value if needed
        return (bspan,constraint_dict["food"])
    else:
        return (bspan,constraint_dict)
def clean_description(dict):
    words_to_remove = ["in","salt"]
    words= dict['food'].split()
    cleaned_words = [word for word in words if word not in words_to_remove]
    # Join the cleaned words back into a string
    unique_words = list(OrderedDict.fromkeys(cleaned_words))
    dict['food'] = ' '.join(unique_words)
    return dict

def evaluate():
    base_response=""
    output = []
    flag =0
    while True:
        response = input("Enter response:")
        if response.lower() in ['quit','q']:
            break
        dialogue_context = base_response+"<sos_u> " + response + " <eos_u>"
        print("dialog_context",dialogue_context)
        context_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dialogue_context))
        #predict belief state
        input_id =  bs_prefix_token + [sos_context_token] + context_id + [eos_context_token]
        input_id = torch.LongTensor(input_id).view(1,-1)
        belief_model = model.model.generate(input_ids = input_id, decoder_start_token_id=sos_btoken,pad_token_id=pad_token,eos_token_id=eos_btoken,max_length=128)
        belief_decoded = model.tokenized_decode(belief_model[0])
        belief_decoded, dict = belief_rule(belief_decoded,output)
        print("belief state",belief_decoded)
        one_queried_db_result = data.reader.bspan_to_DBpointer(belief_decoded, ['[food]'])
        # print("dict",dict)
        matnums = data.reader.db.get_match_num({'food':dict})
        if matnums['food']!=0:
            output = data.reader.db.get_match_num({'food':dict},return_entry=True)['food'][0]        
        # print("output1",output)
        # print("one_queried_db_result",one_queried_db_result)
        one_db_text = '<sos_db> ' + one_queried_db_result + ' <eos_db>' 
        one_db_token_id_input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(one_db_text))
        # print(one_db_token_id_input)

        #predict dialog act
        input_id =  da_prefix_token + [sos_context_token] + context_id + [eos_context_token] + one_db_token_id_input
        input_id = torch.LongTensor(input_id).view(1,-1)
        act_model = model.model.generate(input_ids = input_id, decoder_start_token_id=sos_atoken,pad_token_id=pad_token,eos_token_id=eos_atoken,max_length=128)
        act_decoded = model.tokenized_decode(act_model[0])
        print("dialog act",act_decoded)
        #predict system response
        # print(dict)
        input_id =  response_prefix_token + [sos_context_token] + context_id + [eos_context_token]
        input_id = torch.LongTensor(input_id).view(1,-1)
        resp = model.model.generate(input_ids = input_id, decoder_start_token_id=sos_rtoken,pad_token_id=pad_token,eos_token_id=eos_rtoken,max_length=128)
        system_response = model.tokenized_decode(resp[0])
        try :
            delex_response = data.reader.response_fill_val(dict,system_response) #fill slots with values
        except:
            delex_response = system_response
        # print("model system response",delex_response)

        #change response according to action state
        if '[offerbooked]' in act_decoded and 'value' in dict:
            sample_response = random.sample(rules['finalAns'],1)[0]
            delex_response =sample_response.format(**dict)
            per = str(round(float(dict['value'])/20.0,2))
            oz_val = str(round(float(dict['value'])/(rules['oz-gm']*1000),4))
            tsp_val = str(round(float(dict['value'])/(rules['tsp-gm']*1000),2))
            percent = "you will be having {per}% of your daily intake. ".format(per=per)
            oz = "salt value is {oz_val} oz . ".format(oz_val=oz_val)
            tsp = "it is around {tsp_val} tsp . ".format(tsp_val=tsp_val)
            delex_response = delex_response + percent + tsp + oz
            flag =1
            # print("system response",delex_response,percent,tsp,oz)

        if 'foodweight' in act_decoded and 'food' in dict:
            sample_response = random.sample(rules['foodWeight'],1)[0]
            delex_response =sample_response.format(**dict)
            # print("system response",delex_response)
        print("system response",delex_response)
        if flag==1:
            break   

        
        # predict user intent
        input_id =  intent_prefix_token + [sos_context_token] + context_id + [eos_context_token]
        input_id = torch.LongTensor(input_id).view(1,-1)
        intent = model.model.generate(input_ids = input_id, decoder_start_token_id=sos_ictoken,pad_token_id=pad_token,eos_token_id=eos_ictoken,max_length=128)
        # print("user intent",model.tokenized_decode(intent[0]))
        base_response = dialogue_context + delex_response
    return

if __name__ == "__main__":
    evaluate() 