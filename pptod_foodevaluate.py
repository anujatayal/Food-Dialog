import torch
from transformers import T5Tokenizer
model_path = r'./E2E_TOD/ckpt/small/full_training/epoch_2_dev_e2e_evaluation_inform_0.0_success_0.0_bleu_23.53_combine_score_23.53/'
tokenizer = T5Tokenizer.from_pretrained(model_path)
from E2E_TOD.modelling.T5Model import T5Gen_Model
from E2E_TOD.ontology import sos_eos_tokens
from E2E_TOD.dataclass import MultiWozData
from E2E_TOD.config import Config

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
dialogue_context = "<sos_u> I want to eat pork chop <eos_u>"
dialogue_context = "<sos_u> I want to eat pork chop <eos_u><sos_r> what type of food are you eating? <eos_r><sos_u> it is braised <eos_u><sos_r> what is the quantity of pork chops? <eos_r><sos_u> 100 gms <eos_u>"

data_path_prefix = "./data/multiwoz/data/"
cfg = Config(data_path_prefix)
data = MultiWozData("t5-small", tokenizer, cfg, data_path_prefix, shuffle_mode=True, 
        data_mode='train', use_db_as_input=True, add_special_decoder_token=True, 
        train_data_ratio=0.01)
# print("data",data)

def belief_rule(bspan):
    constraint_dict = data.reader.bspan_to_constraint_dict(bspan)
    print("const",constraint_dict)
    if 'value' in constraint_dict["food"]:
        old_venues = data.reader.db.queryJsons("food", constraint_dict["food"], return_name=True)
        print("old venues",old_venues)
        if len(old_venues)==0:
            old_value = constraint_dict["food"]["value"]
            del constraint_dict["food"]["value"]
            print(constraint_dict)
            print("new cons",constraint_dict)
            venues = data.reader.db.queryJsons("food", constraint_dict["food"], return_name=True)
            print("new venues",venues)
            if len(venues)>=1:
                constraint_dict["food"]['value']=venues[0]
            else:
                constraint_dict["food"]['value']=old_value
            bspan = data.reader.constraint_dict_to_bspan(constraint_dict["food"])
            bspan = "[food]" +bspan
    print("new bspan",bspan)

def evaluate():
    base_response=""
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
        belief_rule(belief_decoded)
        print("belief state",model.tokenized_decode(belief[0]))
        #predict dialog act
        input_id =  da_prefix_token + [sos_context_token] + context_id + [eos_context_token]
        input_id = torch.LongTensor(input_id).view(1,-1)
        act = model.model.generate(input_ids = input_id, decoder_start_token_id=sos_atoken,pad_token_id=pad_token,eos_token_id=eos_atoken,max_length=128)
        print("dialog act",model.tokenized_decode(act[0]))
        #predict system response
        input_id =  response_prefix_token + [sos_context_token] + context_id + [eos_context_token]
        input_id = torch.LongTensor(input_id).view(1,-1)
        resp = model.model.generate(input_ids = input_id, decoder_start_token_id=sos_rtoken,pad_token_id=pad_token,eos_token_id=eos_rtoken,max_length=128)
        system_response = model.tokenized_decode(resp[0])
        print("system response",system_response)
        #predict user intent
        input_id =  intent_prefix_token + [sos_context_token] + context_id + [eos_context_token]
        input_id = torch.LongTensor(input_id).view(1,-1)
        intent = model.model.generate(input_ids = input_id, decoder_start_token_id=sos_ictoken,pad_token_id=pad_token,eos_token_id=eos_ictoken,max_length=128)
        print("user intent",model.tokenized_decode(intent[0]))
        base_response = dialogue_context + system_response
    return

if __name__ == "__main__":
    evaluate() 