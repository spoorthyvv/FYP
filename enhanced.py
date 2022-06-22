import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


import transformers
transformers.logging.set_verbosity_error()


model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text
  





with open('converted.txt','r') as file:
    context = file.read()


from sentence_splitter import SentenceSplitter, split_text_into_sentences

splitter = SentenceSplitter(language='en')

sentence_list = splitter.split(context)

paraphrase = []

for i in sentence_list:
  a = get_response(i,1)
  paraphrase.append(a)
  
paraphrase2 = [' '.join(x) for x in paraphrase]
# Combines the above list into a paragraph
paraphrase3 = [' '.join(x for x in paraphrase2) ]
paraphrased_text = str(paraphrase3).strip('[]').strip("'")


print(paraphrased_text)
















































































# from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# from sentence_splitter import SentenceSplitter, split_text_into_sentences
# import transformers
# import torch
# from transformers import *

# model_name = 'tuner007/pegasus_paraphrase'
# torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
# model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# def get_response(input_text,num_return_sequences):
# 	batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
# 	translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
# 	tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
# 	return tgt_text
  




# with open('converted.txt','r') as file:
#     p = file.read()
#     splitter = SentenceSplitter(language='en')
#     sentence_list = splitter.split(p)
#     paraphrase = []
#     # Takes the input paragraph and splits it into a list of sentences
#     for i in sentence_list:
#         a = get_response(i,1)
#         paraphrase.append(a)	
                
#     paraphrase2 = [' '.join(x) for x in paraphrase]
#     # Combines the above list into a paragraph
#     paraphrase3 = [' '.join(x for x in paraphrase2) ]
#     # paraphrase4 = [' '.join(x for x in paraphrase3) ]
#     # paraphrase5 = [' '.join(x for x in paraphrase4) ]
#     # paraphrase6 = [' '.join(x for x in paraphrase5) ]
#     # paraphrase7 = [' '.join(x for x in paraphrase6) ]
#     paraphrased_text = str(paraphrase3).strip('[]').strip("'")

#     print(paraphrased_text)


            
# # model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
# # tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

# # def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
# #   # tokenize the text to be form of a list of token IDs
# #   inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
# #   # generate the paraphrased sentences
# #   outputs = model.generate(
# #     **inputs,
# #     num_beams=num_beams,
# #     num_return_sequences=num_return_sequences,
# #   )
# #   # decode the generated sentences using the tokenizer to get them back to text
# #   return tokenizer.batch_decode(outputs, skip_special_tokens=True)



# # sentence = "Although the terms data and information are often used interchangeably, this term has distinct meanings. In some popular publications, data are sometimes said to be transformed into information when they are viewed in context or in post-analysis.[3] However, in academic treatments of the subject data are simply units of information. Data are used in scientific research, businesses management (e.g., sales data, revenue, profits, stock price), finance, governance (e.g., crime rates, unemployment rates, literacy rates), and in virtually every other form of human organizational activity (e.g., censuses of the number of homeless people by non-profit organizations).In general, data are atoms of decision making: they are the smallest units of factual information that can be used as a basis for reasoning, discussion, or calculation. Data can range from abstract ideas to concrete measurements, even statistics. Data are measured, collected, reported, and analyzed, and used to create data visualizations such as graphs, tables or images. Data as a general concept refers to the fact that some existing information or knowledge is represented or coded in some form suitable for better usage or processing. Raw data unprocessed data is a collection of numbers or characters before it has been cleaned and corrected by researchers. Raw data needs to be corrected to remove outliers or obvious instrument or data entry errors (e.g., a thermometer reading from an outdoor Arctic location recording a tropical temperature). Data processing commonly occurs by stages, and the processed data from one stage may be considered the raw data of the next stage. Field data is raw data that is collected in an uncontrolled in situ environment. Experimental data is data that is generated within the context of a scientific investigation by observation and recording."
# # get_paraphrased_sentences(model, tokenizer, sentence, num_beams=1, num_return_sequences=1)
