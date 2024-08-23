# !pip install transformers
import transformers
# !pip install transformers --upgrade
# !pip install protobuf --upgrade
import torch
import json
import sys


from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline, T5ForConditionalGeneration, \
    T5Tokenizer, AutoModelForSequenceClassification, GPT2ForSequenceClassification, GPT2ForTokenClassification


## QA models:

def roberta_pred(question, context):
    model_name = "deepset/roberta-base-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    result = nlp(question=question, context=context)
    # if score is NOT good enough, aks for more context
    if result['score'] < 0.2:
        result = 'Please provide more context'
    # else return the answer
    else:
        result = result['answer']
    print(result)
    return result

def bart_pred(question, context):
    model_name = 'valhalla/bart-large-finetuned-squadv1'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    result = nlp(question=question, context=context)
    # if score is NOT good enough, aks for more context
    if result['score'] < 0.2:
        result = 'Please provide more context'
    # else return the answer
    else:
        result = result['answer']
    print(result)
    return result

## Summarization models:

def bart_large_cnn_sum(text):
    model_name = 'facebook/bart-large-cnn'
    # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = pipeline('summarization', model=model_name, tokenizer=model_name)
    result = nlp(text)
    print(result)
    return result

def t5_base_cnn_sum(text):
    model_name = 'flax-community/t5-base-cnn-dm'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    nlp = pipeline("summarization", model=model, tokenizer=tokenizer)
    res = nlp(text)
    print(res)
    return res

## Sequence Classification models:

def gpt2_DialogRPT_seq_class(text):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialogRPT-updown")
    model = GPT2ForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    res = nlp(text)
    print(res)
    return res

## DialoGPT (perfect chat-bot)

from transformers import AutoModelForCausalLM, AutoTokenizer

def dialogpt_medium(user_input, chat_history=None):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    input_ids = torch.cat([chat_history, user_input_ids], dim=-1) if chat_history is not None else user_input_ids
    output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

if __name__ == '__main__':
    data = json.loads(sys.argv[1])
    method = data['method']
    params = data['params']

    if method == 'dialogpt_medium':
        user_input = params['user_input']
        chat_history = params['chat_history']
        result = dialogpt_medium(user_input, chat_history)
        print(result)

# main to test
if __name__ == '__main__':
    sequence_to_classify = "Hello, my dog is cute"
    question = "What is treatment for Fungal infection?"
    context = "In humans, fungal infections occur when an invading fungus takes over an area of the body and is too much for the immune system to handle. Fungi can live in the air, soil, water, and plants. There are also some fungi that live naturally in the human body. Like many microbes, there are helpful fungi and harmful fungi.The name of disease is Fungal infection is an illness when you have itching  skin_rash  nodal_skin_eruptions  dischromic _patches . You must bath twice use detol or neem in bathing water keep infected area dry use clean cloths";
    # print(roberta_pred(question, context))
    # print(bart_pred(question, context))
    text = """summarize: Donald Trump, the 45th President of the United States, is a polarizing
    figure who has left an indelible mark on American politics. Known for his
    larger-than-life personality, Trump's presidency was characterized by
    controversial policies, fiery rhetoric, and a penchant for unconventional
    communication through social media.During his time in office, Trump pursued
    an \"America First\" agenda, aiming to prioritize the interests of the United
    States in areas such as trade, immigration, and foreign policy. His
    administration implemented significant tax cuts, deregulation measures, and
    pursued a more assertive stance on international trade agreements.Trump's
    approach to governance often drew both fervent support and vehement opposition.
    Supporters praised his efforts to revitalize the economy, prioritize national
    security, and challenge traditional political norms. Critics, on the other hand,
    raised concerns about his handling of sensitive issues, including immigration,
    climate change, and racial tensions.Beyond policy, Trump's leadership style
    and unfiltered communication drew considerable attention. His prolific use of
    Twitter became a defining characteristic of his presidency, allowing him to
    directly communicate with his base and express his thoughts, often generating
    controversy and media frenzy.Trump's presidency was not without challenges
    and controversies, including investigations into Russian interference in the
    2016 election and subsequent impeachment proceedings. These events further
    deepened the divisions within the country and fueled passionate debates about
    the state of democracy and the role of the presidency.Love him or hate him,
    Donald Trump's impact on American politics and public discourse cannot be denied.
    His presidency left a lasting impression on the nation and continues to shape
    the political landscape as the United States moves forward."""
    # print(bart_large_cnn_sum(text))
    # print(t5_base_cnn_sum(text))
    # print(gpt2_DialogRPT_seq_class(sequence_to_classify))
    chat_history = None
    while True:
        user_input = input(">> User: ")
        chat_history = dialogpt_medium(user_input, chat_history)
        print("DialoGPT: {}".format(chat_history))

