import os
import requests
import json
import time
import re
import tiktoken

url = 'https://api.openai.com/v1/chat/completions'
headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

def encode_string_by_tiktoken(content, model_name = "gpt-4o"):
    ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens

def split_paragraph(paragraph, max_length=1200):
    sentences = re.split(r'([。！？])', paragraph)
    if sentences[-1] == '':
        sentences = sentences[:-1]
    sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences)-1, 2)] + [sentences[-1]]
    result = []
    current_chunk = ''
    for sentence in sentences:
        if len(encode_string_by_tiktoken(current_chunk)) + len(encode_string_by_tiktoken(sentence)) > max_length:
            result.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence
    if current_chunk:
        result.append(current_chunk.strip())
    return result


def main():
    fw = open('./datasets/gen_query.json', 'w')
    dir_path = './datasets/docdata/offline_parse_li_native_pdf'

    for path in os.listdir(dir_path):
        if not path.endswith('.pdf.document'):
            continue

        doc = json.load(open(os.path.join(dir_path, path)))
        all_text = ''
        for page in doc:
            all_text += page['text']
    
        context = split_paragraph(all_text.strip())

        for text in context:
            line = {}
            line['metadata'] = page['metadata']
            line['text'] = text
            
            content = f'''
                Assuming you are a data generator, please construct four types of queries as required for each given context. You can refer to the given examples.

                ###Requirements###
                1. The queries should be in line with human style and independent of each other;
                2. The queries should be clear, specific and detailed, without vague references such as pronouns (such as "this", "it", etc);
                3. The queries should be able to derive answers from the given content, and the corresponding content can be retrieved through the query;
                4. If a certain type cannot obtain a query that meets the above requirements, the output of the corresponding query type is empty.
                5. The output can only be a json dictionary, which can be parsed by json.loads(). 

                ###Query Type Definition###
                Factual: Seeking specific, clear facts or evidence. Example: When was the Beijing Olympics held? Where is the capital of the United States?
                Analytical: Seeking analytical explanations or summaries of specific concepts, terms, or phenomena. Example: Why is the earth warming? What are the advantages of renewable energy?
                Comparative: Seeking comparisons of information in different dimensions. Example: Which is more developed, Japan or South Korea? What are the differences between Western medicine and traditional Chinese medicine in treating chronic diseases?
                Tutorial: Seeking the steps to perform a task or process. Example: How to get a driver's license? What are the steps to install TensorFlow?

                ###Output Format###
                {{"Factual":"", "Analytical":"", "Comparative":"", "Tutorial":""}}

                ###Examples###
                Context:
                    Company Profile
                    Hangzhou Tiankuan Technology Co., Ltd. (hereinafter referred to as Tiankuan) was established in 2007 and is headquartered in Hangzhou. It is a well-known high-tech enterprise in China.
                    Tiankuan has long focused on the fields of smart grids, cloud computing, and informatization construction in the military industry, providing customers with exclusive solutions including cloud infrastructure construction and maintenance, next-generation communication protection systems, and industry-specific mobile internet applications.
                    Tiankuan has always been customer-oriented, winning the respect and trust of clients through advanced technological concepts and strong technical capabilities. Currently, it maintains long-term and friendly business relationships with world Fortune 500 companies such as State Grid, China Mobile Communications Group, Europe’s O2 Mobile Group, and Alibaba Group. Domestically, the company’s operations have expanded to nearly 20 provinces and municipalities across China. Tiankuan also generates continuous technical service revenue in Germany and Spain, laying a solid foundation for the expansion of its overseas business.
                    Tiankuan insists on technological innovation and has been recognized as a national high-tech enterprise and a Zhejiang provincial software company. It has obtained a Level 2 qualification for computer system integration, as well as honors such as Hangzhou Famous Brand Product and Hangzhou Famous Trademark. It also holds multiple national patents, software product registrations, and copyrights. Its R&D institutions are well-established; the R&D center has passed CMMI Level 3 certification and has been named a municipal high-tech R&D center in Hangzhou.
                    Looking ahead, Tiankuan will continue to enhance its brand influence and strive to promote the development of the global information service industry, aiming to become a world-class provider of specialized information technology solutions.

                    Answer:

                    {
                    "Factual": "In which year was Hangzhou Tiankuan Technology Co., Ltd. established?",
                    "Analytical": "What is the domestic market coverage of Hangzhou Tiankuan Technology Co., Ltd.?",
                    "Comparative": "",
                    "Tutorial": ""
                    }

                Context: {text}
                Answer:
            '''
            data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": content}],
                    }
         
            try:
                response = requests.post(url, headers=headers, json=data)
                res = json.loads(response.text)
                ans = res['data']['response']['choices'][0]['message']['content']
                matches = re.findall(r"\{.*?\}", ans.replace('\n', ''))
                answer_dict = eval(matches[0])  
                line['query_dict'] = answer_dict
                fw.write(json.dumps(line, ensure_ascii=False)+'\n')
            except Exception as e:
                print(f"{e}")
        
if __name__ == '__main__':
    main()