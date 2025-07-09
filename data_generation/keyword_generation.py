import requests
import json
import time
import re
 
url = 'https://api.openai.com/v1/chat/completions'
headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

def main():
  f = open('./datasets/gen_query.json', 'r')
  fw = open('./datasets/gen_keyword.json', 'w')

  for line in f:
      line = json.loads(line)
      text = line['text']
      query_dict = line['query_dict']
      for query_type in query_dict.keys():
        if not query_dict[query_type]:
          continue
        
        query = query_dict[query_type]
        line['query_type'] = query_type
        line['query'] = query
        content = f'''
          Assuming you are a data generator, please generate coarse-grained keywords and fine-grained keywords according to the query and context as required. You can refer to the given examples.

         Requirements
          Both coarse-grained keywords and fine-grained keywords must be text fragments from the given context.

          The output must be a JSON dictionary that can be parsed by json.loads().

          The output dictionary should contain:

          coarse_keywords: a list of one or more coarse-grained keywords.

          fine_keywords: a list of lists. Each sublist represents one information point broken down into clauses.

          Coarse-grained keywords are generally entities that best represent the topic of the given context and query.

          Fine-grained keywords must be grouped according to the query. Each sublist corresponds to a key point needed to answer the question. Different queries will produce different fine-grained groupings.

          If there are no suitable keywords in the context, the corresponding output should be an empty list.

          Examples
          Query: Why is the stable supply of coal crucial for downstream coal-consuming enterprises?
          Context: The author believes there are several key points. Stable supply and marketing is the foundation. As coal is the "grain" of national industry, it's closely linked to industries such as electricity, metallurgy, chemical engineering, building materials, and cement. Stable coal supply is the ballast and stabilizer for downstream coal-consuming enterprises. Regardless of market changes, coal producers and sellers should aim for maximum economic benefit while fulfilling their social responsibility and ensuring market supply. Coal producers should strengthen safety management to ensure stable production. Marketers should balance production and sales, make use of sales channels for efficient marketing, and build logistics and reserve bases at ports and hubs, using them as reservoirs to achieve scientific scheduling for seasonal supply and demand. Stable transportation is a guarantee. Rail is the main mode of coal transport and a key to supply stability. Marketers should coordinate with railways on capacity, sales increases, and scheduling. Rail operators should provide stable capacity based on market layout and supply capabilities. Fulfilling contracts is key. Medium- and long-term contracts are a mechanism for market stability. All parties must honor contracts as a sign of credibility. Sales and end users should implement large annual and small monthly contracts to ensure timely delivery and receipt. Sales and transport must cooperate for efficient fulfillment. Government supervision is essential. Contracts alone aren't enough; strict government regulation is needed. Authorities should increase inspection, apply dynamic evaluation and elimination mechanisms, provide guidance, and enforce rules to ensure healthy supply-demand relations.

          Answer:
          {
            "coarse_keywords": [
              "coal"
            ],
            "fine_keywords": [
              ["Stable supply and marketing is the foundation"],
              ["Coal is the grain of national industry", "closely linked to electricity, metallurgy, chemicals, building materials, cement"],
              ["Stable coal supply is the ballast and stabilizer for downstream enterprises"],
              ["Coal producers", "strengthen safety management", "ensure stable production"],
              ["Marketers", "balance production and sales", "use sales channels for efficient marketing", "build logistics and coal reserve bases"],
              ["Stable transportation is a guarantee"],
              ["Rail is the main coal transport mode", "key to stable supply"],
              ["Marketers", "coordinate with railways", "adjust capacity based on sales and market layout"]
            ]
          }
          Query: What are the three generations of AI, and which companies represent each?
          Context: Technical background – AI development trends. First generation: Symbolic AI – symbolic models, rule-based models, perception machines. Second generation: Perceptual intelligence – data-driven statistical learning enabled perception and recognition of text, images, and speech. Third generation: Cognitive intelligence – proposed by Academician Zhang Bo; DARPA’s 2018 AI Next campaign aims to fuse statistical learning with knowledge reasoning, and merge with brain cognitive mechanisms.

          Answer:
          Edit
          {
            "coarse_keywords": [
              "AI technology"
            ],
            "fine_keywords": [
              ["Technical background – AI trends"],
              ["First generation: Symbolic AI", "symbolic models", "rule-based models", "perception machines"],
              ["Second generation: Perceptual intelligence", "data-driven statistical learning", "enabled perception of text, images, speech"],
              ["Third generation: Cognitive intelligence"],
              ["Academician Zhang Bo proposed the concept", "DARPA 2018 launched AI Next plan", "goal: fusion of data statistics and knowledge reasoning"],
              ["Fusion with brain cognitive mechanisms"]
            ]
          }
          Query: {query}
          Context: {text}
          Answer:
      '''
        data = {
                  "model": "gpt-4o",
                  "messages": [{"role": "user", "content": content}]
                }
        try:
            response = requests.post(url, headers=headers, json=data)
            res = json.loads(response.text)
            ans = res['data']['response']['choices'][0]['message']['content']
            matches = re.findall(r"\{.*?\}", ans.replace('\n', ''))
            answer_dict = eval(matches[0])  
            line['fine_keywords'] = answer_dict['fine_keywords']
            line['coarse_keywords'] = answer_dict['fine_keywords']
            fw.write(json.dumps(line, ensure_ascii=False)+'\n')
        except Exception as e:
            print(f"{e}")
    
if __name__ == '__main__':
  main()
