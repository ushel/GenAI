ollama ps
ollama stop llama3.2
ollama run llama3.2

tracking is done using langsmith
ollama model used downloaded on local system so no cost
can check cost of per token of openai using langsmith

API for Deployment...

swager ui 
langserve
                                                   ------> llama2 
web app     ---------> API's --------------> Route ------> openai
mobile app                                         ------> claude
                                                   ------> etc

client.py contains webapp or mobile app

for tracking of env file, if cached then use
git rm .env --cached
git add .
git commit -m "message"
git push origin main



load source --> load --> Transform(chunks) --> Embed(openAi embeddings) --> vector store 

query --> embed --> similarity search --> vector store --> Retrive most similar


using retriver (backbone is vector store)

load source --> load --> Transform(chunks) --> Embed(openAi embeddings) --> vector store 

promopts --> Chain and Retriver (using LLM models). (stuff_document_chain)

user --> retriver --> vectorstore
                  --> LLM model --> promopt

                 (using stuff document chain)






                 arxiv    wiki    pdfs

                         LLM  

                       Response

Here you have multiple data sources and you want to integrate all them as a wrapper so that you will 
able to implement your entire Q&A solution.

Tools  -> Toolkits and create wrapper on Toolkits
Agents

here we have dependencies on arxiv, wiki and pdfs platforms so can use all these platforms as seperate tools so that i can asked different 
questions to them along with i also have my customized pdf custom data which will be inform of vector embeddings.

what we will do is wrapped i up with the help of toolkit and with help of agent will be able to execute Q&A search

To print output of response vertically...

import json

print(json.dumps(response.model_dump(), indent=2))



git branch llmb7  - create new branch

git checkout llmb3 - select branch

git branch  - get names of branch

git branch -D llmb5 - delete the branch llmb6
