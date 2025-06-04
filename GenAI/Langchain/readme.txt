
ConversationBufferMemory = This memory allows for storing of messages and then extracts the messages in a variable

ConversationBufferWindowMemory = This memory keeps a list of the interactions of the conversation over time. It only uses the last k interactions

ConversationTokenBufferMemory = This memory keeps a buffer of recent transactions in memory, and uses token length rather than number of interactions 
                                to determine when to flush interactions.

ConversationSummaryMemory = This memory creates a summary of the conversation over time.

vector data memory = Stores text ( from conversation or elsewhere) in a vector database and retrieves the most relevant block of text

Entity memories = Using a LLM, it remembers details about specific entities.

you can also use multiple memories at one time.

Conversation memory + Entity memory to recall individuals



Embeddings: -

Creates numerical representation of the text. 

This captures the semantic meaning of the text.

Pieces of text with similar content will have similar vectors.

This helps compar pieces of text in vector spaces.



Vectorstores or Vector Databases:-

A vector database is a way to store these vector representations that we created above.

The way we create vector database is we populate it with chunks of text coming from incoming documents.

When we get a big incoming document, we first going to break it into a smaller chunks.

It helps create pieces of text which are smaller than documents.

we may not able to pass the whole document to language model, so we want to create these small chunks, so we can only pass the most relevant ones.

We then create embeddings for each of these chunks and then we store in a vector database.

This what happens when we create the index, now we got this index we can use it during runtime to find the pieces of text most relevant to an incoming query.

When a query come in, we first create embedding for that query, then we compare it to all the vectors in vector database, and we pick the n most similar.

These are then returned, and we pass those in the prompt to the language model to get back a final answer.




How can we use above embeddings for our document search ie chatbot

will ise retrieval method from vector store

retriever - Generic interface that can be underpinned by ny method that takes in a query and returns documents.

we want to do text generation and want to return a natural language response, we are going to import a language mode

Then combine documents and pass that documents to LLM model with query.









What is stuff method: -

Simplest method, you simply stuff all data into the prompt as context to pass to language model and get back one response.

pros:- It makes a single call to LLM. The LLM has access to all data at once.

Cons:- LLM have a context length, and for large documents or many documents this will not work as it will result in a prompt longer than contextual length.

Other methods:-

1. Map_reduce: - This bascially takes all the chunks, passes them along with question to a large language model, get backs response, and then uses another

language model call to summarize all of the individual responses into a final answer.

It is powerful as it can operate over any number of documents and it's also really powerful you can do individual questions in parallel

But it does takes lot more calls and treats all the documents as independent, which may not always be most desired thing.

2. Refine :- is used to loop over many documents. But it actually does it iteratively. It builds upon the answer from previous documents.

So this is really good for combining information and building up an answer over time. It will generally lead to longer answers.

And it's also not as fast because now the calls aren't independent. They depend on results of previous calls.

Its takes longer time and takes just as many calls as Map_reduce.

3. Map_rerank :- experimental method where you do a single call to the large language model for each documents, and you also ask it to return a score.

and then you select the highest score. This realies on language model to know what the score should be.

We often have to tell it should be high score if it's relevant to the document and really refines the instructions there.

Similar to Map_reduce, all the call are independents, can batch them and its relatively fast, but again you are making a bunch of large language model calls.

So it will be expensive.

Most common method
1. Stuff documents
2. Map_reduce :- Eg summarization