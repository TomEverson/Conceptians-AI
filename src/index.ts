import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RetrievalQAChain, loadQARefineChain } from "langchain/chains";
import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAI } from "langchain";
import "dotenv/config";

const key = process.env.HUGGING_FACE;

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: key,
});

const model = new OpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const vectorStore = await FaissStore.load("./faiss/", embeddings);
console.log("Vector DB Loaded");

//Find Similar Docs
const docs = await vectorStore.similaritySearch("Best College in Cali", 1);
console.log("Docs Found");

//New Memoryvector Store
const newVectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
console.log("Memory Vectore Created");

//Search for the most similar document
const chains = new RetrievalQAChain({
  combineDocumentsChain: loadQARefineChain(model),
  retriever: newVectorStore.asRetriever(),
  verbose: true,
});

//Make A Response
const res = await chains.call({ query: "Best College in Cali" });

console.log(res);
