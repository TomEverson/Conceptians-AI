import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { HuggingFaceInferenceEmbeddings } from "langchain/embeddings/hf";
import { FaissStore } from "langchain/vectorstores/faiss";
import { TextLoader } from "langchain/document_loaders/fs/text";
import "dotenv/config";

const key = process.env.HUGGING_FACE;

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: key,
});

//Loaded All The Docs
const loader = new DirectoryLoader("./data", {
  ".txt": (path) => new TextLoader(path),
});
const docs = await loader.load();
console.log("Doc innit");

//Stored To Faiss
const vectorStore = await FaissStore.fromDocuments(docs, embeddings);
console.log("Vector DB Loaded");

//Vector DB saved
vectorStore.save("./faiss/");
console.log("Vector DB saved");
