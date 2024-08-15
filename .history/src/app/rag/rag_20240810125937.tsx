import { NextResponse } from "next/server";
import { OpenAI } from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

const openai = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY as string,
});

const index = pinecone.Index("chat");

async function queryPinecone(query: string) {
  const q_embedding = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: query,
  });

  const queryResponse = await index.query({
    vector: q_embedding.data[0].embedding,
    topK: 3,
    includeMetadata: true,
  });

  return queryResponse.matches.map((match) => match.metadata.text);
}

export async function POST(req) {
  // const {messages,targetLanguage} =  await req.json();
  // console.log(messages,targetLanguage);
  const { messages } = await req.json();
  const userQuery = messages[messages.length - 1].content;

  const releventContext = await queryPinecone(userQuery);

  return;
}
