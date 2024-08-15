import { NextRequest, NextResponse } from "next/server";
import { OpenAI } from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import axios from "axios";

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

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json();
    const userQuery = messages[messages.length - 1].content;

    const releventContext = await queryPinecone(userQuery);

    const primer = `You are a personal assistant. Answer any questions I have about the Youtube Video provided.`;

    const augmented_query = `${releventContext.join(
      "\n"
    )}\n---------\nquestion:\n${userQuery}`;

    const response = await axios.post(
      process.env.API_URL as string, // Ensure this is correctly defined in your environment variables
      {
        model: "mistralai/mistral-nemo",
        messages: [
          { role: "system", content: primer },
          { role: "user", content: augmented_query },
        ],
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.API_KEY}`, // Ensure this is correctly defined in your environment variables
          "Content-Type": "application/json",
        },
      }
    );

    return NextResponse.json(response.data);
  } catch (error) {
    console.error("Error:", error);
    return NextResponse.json(
      { error: "An error occurred while processing your request." },
      { status: 500 }
    );
  }
}
