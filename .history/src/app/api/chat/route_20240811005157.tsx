import { NextRequest, NextResponse } from "next/server";
import axios from "axios";
import { Pinecone } from "@pinecone-database/pinecone";
import { YoutubeTranscript } from "youtube-transcript";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import OpenAI from "openai";
import { HfInference } from "@huggingface/inference";

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY as string,
});

const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
});

const index = pinecone.Index("chat");

async function getYoutubeTranscript(videoId: string) {
  try {
    const transcript = await YoutubeTranscript.fetchTranscript(videoId);
    return transcript.map((item) => item.text).join(" ");
  } catch (error) {
    console.error("Error fetching YouTube transcript:", error);
    throw error;
  }
}

async function splitText(text: string) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 100,
  });
  return await splitter.splitText(text);
}

const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

async function getEmbedding(text: string): Promise<number[]> {
  const response = await hf.featureExtraction({
    model: "sentence-transformers/nli-bert-large",
    inputs: text,
  });

  // Ensure the response is a number array
  if (
    Array.isArray(response) &&
    response.every((item) => typeof item === "number")
  ) {
    return response;
  } else {
    throw new Error("Unexpected embedding format");
  }
}

async function embedAndStore(texts: string[], videoId: string) {
  for (const text of texts) {
    const embedding = await getEmbedding(text);

    await index.upsert([
      {
        id: `${videoId}-${Date.now()}`,
        values: embedding,
        metadata: { text, source: `youtube-${videoId}` },
      },
    ]);
  }
}

async function queryPinecone(query: string) {
  const q_embedding = await getEmbedding(query);

  const queryResponse = await index.query({
    vector: q_embedding,
    topK: 3,
    includeMetadata: true,
  });

  return queryResponse.matches
    .filter((match) => match.metadata?.text)
    .map((match) => match.metadata?.text || "");
}

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json();
    const userQuery = messages[messages.length - 1].content;
    const youtubeUrl = "https://www.youtube.com/watch?v=w8M76fuyn8o&t=359s";

    // Process YouTube URL if provided
    if (youtubeUrl) {
      const videoId = new URL(youtubeUrl).searchParams.get("v");
      if (!videoId) throw new Error("Invalid YouTube URL");

      const transcript = await getYoutubeTranscript(videoId);
      const splitTexts = await splitText(transcript);
      await embedAndStore(splitTexts, videoId);
    }

    const relevantContext = await queryPinecone(userQuery);

    const primer = `You are a personal assistant. Answer any questions I have about the Youtube Video provided.`;

    const augmented_query = `${relevantContext.join(
      "\n"
    )}\n---------\nquestion:\n${userQuery}`;

    const completion = await openai.chat.completions.create({
      model: "mistralai/mistral-nemo",
      messages: [
        { role: "system", content: primer },
        { role: "user", content: augmented_query },
      ],
      temperature: 0.7,
      max_tokens: 150,
    });

    return NextResponse.json(completion.choices[0].message.content);
  } catch (error) {
    console.error("Error:", error);
    return NextResponse.json(
      { error: "An error occurred while processing your request." },
      { status: 500 }
    );
  }
}
