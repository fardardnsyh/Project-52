import { NextRequest, NextResponse } from "next/server";
import { OpenAI } from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import axios from "axios";
import { YoutubeTranscript } from "youtube-transcript";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const openai = new OpenAI({
  apiKey: process.env.OPENROUTER_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY as string,
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

async function embedAndStore(texts: string[], videoId: string) {
  for (const text of texts) {
    const embedding = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: text,
    });

    await index.upsert([
      {
        id: `${videoId}-${Date.now()}`,
        values: embedding.data[0].embedding,
        metadata: { text, source: `youtube-${videoId}` },
      },
    ]);
  }
}

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

  return queryResponse.matches
    .filter((match) => match.metadata?.text)
    .map((match) => match.metadata?.text || "");
}

export async function POST(req: NextRequest) {
  try {
    const { messages, youtubeUrl } = await req.json();
    const userQuery = messages[messages.length - 1].content;

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

    const response = await axios.post(
      "https://api.openrouter.ai/v1/complete",
      {
        model: "mistralai/mistral-nemo",
        messages: [
          { role: "system", content: primer },
          { role: "user", content: augmented_query },
        ],
        temperature: 0.7,
        max_tokens: 150,
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
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
