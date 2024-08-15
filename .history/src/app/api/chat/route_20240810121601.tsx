import { NextResponse, NextRequest } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPEN_AI_API,
});

const systemPrompt = `
  - Assist users of a platform designed for Full Stack Developer interviews, including coding challenges, mock interviews, and feedback.
  - Help with account issues, interview scheduling, technical support for platform features, and interpreting feedback from mock interviews.
  - Provide information on Full Stack Development concepts (front-end and back-end) and best practices for SWE interviews.
  - Maintain a professional, empathetic tone; provide clear, concise answers; and explain technical terms when necessary.
  - Escalate complex issues or those requiring human intervention to a human support representative.
  
  Your goal is to provide accurate, assist with common inquiries, helpful, and timely assistance for users on a platform dedicated to Full Stack Developer interviews for SWE jobs.
`;

export async function POST(req: NextRequest) {
  try {
    const data = await req.json();
    // Example call to OpenAI's completion endpoint
    const response = await openai.chat.completions.create({
      messages: [
        {
          role: "system",
          content: systemPrompt,
        },
        ...data,
      ],
      model: "gpt-4",
      stream: true,
    });

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        try {
          for await (const chunk of response) {
            const content = chunk.choices[0]?.delta?.content || "";
            if (content) {
              const text = encoder.encode(content);
              controller.enqueue(text);
            }
          }
        } catch (err) {
          controller.error(err);
        } finally {
          controller.close();
        }
      },
    });

    return new NextResponse(stream);
  } catch (error) {
    console.error("Error handling POST request:", error);
    return NextResponse.error();
  }
}
