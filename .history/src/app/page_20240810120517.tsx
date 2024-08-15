"use client";
import { Box, Button, Stack, TextField, Typography, CircularProgress } from "@mui/material";
import { useState } from "react";

interface Message {
  role: string;
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hi, I'm Converso. How can I assist you today?",
    },
  ]);
  const [message, setMessage] = useState<string>("");
  const [isTyping, setIsTyping] = useState<boolean>(false);

  const sendMessage = async () => {
    const userMessage: Message = { role: "user", content: message };
    setMessages((prevMessages) => [
      ...prevMessages,
      userMessage,
      { role: "assistant", content: "" },
    ]);
    setMessage("");
    setIsTyping(true);

    const response = await fetch("api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify([...messages, userMessage]),
    });

    if (!response.body) {
      console.error("Response body is null");
      setIsTyping(false);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let result = "";

    const processText = async ({
      done,
      value,
    }: {
      done: boolean;
      value?: Uint8Array;
    }) => {
      if (done) {
        setMessages((prevMessages) => {
          const updatedMessages = [...prevMessages];
          updatedMessages[updatedMessages.length - 1].content = result;
          setIsTyping(false);
          return updatedMessages;
        });
        return;
      }

      const text = decoder.decode(value, { stream: true });
      result += text;
      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        updatedMessages[updatedMessages.length - 1].content = result;
        return updatedMessages;
      });

      reader.read().then(processText);
    };

    reader.read().then(processText);
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4" style={{ backgroundColor: "#000" }}>
      <Typography
        variant="h4"
        align="center"
        gutterBottom
        style={{ color: "#fff", margin: "20px 0" }}
      >
        Converso
      </Typography>

      <Box
        width="100%"
        maxWidth="600px"
        height="80vh"
        display="flex"
        flexDirection="column"
        justifyContent="space-between"
        p={2}
        sx={{ borderRadius: 2, backgroundColor: "#1e1e1e", overflow: "hidden" }}
      >
        <Stack
          direction="column"
          spacing={2}
          overflow="auto"
          flexGrow={1}
          style={{ paddingRight: 10 }} // Ensure proper spacing for scrollbar
        >
          {messages.map((msg, index) => (
            <Box
              key={index}
              display="flex"
              justifyContent={msg.role === "assistant" ? "flex-end" : "flex-start"}
              mb={1}
            >
              <Box
                bgcolor={msg.role === "assistant" ? "#333" : "#007bff"} // Assistant messages in dark gray, user messages in blue
                color={msg.role === "assistant" ? "#fff" : "#fff"} // Text color white for both
                borderRadius={10}
                p={2}
                maxWidth="80%"
                sx={{ position: "relative" }}
              >
                {msg.content}
                {msg.role === "assistant" && isTyping && (
                  <Box
                    sx={{
                      position: "absolute",
                      bottom: 0,
                      left: 0,
                      display: "flex",
                      alignItems: "center",
                    }}
                  >
                    <CircularProgress
                      size={10}
                      sx={{ color: "#fff", marginRight: 1 }}
                    />
                    <Typography variant="body2" color="grey">
                      ...
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          ))}
        </Stack>

        <Stack direction="row" spacing={1} mt={2}>
          <TextField
            placeholder="Ask a question?"
            fullWidth
            variant="outlined"
            size="small"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            sx={{
              "& .MuiOutlinedInput-root": {
                borderRadius: 15,
              },
              "& .MuiInputBase-input": {
                color: "#fff",
              },
              "& .MuiOutlinedInput-notchedOutline": {
                borderColor: "#555",
              },
            }}
          />
          <Button
            sx={{
              backgroundColor: "#555",
              color: "#fff",
              "&:hover": { backgroundColor: "#666" },
              height: "40px",
              minWidth: "40px",
              borderRadius: 15,
            }}
            onClick={sendMessage}
            variant="contained"
          >
            &#x27A4;
          </Button>
        </Stack>
      </Box>
    </main>
  );
}
