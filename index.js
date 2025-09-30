import express from "express";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import cors from "cors";

dotenv.config();
const app = express();   
const port = 3000;

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.use(cors());
app.use(express.json());

app.post("/chat", async (req, res) => {
  try {
    const userMessage = req.body.message;
    console.log("User asked:", userMessage); // log input

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [{ role: "user", parts: [{ text: userMessage }] }],
    });

    console.log("Gemini response:", response); // log raw response
    res.json({ reply: response.text || "⚠ No reply from Gemini" });
  } catch (error) {
    console.error("Error in Gemini API:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(port, () => {
  console.log(console.log(`✅ Server running on http://localhost:${port}`)
  )
});