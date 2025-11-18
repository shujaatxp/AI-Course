using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using OllamaSharp;
using OllamaSharp.Models.Chat;
using OllamaChatRole = OllamaSharp.Models.Chat.ChatRole;

class Program
{
    static async Task Main()
    {
        var ollama = new OllamaApiClient(new Uri("http://localhost:11434"));
        ollama.SelectedModel = "qwen3:4b";

        Console.WriteLine("💬 Q&A Assistant - type 'exit' to quit");

        // Load Q&A file
        string filePath = "C:\\Personal\\AICourse\\MLNet\\QAOLlama\\qa_data.txt"; // your text file with questions & answers
        if (!File.Exists(filePath))
        {
            Console.WriteLine("❌ File not found.");
            return;
        }

        var lines = File.ReadAllLines(filePath);

        // Split file into chunks of ~5 lines
        var chunks = new List<string>();
        int chunkSize = 5;
        for (int i = 0; i < lines.Length; i += chunkSize)
        {
            chunks.Add(string.Join("\n", lines.Skip(i).Take(chunkSize)));
        }

        while (true)
        {
            Console.Write("\nYou: ");
            string input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input) || input.Equals("exit", StringComparison.OrdinalIgnoreCase))
                break;

            // Find top 2 chunks based on keyword matches
            var topChunks = chunks
                .Select(c => new { Chunk = c, Score = CountMatches(c, input) })
                .OrderByDescending(x => x.Score)
                .Take(2)
                .Select(x => x.Chunk);

            string context = string.Join("\n", topChunks);

            // Ask Ollama to answer using context
            var chatRequest = new ChatRequest
            {
                Messages = new List<Message>
                {
                    new Message
                    {
                        Role = OllamaChatRole.System,
                        Content = "You are an assistant. Answer questions based ONLY on the provided Q&A context. Do not make up answers."
                    },
                    new Message
                    {
                        Role = OllamaChatRole.User,
                        Content = $"Context:\n{context}\n\nQuestion: {input}"
                    }
                }
            };

            string answer = "";
            try
            {
                await foreach (var chunk in ollama.ChatAsync(chatRequest))
                {
                    if (chunk?.Message?.Content != null)
                        answer += chunk.Message.Content;
                }

                Console.WriteLine($"\n[ASSISTANT]: {answer.Trim()}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n❌ Error: {ex.Message}");
            }
        }

        Console.WriteLine("\n👋 Goodbye!");
    }

    // Simple keyword matching
    static int CountMatches(string text, string query)
    {
        int count = 0;
        var words = query.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        foreach (var word in words)
        {
            if (text.IndexOf(word, StringComparison.OrdinalIgnoreCase) >= 0)
                count++;
        }
        return count;
    }
}
